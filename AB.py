"""
Optimized AQA (Attention Query Alignment), reference-free.

What changed vs. the previous script
------------------------------------
SPEED
  * All (head, position) patches of a layer run in ONE batched forward
    (chunks of CHUNK rows) instead of one batch-1 forward per patch.
    ~1,900 forwards/example -> ~25 batched forwards (Llama, 10 query tokens).
  * No CPU copies of activations; clean head activations stay on GPU.
  * Only the last position's hidden state is projected, in fp32, onto the
    target-token row of lm_head: no [B,S,V] logits tensor (OOM on Gemma) and
    no bf16 logit quantization.
  * Model/tokenizer/hooks are loaded ONCE per model (not once per dataset).
  * Examples whose corruption does not move the target logit are skipped
    BEFORE any patching.
  * Resume support: finished (model, dataset, control) cells are skipped.

CORRECTNESS
  * Query span via offset_mapping on the LAST "Query:" marker; BOS preserved
    under truncation; span sanity check recorded.
  * Every batch contains its own no-patch corrupted row and clean row, so the
    baseline/clean logits come from the SAME kernel/batch size as the patches.
  * Scores are UNCLIPPED fraction-of-gap recovered (clipped version also saved)
    and circuits use random tie-breaking, so ties cannot make circuits identical
    across examples by construction.
  * Head-circuit Jaccard is reported with a random-head baseline.
  * head_dim is taken from the o_proj input width (correct for Gemma-2 where
    head_dim != hidden/num_heads).
  * Gemma-2 final-logit softcapping is applied; Gemma runs with eager attention.
  * All three controls are implemented:
      actual         : random-token corruption, target = clean greedy token
      random_target  : random-token corruption, target = random vocab token
      shuffled_query : permute identities of the selected query tokens
                       (same positions/seed as `actual`), target = clean greedy
"""
import os, gc, ast, math, time, random, traceback
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================ CONFIG
MODELS = {
    "llama": {"model_name": "meta-llama/Llama-3.1-8B", "layers": [15, 18, 20, 22, 25]},
    "qwen":  {"model_name": "Qwen/Qwen2.5-7B",          "layers": [13, 15, 18, 20, 22]},
    "gemma": {"model_name": "google/gemma-2-9b",        "layers": [19, 23, 26, 29, 33],
              "attn_impl": "eager"},
}
DATASETS = {
    "squad":    {"test_csv": "dataset/squad_3_column_test.csv",    "schema": "generic"},
    "ms_marco": {"test_csv": "dataset/ms_marco_3_column_test.csv", "schema": "generic"},
    "qtsumm":   {"test_csv": "dataset/qtsumm_test.csv",            "schema": "qtsumm"},
}
CONTROLS = ["actual", "random_target", "shuffled_query"]

NUM_ROWS = 50                 # None = full
MAX_LENGTH = 512
CORRUPT_FRACTION = 0.40
TOP_K_PERCENT = 0.50          # kappa for head circuits
CHUNK = 64                    # patch rows per forward (lower if OOM)
MIN_ABS_GAP = 1.0             # skip example if |clean - corrupt| target logit < this
SKIP_BLANK_TARGET = True      # skip if first generated token is whitespace-only
SEED = 42
RESUME = True
SINGLE_GPU = True            # fail loudly instead of silently offloading layers to CPU
OUTPUT_DIR = "aqa_optimized_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ============================================================ PROMPTS (no reference answer)
def format_generic(document, question):
    return f"Context: {str(document)[:1000]}\n\nQuery: {question}\n\nAnswer:"

def format_qtsumm(table_raw, query):
    try:
        t = ast.literal_eval(table_raw)
    except (ValueError, SyntaxError):
        t = {"title": "", "headers": [], "data": []}
    md = ""
    if t.get("title"):
        md += f"### {t['title']}\n\n"
    hdr = t.get("headers", [])
    if hdr:
        md += "| " + " | ".join(str(h) for h in hdr) + " |\n"
        md += "|" + "|".join(["---"] * len(hdr)) + "|\n"
    for r in t.get("data", []):
        md += "| " + " | ".join(str(c) for c in r) + " |\n"
    return f"Table Data:\n{md}\nQuery: {query}\n\nSummary:"

def build_prompt(schema, row, q):
    return format_qtsumm(row["table"], q) if schema == "qtsumm" else format_generic(row["document"], q)

def get_question(schema, row):
    return str(row["query"]) if schema == "qtsumm" else str(row["question"])


# ============================================================ QUERY SPAN
def find_query_char_span(prompt):
    qs = prompt.rfind("Query:")
    if qs < 0:
        raise ValueError("no Query: marker")
    qs += len("Query:")
    ends = [i for i in (prompt.find("\n\nAnswer:", qs), prompt.find("\n\nSummary:", qs)) if i >= 0]
    if not ends:
        raise ValueError("no Answer:/Summary: marker")
    qe = min(ends)
    while qs < qe and prompt[qs].isspace():
        qs += 1
    while qe > qs and prompt[qe - 1].isspace():
        qe -= 1
    return qs, qe

def tokenize_with_query_span(tokenizer, prompt, max_length):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    q0, q1 = find_query_char_span(prompt)
    qpos = [i for i, (s, e) in enumerate(offs) if e > s and max(s, q0) < min(e, q1)]
    if not qpos:
        raise ValueError("empty query span")
    nb = 1 if (tokenizer.bos_token_id is not None and ids[0] == tokenizer.bos_token_id) else 0
    if len(ids) > max_length:                       # keep BOS + the LAST tokens
        start = len(ids) - (max_length - nb)
        ids = ids[:nb] + ids[start:]
        qpos = [p - start + nb for p in qpos if p >= start]
        if not qpos:
            raise ValueError("query truncated away")
    return torch.tensor([ids]), qpos

def span_ok(tokenizer, ids, qpos, question):
    dec = "".join(tokenizer.decode(ids[0, qpos].tolist()).split())
    return dec == "".join(question.split())


# ============================================================ CORRUPTION
def corrupt_ids_fn(ids, qpos, frac, tokenizer, seed, mode, vocab_n):
    rng = random.Random(seed)
    out = ids.clone()
    n = min(len(qpos), max(1, math.ceil(frac * len(qpos))))
    sel = sorted(rng.sample(qpos, n))
    special = set(tokenizer.all_special_ids)
    if mode == "random_token":
        for p in sel:
            orig, c = int(out[0, p]), int(out[0, p])
            for _ in range(50):
                c = rng.randrange(vocab_n)
                if c not in special and c != orig:
                    break
            out[0, p] = c
    elif mode == "shuffle":
        vals = [int(out[0, p]) for p in sel]
        if len(set(vals)) > 1:
            orig = list(vals)
            for _ in range(20):
                rng.shuffle(vals)
                if vals != orig:
                    break
        for p, v in zip(sel, vals):
            out[0, p] = v
    else:
        raise ValueError(mode)
    return out, sel


# ============================================================ HOOKS
class HeadPatcher:
    """Pre-hook on o_proj of monitored layers. mode: off | capture | patch."""
    def __init__(self, model, layers):
        self.model, self.layers = model, list(layers)
        self.n_heads = model.config.num_attention_heads
        self.mode, self.cur = "off", None
        self.captured = {}
        self.pos_idx = self.head_idx = self.vals = None
        self.handles = [
            model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(self._mk(l))
            for l in self.layers
        ]

    def _mk(self, l):
        def hook(mod, args):
            x = args[0]                                   # [B, S, n_heads*head_dim]
            if self.mode == "capture":
                B, S, D = x.shape
                self.captured[l] = x[0].reshape(S, self.n_heads, D // self.n_heads).detach().clone()
                return None
            if self.mode == "patch" and l == self.cur:
                B, S, D = x.shape
                heads = x.reshape(B, S, self.n_heads, D // self.n_heads).clone()
                ok = self.pos_idx >= 0
                b = torch.arange(B, device=x.device)[ok]
                heads[b, self.pos_idx[ok], self.head_idx[ok]] = self.vals[ok].to(heads.dtype)
                return (heads.reshape(B, S, D),)
            return None
        return hook

    def set_patch(self, layer, pos_idx, head_idx, vals):
        self.mode, self.cur = "patch", layer
        self.pos_idx, self.head_idx, self.vals = pos_idx, head_idx, vals

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ============================================================ LOGITS
@torch.no_grad()
def last_hidden(model, ids):
    return model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -1, :]

@torch.no_grad()
def target_logit(model, ids, target_id):
    """fp32 logit of target_id at the last position for every batch row -> [B]."""
    h = last_hidden(model, ids).float()
    z = h @ model.lm_head.weight[target_id].float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    return cap * torch.tanh(z / cap) if cap else z

@torch.no_grad()
def greedy_target(model, ids):
    return int(model.lm_head(last_hidden(model, ids))[0].float().argmax().item())


# ============================================================ AQA FOR ONE EXAMPLE
@torch.no_grad()
def _aqa_example(model, patcher, clean_ids, corrupt_ids, qpos, target_id, layers, chunk):
    dev, H, P = clean_ids.device, patcher.n_heads, len(qpos)

    # reference pass: clean (capturing head outputs) and corrupted, for the gap check
    patcher.mode, patcher.captured = "capture", {}
    zc = target_logit(model, clean_ids, target_id)[0]
    patcher.mode = "off"
    zk = target_logit(model, corrupt_ids, target_id)[0]
    gap0 = float(zc - zk)
    if abs(gap0) < MIN_ABS_GAP:
        return None, gap0
    clean_heads = patcher.captured                       # layer -> [S, H, dh] (GPU)

    combos_h = torch.arange(H, device=dev).repeat_interleave(P)          # head-major
    combos_p = torch.tensor(qpos, device=dev).repeat(H)
    total = H * P

    unclip = np.zeros((len(layers), H), dtype=np.float64)
    clip = np.zeros((len(layers), H), dtype=np.float64)

    for li, l in enumerate(layers):
        rec = torch.zeros(total, device=dev)
        for s in range(0, total, chunk):
            e = min(total, s + chunk)
            B = (e - s) + 2                              # row0: corrupt no-patch, row1: clean
            ids = corrupt_ids.expand(B, -1).clone()
            ids[1] = clean_ids[0]
            pos_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
            head_idx = torch.full((B,), -1, dtype=torch.long, device=dev)
            pos_idx[2:], head_idx[2:] = combos_p[s:e], combos_h[s:e]
            vals = clean_heads[l][pos_idx.clamp(min=0), head_idx.clamp(min=0)]   # [B, dh]
            patcher.set_patch(l, pos_idx, head_idx, vals)
            z = target_logit(model, ids, target_id)
            patcher.mode = "off"
            gap = z[1] - z[0]
            rec[s:e] = (z[2:] - z[0]) / gap if abs(float(gap)) > 1e-6 else 0.0
        r = rec.view(H, P).float().cpu().numpy()
        unclip[li] = r.mean(1)
        clip[li] = np.clip(r, 0.0, 1.0).mean(1)
    return (unclip, clip), gap0


def aqa_example(*args, chunk=CHUNK, **kw):
    """OOM-safe wrapper: halves the chunk and retries."""
    while True:
        try:
            return _aqa_example(*args, chunk=chunk, **kw)
        except torch.cuda.OutOfMemoryError:
            gc.collect(); torch.cuda.empty_cache()
            if chunk <= 8:
                raise
            chunk //= 2
            print(f"[OOM] retrying with chunk={chunk}")


# ============================================================ AGGREGATION
def topk_mask(scores, kappa, rng):
    """scores [N,L,H] -> bool top-k mask with random tie-breaking."""
    N, L, H = scores.shape
    k = max(1, math.ceil(kappa * H))
    x = scores.astype(np.float64) + 1e-9 * rng.random(scores.shape)
    idx = np.argsort(-x, axis=-1)[..., :k]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return mask

def mean_pairwise_jaccard(mask_layer):
    """mask_layer [N,H] bool -> mean Jaccard over all i<j pairs."""
    N = mask_layer.shape[0]
    if N < 2:
        return float("nan")
    A = mask_layer.astype(np.float32)
    inter = A @ A.T
    sz = A.sum(1)
    union = sz[:, None] + sz[None, :] - inter
    J = np.where(union > 0, inter / np.maximum(union, 1e-9), 1.0)
    return float((J.sum() - np.trace(J)) / (N * (N - 1)))


# ============================================================ ONE (model, dataset, control)
def run_pair(model, tokenizer, patcher, model_key, dataset_key, control, layers):
    tag = f"{model_key}_{dataset_key}_{control}"
    summ_path = os.path.join(OUTPUT_DIR, f"aqa_{tag}_summary.csv")
    if RESUME and os.path.exists(summ_path):
        print(f"[skip] {tag} already done")
        return pd.read_csv(summ_path).iloc[0].to_dict()

    schema = DATASETS[dataset_key]["schema"]
    df = pd.read_csv(DATASETS[dataset_key]["test_csv"])
    if NUM_ROWS is not None:
        df = df.iloc[:NUM_ROWS].reset_index(drop=True)

    corr_mode = "shuffle" if control == "shuffled_query" else "random_token"
    dev = model.get_input_embeddings().weight.device
    vocab_n = min(len(tokenizer), model.config.vocab_size)
    special = set(tokenizer.all_special_ids)

    rows, unclip_all, clip_all, kept = [], [], [], []
    printed_tb = False
    t0 = time.time()

    for i in tqdm(range(len(df)), desc=tag):
        row = df.iloc[i]
        question = get_question(schema, row)
        rec = {"row": i, "control": control, "status": "ok"}
        try:
            ids, qpos = tokenize_with_query_span(tokenizer, build_prompt(schema, row, question), MAX_LENGTH)
        except Exception as e:
            rec["status"] = f"span_fail:{e}"; rows.append(rec); continue
        ids = ids.to(dev)
        rec["n_query_tokens"] = len(qpos)
        rec["span_ok"] = span_ok(tokenizer, ids, qpos, question)

        gen = greedy_target(model, ids)
        if gen in special:
            rec["status"] = "special_target"; rows.append(rec); continue
        if SKIP_BLANK_TARGET and not tokenizer.decode([gen]).strip():
            rec["status"] = "blank_target"; rows.append(rec); continue

        target = gen
        if control == "random_target":
            rng = random.Random(SEED * 7 + i)
            for _ in range(100):
                target = rng.randrange(vocab_n)
                if target not in special and target != gen:
                    break
        rec["target_id"], rec["target_text"] = target, tokenizer.decode([target])
        rec["generated_target"] = int(target == gen)

        cids, sel = corrupt_ids_fn(ids, qpos, CORRUPT_FRACTION, tokenizer, SEED + i, corr_mode, vocab_n)
        rec["n_corrupted"] = len(sel)

        try:
            out, gap0 = aqa_example(model, patcher, ids, cids, qpos, target, layers)
        except Exception as e:
            rec["status"] = f"error:{type(e).__name__}:{str(e)[:300]}"
            if not printed_tb:
                printed_tb = True
                print(f"\n[{tag}] FIRST TRACEBACK (row {i}):")
                traceback.print_exc()
            rows.append(rec); continue
        rec["gap"] = gap0
        if out is None:
            rec["status"] = "small_gap"; rows.append(rec); continue

        unclip, clip = out
        unclip_all.append(unclip); clip_all.append(clip); kept.append(i)
        for li, l in enumerate(layers):
            rec[f"layer_{l}_aqa"] = float(unclip[li].mean())
            rec[f"layer_{l}_aqa_clipped"] = float(clip[li].mean())
        rows.append(rec)

    elapsed = time.time() - t0
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(os.path.join(OUTPUT_DIR, f"aqa_{tag}_rows.csv"), index=False)
    print(f"[{tag}] status counts: {rows_df['status'].str.split(':').str[0].value_counts().to_dict()}")
    if "gap" in rows_df and rows_df["gap"].notna().any():
        g = rows_df["gap"].abs().dropna()
        print(f"[{tag}] |gap| quantiles 10/50/90%: "
              f"{g.quantile(.1):.3f} / {g.quantile(.5):.3f} / {g.quantile(.9):.3f}  (threshold {MIN_ABS_GAP})")
    if "target_text" in rows_df:
        print(f"[{tag}] top first-tokens: {rows_df['target_text'].value_counts().head(5).to_dict()}")
    errs = rows_df[rows_df["status"].str.startswith(("error", "span_fail"))]["status"]
    if len(errs):
        print(f"[{tag}] first error: {errs.iloc[0]}")

    summary = {"model": model_key, "dataset": dataset_key, "control": control,
               "n_total": len(df), "n_valid": len(kept), "seconds": round(elapsed, 1),
               "corrupt_fraction": CORRUPT_FRACTION, "kappa": TOP_K_PERCENT,
               "min_abs_gap": MIN_ABS_GAP}

    if kept:
        U = np.stack(unclip_all)                        # [N, L, H]
        C = np.stack(clip_all)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"aqa_{tag}_heads.npz"),
                            unclipped=U, clipped=C, rows=np.array(kept), layers=np.array(layers))
        rng = np.random.default_rng(SEED)
        mask = topk_mask(U, TOP_K_PERCENT, rng)
        rmask = topk_mask(rng.random(U.shape), TOP_K_PERCENT, rng)      # random-head baseline
        for li, l in enumerate(layers):
            per_ex = U[:, li, :].mean(1)
            summary[f"layer_{l}_aqa"] = float(per_ex.mean())
            summary[f"layer_{l}_aqa_sem"] = float(per_ex.std(ddof=1) / math.sqrt(len(per_ex))) if len(per_ex) > 1 else float("nan")
            summary[f"layer_{l}_aqa_clipped"] = float(C[:, li, :].mean())
            summary[f"layer_{l}_head_jaccard"] = mean_pairwise_jaccard(mask[:, li, :])
            summary[f"layer_{l}_head_jaccard_random"] = mean_pairwise_jaccard(rmask[:, li, :])
        summary["AQA"] = float(np.mean([summary[f"layer_{l}_aqa"] for l in layers]))
        summary["AQA_clipped"] = float(np.mean([summary[f"layer_{l}_aqa_clipped"] for l in layers]))
        summary["head_jaccard"] = float(np.mean([summary[f"layer_{l}_head_jaccard"] for l in layers]))
        summary["head_jaccard_random"] = float(np.mean([summary[f"layer_{l}_head_jaccard_random"] for l in layers]))
        print(f"{tag}: AQA={summary['AQA']:.4f} (clipped {summary['AQA_clipped']:.4f}) "
              f"headJ={summary['head_jaccard']:.3f} vs random {summary['head_jaccard_random']:.3f} "
              f"valid={len(kept)}/{len(df)} {elapsed:.0f}s")
    else:
        print(f"{tag}: no valid examples")

    pd.DataFrame([summary]).to_csv(summ_path, index=False)
    return summary


# ============================================================ MAIN
def main():
    all_results = []
    for model_key, cfg in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        kw = {"attn_implementation": cfg["attn_impl"]} if "attn_impl" in cfg else {}
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], torch_dtype=torch.bfloat16,
            device_map=({"": 0} if SINGLE_GPU else "auto"), **kw).eval()
        devs = {str(p.device) for p in model.parameters()}
        print(f"[{model_key}] parameter devices: {devs}  "
              f"GPU mem used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
        patcher = HeadPatcher(model, cfg["layers"])
        try:
            for dataset_key in DATASETS:
                for control in CONTROLS:
                    all_results.append(run_pair(model, tokenizer, patcher, model_key,
                                                dataset_key, control, cfg["layers"]))
        finally:
            patcher.remove()
            patcher.model = None      # patcher held a reference -> model was never freed
            patcher = None
            model = None
            gc.collect()
            torch.cuda.empty_cache()
    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, "grand_aqa_summary.csv"), index=False)


if __name__ == "__main__":
    main()