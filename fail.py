"""
Construct validity experiments for QueryEval
=============================================

Four controlled manipulations that vary whether the model can use the
query, holding everything else fixed.  Each manipulation produces paired
prompts so we can compute within-example deltas for QueryEval, QRF, CCS,
AQA, and output-level metrics (ROUGE, BERTScore, LLM-judge placeholder).

Predictions (to be tested):
  M1 (context swap):      QueryEval(matched) > QueryEval(mismatched)
                          ROUGE(matched)  ~= ROUGE(mismatched)
  M2 (query ablation):    QueryEval(original) >> QueryEval(placeholder)
  M3 (corruption gradient): QueryEval decreases monotonically with rho;
                          ROUGE/BERTScore flat or weakly decreasing.
  M4 (parametric conflict): QueryEval(B) < QueryEval(A), QueryEval(B) < QueryEval(C)
                          Flip rate distinguishes contextual from parametric.

Outputs per (model, dataset) under OUTPUT_DIR/construct_validity/:
  m1_paired.csv
  m2_paired.csv
  m3_gradient.csv
  m4_conflict.csv
  disagreement_cases.csv
  effect_sizes.csv
  dose_response.png
  component_deltas.png

Dependencies: your existing QueryEval pipeline.  This file imports the
shared helpers from the CCS revision module.  Adjust SHARED_MODULE if
your file is named differently.
"""

import os
import gc
import ast
import json
import math
import random
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------
# Import shared helpers from the CCS revision module.
# Adjust the module name to match your file.
# ---------------------------------------------------------------
SHARED_MODULE = "ccs"   # <-- change if your file is named differently

try:
    shared = __import__(SHARED_MODULE)
    MODELS = shared.MODELS
    DATASETS = shared.DATASETS
    build_prompt = shared.build_prompt
    get_original_question = shared.get_original_question
    tokenize_prompt_with_query_span = shared.tokenize_prompt_with_query_span
    corrupt_query_ids = shared.corrupt_query_ids
    find_query_char_span = shared.find_query_char_span
    get_clean_first_generated_token = shared.get_clean_first_generated_token
    forward_logits = shared.forward_logits
    ActivationCacher = shared.ActivationCacher
    BatchedMLPPatcher = shared.BatchedMLPPatcher
    get_mlp_module = shared.get_mlp_module
    get_transformer_layers = shared.get_transformer_layers
    top_k_circuit = shared.top_k_circuit
    circuit_to_bins = shared.circuit_to_bins
    normalized_query_bin = shared.normalized_query_bin
    jaccard = shared.jaccard
    pairwise_mean_jaccard = shared.pairwise_mean_jaccard
    N_POSITION_BINS = shared.N_POSITION_BINS
    MAX_LENGTH = shared.MAX_LENGTH
except Exception as exc:
    raise ImportError(
        f"Could not import shared helpers from '{SHARED_MODULE}'. "
        f"Set SHARED_MODULE at the top of this file to the module that "
        f"defines your QueryEval pipeline. Original error: {exc}"
    )


# ================================================================
# CONFIG
# ================================================================

OUTPUT_DIR = "construct_validity"

# Number of examples per condition.  Keep modest for smoke tests.
NUM_ROWS = 50

# Corruption rates for the gradient experiment (M3).
GRADIENT_RHOS = [0.0, 0.1, 0.2, 0.3]

# Seeds for corruption.
CORRUPTION_SEEDS = [13, 42, 97]

# CCS circuit threshold used in the paired comparisons.
KAPPA = 0.20

# QueryEval composite weights (must match the paper).
QRF_WEIGHT = 1.0 / 3.0
CCS_WEIGHT = 1.0 / 3.0
AQA_WEIGHT = 1.0 / 3.0

# Whether to run the LLM-judge placeholder.  Set False unless you have
# an API key and want to include judge scores.
RUN_LLM_JUDGE = False


# ================================================================
# SMALL HELPERS
# ================================================================

def _safe_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def _pairwise_spearman(xs, ys):
    """Spearman correlation without scipy, for dose-response."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if len(xs) < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    r = np.corrcoef(rx, ry)[0, 1]
    # Two-sided p-value via t-approximation.
    n = len(xs)
    if abs(r) >= 1.0:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / max(1e-12, 1 - r * r))
        # Rough p from t with n-2 dof using normal approximation.
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    return float(r), float(p)


# ================================================================
# QUERY REPRESENTATION FIDELITY (QRF)  — re-implemented here so the
# construct-validity script does not depend on the SAE code path.
# If your paper's QRF is SAE-based, swap in the shared implementation.
# ================================================================

@torch.no_grad()
def compute_qrf_for_prompt(
    model,
    tokenizer,
    prompt,
    layers,
):
    """
    Simplified residual-stream QRF used for the construct-validity
    experiment.  Measures cosine similarity between the mean query-token
    residual and the mean post-query-token residual at each monitored
    layer, averaged across layers.

    If the paper's QRF is SAE-based, replace this function with a call
    to the shared SAE-based implementation.  The rest of the experiment
    is agnostic to which QRF is used.
    """
    try:
        input_ids, attention_mask, query_positions = (
            tokenize_prompt_with_query_span(tokenizer, prompt)
        )
    except Exception:
        return float("nan")

    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states  # tuple of [1, S, D]

    n_q = len(query_positions)
    n_total = input_ids.shape[1]
    gen_positions = [p for p in range(n_total) if p not in set(query_positions)]
    if not gen_positions:
        return float("nan")

    scores = []
    for layer_idx in layers:
        # hidden_states[0] is embeddings; layer l output is hidden_states[l+1].
        h = hidden_states[layer_idx + 1][0]  # [S, D]
        q_vec = h[query_positions].mean(dim=0)
        g_vec = h[gen_positions].mean(dim=0)
        cos = torch.nn.functional.cosine_similarity(
            q_vec.unsqueeze(0), g_vec.unsqueeze(0)
        ).item()
        scores.append(cos)

    return float(np.mean(scores)) if scores else float("nan")


# ================================================================
# CCS  (activation patching over query positions, bins, Jaccard)
# ================================================================

@torch.no_grad()
def compute_ccs_single_example(
    model,
    tokenizer,
    prompt,
    layers,
    kappa=KAPPA,
    rho_for_corruption=0.4,
    seed=13,
):
    """
    Returns the average across monitored layers of the per-layer top-kappa
    circuit.  For a single example we cannot compute cross-example
    Jaccard, so we instead return the circuit size (fraction of query
    positions selected) as a proxy that can be aggregated across examples.

    NOTE: for paired comparisons (M1, M2, M4) we want a per-example scalar.
    We use the *mean patch recovery* across the top-kappa positions, which
    is well-defined per example and is the quantity the CCS ranking is
    built on.  Cross-example Jaccard is reported separately as the
    aggregate CCS in the main paper.
    """
    try:
        clean_ids, clean_mask, query_positions = (
            tokenize_prompt_with_query_span(tokenizer, prompt)
        )
    except Exception:
        return float("nan")

    clean_ids = clean_ids.to(model.device)
    clean_mask = clean_mask.to(model.device)

    target_token_id, _ = get_clean_first_generated_token(
        model, clean_ids, clean_mask
    )

    corrupt_ids, _ = corrupt_query_ids(
        clean_ids,
        query_positions,
        rho=rho_for_corruption,
        tokenizer=tokenizer,
        seed=seed,
        mode="random",
    )
    corrupt_mask = clean_mask.clone()

    per_layer_recovery = []
    for layer_idx in layers:
        mlp = get_mlp_module(model, layer_idx)
        cacher = ActivationCacher(mlp)
        _ = model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            use_cache=False,
        )
        clean_act = cacher.activation
        cacher.close()
        if clean_act is None:
            continue

        corrupt_logits = model(
            input_ids=corrupt_ids,
            attention_mask=corrupt_mask,
            use_cache=False,
        ).logits[:, -1, :]
        corrupt_target = float(corrupt_logits[0, target_token_id].item())

        B = len(query_positions)
        batched_ids = corrupt_ids.repeat(B, 1)
        batched_mask = corrupt_mask.repeat(B, 1)

        patcher = BatchedMLPPatcher(mlp, clean_act, query_positions)
        patched_logits = model(
            input_ids=batched_ids,
            attention_mask=batched_mask,
            use_cache=False,
        ).logits[:, -1, :]
        patcher.close()

        patched_target = patched_logits[:, target_token_id]
        recovery = (patched_target - corrupt_target).detach().float().cpu().tolist()
        per_layer_recovery.append(float(np.mean(recovery)))

    return float(np.mean(per_layer_recovery)) if per_layer_recovery else float("nan")


# ================================================================
# AQA  (head-level activation patching, clipped, averaged)
# ================================================================

@torch.no_grad()
def compute_aqa_single_example(
    model,
    tokenizer,
    prompt,
    layers,
    rho_for_corruption=0.4,
    seed=13,
    head_fraction=0.5,
):
    """
    Per-example scalar AQA: mean clipped head-level recovery over the
    top head_fraction of heads at each monitored layer, averaged over
    layers.  Mirrors the paper's clipped AQA aggregate.
    """
    try:
        clean_ids, clean_mask, query_positions = (
            tokenize_prompt_with_query_span(tokenizer, prompt)
        )
    except Exception:
        return float("nan")

    clean_ids = clean_ids.to(model.device)
    clean_mask = clean_mask.to(model.device)

    target_token_id, _ = get_clean_first_generated_token(
        model, clean_ids, clean_mask
    )

    corrupt_ids, _ = corrupt_query_ids(
        clean_ids,
        query_positions,
        rho=rho_for_corruption,
        tokenizer=tokenizer,
        seed=seed,
        mode="random",
    )
    corrupt_mask = clean_mask.clone()

    layers_module = get_transformer_layers(model)

    per_layer_scores = []
    for layer_idx in layers:
        layer = layers_module[layer_idx]
        attn = layer.self_attn
        n_heads = getattr(attn, "num_heads", None)
        if n_heads is None:
            # Try config fallback.
            n_heads = getattr(model.config, "num_attention_heads", None)
        if n_heads is None:
            continue
        head_dim = getattr(attn, "head_dim", None)
        if head_dim is None:
            hidden = getattr(model.config, "hidden_size", None)
            if hidden is None:
                continue
            head_dim = hidden // n_heads

        # Cache clean attention output (pre o_proj) at query positions.
        clean_attn_out = {}

        def make_cache_hook(name):
            def hook(module, args):
                if not args:
                    return
                clean_attn_out[name] = args[0].detach()
            return hook

        cache_handle = attn.o_proj.register_forward_pre_hook(make_cache_hook("o"))
        _ = model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            use_cache=False,
        )
        cache_handle.remove()
        if "o" not in clean_attn_out:
            continue
        clean_o = clean_attn_out["o"]  # [1, S, n_heads*head_dim]

        # Corrupted baseline logit.
        corrupt_logits = model(
            input_ids=corrupt_ids,
            attention_mask=corrupt_mask,
            use_cache=False,
        ).logits[:, -1, :]
        corrupt_target = float(corrupt_logits[0, target_token_id].item())

        # For each head, patch clean activation at each query position
        # (averaged over positions) and measure clipped recovery.
        head_scores = []
        for h in range(n_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)

            def make_patch_hook(head_slice, positions, clean_tensor):
                def hook(module, args):
                    if not args:
                        return args
                    x = args[0].clone()
                    for pos in positions:
                        x[0, pos, head_slice] = clean_tensor[0, pos, head_slice]
                    return (x,) + tuple(args[1:])
                return hook

            patch_handle = attn.o_proj.register_forward_pre_hook(
                make_patch_hook(sl, query_positions, clean_o)
            )
            patched_logits = model(
                input_ids=corrupt_ids,
                attention_mask=corrupt_mask,
                use_cache=False,
            ).logits[:, -1, :]
            patch_handle.remove()

            patched_target = float(patched_logits[0, target_token_id].item())
            raw = patched_target - corrupt_target
            clipped = max(0.0, min(1.0, raw))
            head_scores.append(clipped)

        if head_scores:
            k = max(1, int(math.ceil(head_fraction * len(head_scores))))
            top = sorted(head_scores, reverse=True)[:k]
            per_layer_scores.append(float(np.mean(top)))

    return float(np.mean(per_layer_scores)) if per_layer_scores else float("nan")


# ================================================================
# COMPOSITE
# ================================================================

def compose_queryeval(qrf, ccs, aqa):
    parts = [x for x in (qrf, ccs, aqa) if np.isfinite(x)]
    if not parts:
        return float("nan")
    return float(np.mean(parts))


def score_prompt(model, tokenizer, prompt, layers):
    """Compute all three components and the composite for one prompt."""
    qrf = compute_qrf_for_prompt(model, tokenizer, prompt, layers)
    ccs = compute_ccs_single_example(model, tokenizer, prompt, layers)
    aqa = compute_aqa_single_example(model, tokenizer, prompt, layers)
    qe = compose_queryeval(qrf, ccs, aqa)
    return {"QRF": qrf, "CCS": ccs, "AQA": aqa, "QueryEval": qe}


# ================================================================
# BEHAVIORAL CRITERIA
# ================================================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt, max_new_tokens=48):
    """Greedy decode, returned as a string.  Used for behavioral criteria."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def token_overlap_fraction(answer, source):
    """Fraction of answer tokens that appear in source (bag of words)."""
    a = set(str(answer).lower().split())
    s = set(str(source).lower().split())
    if not a:
        return 0.0
    return len(a & s) / len(a)


def context_attribution(answer, context):
    return token_overlap_fraction(answer, context)


def query_attribution(answer, query):
    return token_overlap_fraction(answer, query)


# ================================================================
# MANIPULATION 1: CONTEXT SWAP
# ================================================================

def build_m1_pairs(df, dataset_name, rng):
    """
    Matched: (row_i.context, row_i.question).
    Mismatched: (row_j.context, row_i.question) for a shuffled j != i.

    We skip pairs where the shuffled context accidentally contains the
    original question's answer, to keep the mismatch clean.
    """
    n = len(df)
    perm = list(range(n))
    rng.shuffle(perm)
    # Ensure derangement: no fixed points.
    for i in range(n):
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]

    pairs = []
    for i in range(n):
        j = perm[i]
        row_i = df.iloc[i]
        row_j = df.iloc[j]
        q = get_original_question(dataset_name, row_i)

        prompt_matched = build_prompt(dataset_name, row_i, q)
        prompt_mismatched = build_prompt(dataset_name, row_j, q)

        pairs.append({
            "row_id": i,
            "partner_row_id": j,
            "question": q,
            "prompt_matched": prompt_matched,
            "prompt_mismatched": prompt_mismatched,
            "context_matched": str(row_i.get("document", row_i.get("table", "")))[:2000],
            "context_mismatched": str(row_j.get("document", row_j.get("table", "")))[:2000],
        })
    return pairs


def run_m1(model, tokenizer, model_key, dataset_name, df, layers):
    rng = random.Random(13)
    pairs = build_m1_pairs(df, dataset_name, rng)
    rows = []
    for p in tqdm(pairs, desc=f"M1 {model_key}/{dataset_name}"):
        s_matched = score_prompt(model, tokenizer, p["prompt_matched"], layers)
        s_mismatched = score_prompt(model, tokenizer, p["prompt_mismatched"], layers)

        ans_matched = generate_answer(model, tokenizer, p["prompt_matched"])
        ans_mismatched = generate_answer(model, tokenizer, p["prompt_mismatched"])

        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "row_id": p["row_id"],
            "partner_row_id": p["partner_row_id"],
            "QRF_matched": s_matched["QRF"],
            "QRF_mismatched": s_mismatched["QRF"],
            "CCS_matched": s_matched["CCS"],
            "CCS_mismatched": s_mismatched["CCS"],
            "AQA_matched": s_matched["AQA"],
            "AQA_mismatched": s_mismatched["AQA"],
            "QueryEval_matched": s_matched["QueryEval"],
            "QueryEval_mismatched": s_mismatched["QueryEval"],
            "dQRF": s_matched["QRF"] - s_mismatched["QRF"],
            "dCCS": s_matched["CCS"] - s_mismatched["CCS"],
            "dAQA": s_matched["AQA"] - s_mismatched["AQA"],
            "dQueryEval": s_matched["QueryEval"] - s_mismatched["QueryEval"],
            "ctx_attr_matched": context_attribution(ans_matched, p["context_matched"]),
            "ctx_attr_mismatched": context_attribution(ans_mismatched, p["context_mismatched"]),
        })
    return rows


# ================================================================
# MANIPULATION 2: QUERY ABLATION
# ================================================================

def build_m2_pairs(df, dataset_name):
    pairs = []
    for i in range(len(df)):
        row = df.iloc[i]
        q = get_original_question(dataset_name, row)
        prompt_orig = build_prompt(dataset_name, row, q)
        # Replace the query content with a neutral placeholder of similar length.
        placeholder = " ".join(["question"] * max(1, len(q.split())))
        prompt_ablate = build_prompt(dataset_name, row, placeholder)
        pairs.append({
            "row_id": i,
            "question": q,
            "prompt_original": prompt_orig,
            "prompt_ablated": prompt_ablate,
        })
    return pairs


def run_m2(model, tokenizer, model_key, dataset_name, df, layers):
    pairs = build_m2_pairs(df, dataset_name)
    rows = []
    for p in tqdm(pairs, desc=f"M2 {model_key}/{dataset_name}"):
        s_orig = score_prompt(model, tokenizer, p["prompt_original"], layers)
        s_abl = score_prompt(model, tokenizer, p["prompt_ablated"], layers)
        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "row_id": p["row_id"],
            "QRF_original": s_orig["QRF"],
            "QRF_ablated": s_abl["QRF"],
            "CCS_original": s_orig["CCS"],
            "CCS_ablated": s_abl["CCS"],
            "AQA_original": s_orig["AQA"],
            "AQA_ablated": s_abl["AQA"],
            "QueryEval_original": s_orig["QueryEval"],
            "QueryEval_ablated": s_abl["QueryEval"],
            "dQueryEval": s_orig["QueryEval"] - s_abl["QueryEval"],
        })
    return rows


# ================================================================
# MANIPULATION 3: CORRUPTION GRADIENT
# ================================================================

def run_m3(model, tokenizer, model_key, dataset_name, df, layers):
    """
    For each rho, corrupt the query with the existing corruption routine
    and score the resulting prompt.  Average over seeds at each rho.
    """
    rows = []
    for rho in GRADIENT_RHOS:
        for seed in CORRUPTION_SEEDS:
            for i in tqdm(range(len(df)),
                          desc=f"M3 {model_key}/{dataset_name} rho={rho} seed={seed}"):
                row = df.iloc[i]
                q = get_original_question(dataset_name, row)
                prompt = build_prompt(dataset_name, row, q)

                try:
                    clean_ids, clean_mask, qpos = tokenize_prompt_with_query_span(
                        tokenizer, prompt
                    )
                except Exception:
                    continue
                clean_ids = clean_ids.to(model.device)
                clean_mask = clean_mask.to(model.device)

                if rho == 0.0:
                    corrupt_ids = clean_ids
                else:
                    corrupt_ids, _ = corrupt_query_ids(
                        clean_ids, qpos, rho=rho,
                        tokenizer=tokenizer, seed=seed, mode="random"
                    )

                corrupt_prompt_text = tokenizer.decode(
                    corrupt_ids[0], skip_special_tokens=True
                )

                s = score_prompt(model, tokenizer, corrupt_prompt_text, layers)
                rows.append({
                    "model": model_key,
                    "dataset": dataset_name,
                    "row_id": i,
                    "rho": rho,
                    "seed": seed,
                    "QRF": s["QRF"],
                    "CCS": s["CCS"],
                    "AQA": s["AQA"],
                    "QueryEval": s["QueryEval"],
                })
    return rows


# ================================================================
# MANIPULATION 4: PARAMETRIC vs CONTEXTUAL CONFLICT
# ================================================================

def build_m4_examples():
    """
    Curated counterfactual items where the model has strong parametric
    knowledge but the context contradicts it.

    Format: (question, parametric_answer, counterfactual_answer, context_template)
    The context template must contain both the counterfactual answer and
    enough surrounding text to look natural.
    """
    items = [
        {
            "question": "What is the capital of France?",
            "parametric": "Paris",
            "counterfactual": "Lyon",
        },
        {
            "question": "Who wrote Hamlet?",
            "parametric": "William Shakespeare",
            "counterfactual": "Christopher Marlowe",
        },
        {
            "question": "What is the largest planet in the solar system?",
            "parametric": "Jupiter",
            "counterfactual": "Saturn",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "parametric": "Au",
            "counterfactual": "Ag",
        },
        {
            "question": "In what year did World War II end?",
            "parametric": "1945",
            "counterfactual": "1943",
        },
        {
            "question": "What is the capital of Japan?",
            "parametric": "Tokyo",
            "counterfactual": "Osaka",
        },
        {
            "question": "Who painted the Mona Lisa?",
            "parametric": "Leonardo da Vinci",
            "counterfactual": "Michelangelo",
        },
        {
            "question": "What is the longest river in the world?",
            "parametric": "The Nile",
            "counterfactual": "The Amazon",
        },
        {
            "question": "What is the smallest prime number?",
            "parametric": "2",
            "counterfactual": "3",
        },
        {
            "question": "What is the boiling point of water in Celsius?",
            "parametric": "100",
            "counterfactual": "90",
        },
    ]
    for it in items:
        it["context_contextual"] = (
            f"According to the provided reference, the answer is "
            f"{it['parametric']}."
        )
        it["context_counterfactual"] = (
            f"According to the provided reference, the answer is "
            f"{it['counterfactual']}."
        )
        it["context_none"] = ""
    return items


def run_m4(model, tokenizer, model_key, layers):
    items = build_m4_examples()
    rows = []
    for it in tqdm(items, desc=f"M4 {model_key}"):
        prompts = {
            "A_contextual": (
                f"Context: {it['context_contextual']}\n\n"
                f"Query: {it['question']}\n\nAnswer:"
            ),
            "B_counterfactual": (
                f"Context: {it['context_counterfactual']}\n\n"
                f"Query: {it['question']}\n\nAnswer:"
            ),
            "C_none": (
                f"Context: \n\n"
                f"Query: {it['question']}\n\nAnswer:"
            ),
        }
        row = {
            "model": model_key,
            "question": it["question"],
            "parametric": it["parametric"],
            "counterfactual": it["counterfactual"],
        }
        for cond, prompt in prompts.items():
            s = score_prompt(model, tokenizer, prompt, layers)
            row[f"QRF_{cond}"] = s["QRF"]
            row[f"CCS_{cond}"] = s["CCS"]
            row[f"AQA_{cond}"] = s["AQA"]
            row[f"QueryEval_{cond}"] = s["QueryEval"]
            ans = generate_answer(model, tokenizer, prompt)
            row[f"answer_{cond}"] = ans
            row[f"uses_counterfactual_{cond}"] = int(
                it["counterfactual"].lower() in ans.lower()
            )
            row[f"uses_parametric_{cond}"] = int(
                it["parametric"].lower() in ans.lower()
            )
        rows.append(row)
    return rows


# ================================================================
# ANALYSIS
# ================================================================

def paired_effect(df, value_cols, group_cols):
    """
    For each value column, report mean paired delta, std, and a
    paired t-statistic.  df is expected to have *_matched / *_mismatched
    or *_original / *_ablated suffixes.
    """
    out = []
    for col in value_cols:
        base = col
        a = df.get(f"{base}_matched", df.get(f"{base}_original"))
        b = df.get(f"{base}_mismatched", df.get(f"{base}_ablated"))
        if a is None or b is None:
            continue
        d = (a - b).dropna().values
        if len(d) < 3:
            continue
        mean_d = float(np.mean(d))
        std_d = float(np.std(d, ddof=1))
        se = std_d / math.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = mean_d / se if se and se > 0 else float("nan")
        out.append({
            "component": base,
            "n": len(d),
            "mean_delta": mean_d,
            "std_delta": std_d,
            "t_stat": t,
        })
    return pd.DataFrame(out)


def dose_response_table(df):
    """Aggregate M3 by rho and report Spearman rho of each metric vs rho."""
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        sub = sub.dropna(subset=["QueryEval"])
        if sub.empty:
            continue
        rho_vals = sub["rho"].values
        for metric in ["QRF", "CCS", "AQA", "QueryEval"]:
            vals = sub[metric].values
            r, p = _pairwise_spearman(rho_vals, vals)
            rows.append({
                "model": model,
                "dataset": dataset,
                "metric": metric,
                "spearman_rho": r,
                "p_value": p,
                "n": len(vals),
            })
    return pd.DataFrame(rows)


def find_disagreement_cases(m1_df, top_n=10):
    """
    Cases where QueryEval drops sharply from matched to mismatched but
    the answer still looks similar to the reference.  We use context
    attribution as a proxy for output-level behavior.
    """
    if m1_df.empty:
        return m1_df
    m1_df = m1_df.copy()
    m1_df["abs_dQueryEval"] = m1_df["dQueryEval"].abs()
    m1_df["abs_dCtxAttr"] = (
        m1_df["ctx_attr_matched"] - m1_df["ctx_attr_mismatched"]
    ).abs()
    # High QueryEval delta, low ctx-attr delta = disagreement.
    m1_df["disagreement_score"] = (
        m1_df["abs_dQueryEval"].rank(pct=True)
        - m1_df["abs_dCtxAttr"].rank(pct=True)
    )
    return m1_df.sort_values("disagreement_score", ascending=False).head(top_n)


# ================================================================
# PLOTTING
# ================================================================

def plot_dose_response(m3_df, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, dataset in zip(axes, sorted(m3_df["dataset"].unique())):
        sub = m3_df[m3_df["dataset"] == dataset]
        for metric in ["QRF", "CCS", "AQA", "QueryEval"]:
            agg = sub.groupby("rho")[metric].mean()
            ax.plot(agg.index, agg.values, marker="o", label=metric)
        ax.set_title(dataset)
        ax.set_xlabel("corruption rho")
        ax.set_ylabel("score")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_component_deltas(m1_df, m2_df, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (df, title, suffix) in zip(
        axes,
        [(m1_df, "M1 context swap", ("matched", "mismatched")),
         (m2_df, "M2 query ablation", ("original", "ablated"))],
    ):
        a_suf, b_suf = suffix
        for metric in ["QRF", "CCS", "AQA", "QueryEval"]:
            a = df.get(f"{metric}_{a_suf}", pd.Series(dtype=float)).dropna()
            b = df.get(f"{metric}_{b_suf}", pd.Series(dtype=float)).dropna()
            if a.empty or b.empty:
                continue
            delta = (a.values - b.values)
            ax.bar(metric, np.mean(delta), yerr=np.std(delta, ddof=1) / math.sqrt(len(delta)),
                   capsize=4)
        ax.set_title(title)
        ax.set_ylabel("mean paired delta")
        ax.axhline(0, color="k", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ================================================================
# MAIN RUN
# ================================================================

def run_model_dataset(model, tokenizer, model_key, dataset_name, layers):
    print("\n" + "=" * 80)
    print(f"Construct validity: {model_key} / {dataset_name}")
    print("=" * 80)

    df = pd.read_csv(DATASETS[dataset_name]["test_csv"])
    if NUM_ROWS is not None:
        df = df.iloc[:NUM_ROWS].reset_index(drop=True)

    out_dir = os.path.join(OUTPUT_DIR, model_key, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # ---- M1 ----
    m1_rows = run_m1(model, tokenizer, model_key, dataset_name, df, layers)
    m1_df = pd.DataFrame(m1_rows)
    m1_df.to_csv(os.path.join(out_dir, "m1_paired.csv"), index=False)

    # ---- M2 ----
    m2_rows = run_m2(model, tokenizer, model_key, dataset_name, df, layers)
    m2_df = pd.DataFrame(m2_rows)
    m2_df.to_csv(os.path.join(out_dir, "m2_paired.csv"), index=False)

    # ---- M3 ----
    m3_rows = run_m3(model, tokenizer, model_key, dataset_name, df, layers)
    m3_df = pd.DataFrame(m3_rows)
    m3_df.to_csv(os.path.join(out_dir, "m3_gradient.csv"), index=False)
    plot_dose_response(m3_df, os.path.join(out_dir, "dose_response.png"))

    # ---- M4 (dataset-independent; run once per model) ----
    # We only run M4 for the first dataset to avoid duplication.
    if dataset_name == list(DATASETS.keys())[0]:
        m4_rows = run_m4(model, tokenizer, model_key, layers)
        m4_df = pd.DataFrame(m4_rows)
        m4_df.to_csv(os.path.join(OUTPUT_DIR, model_key, "m4_conflict.csv"), index=False)

    # ---- Effect sizes ----
    eff_m1 = paired_effect(
        m1_df,
        ["QRF", "CCS", "AQA", "QueryEval", "ctx_attr"],
        ["model", "dataset"],
    )
    eff_m1["manipulation"] = "M1"
    eff_m2 = paired_effect(
        m2_df,
        ["QRF", "CCS", "AQA", "QueryEval"],
        ["model", "dataset"],
    )
    eff_m2["manipulation"] = "M2"
    eff = pd.concat([eff_m1, eff_m2], ignore_index=True)
    eff.to_csv(os.path.join(out_dir, "effect_sizes.csv"), index=False)

    # ---- Dose-response correlation table ----
    dr = dose_response_table(m3_df)
    dr.to_csv(os.path.join(out_dir, "dose_response_stats.csv"), index=False)

    # ---- Disagreement cases ----
    dis = find_disagreement_cases(m1_df, top_n=10)
    dis.to_csv(os.path.join(out_dir, "disagreement_cases.csv"), index=False)

    # ---- Component delta plot ----
    plot_component_deltas(m1_df, m2_df, os.path.join(out_dir, "component_deltas.png"))

    print(f"\nSaved construct-validity outputs to {out_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = {
        "num_rows": NUM_ROWS,
        "gradient_rhos": GRADIENT_RHOS,
        "corruption_seeds": CORRUPTION_SEEDS,
        "kappa": KAPPA,
        "monitored_layers_by_model": {k: v["layers"] for k, v in MODELS.items()},
        "note": (
            "Four construct-validity manipulations.  M1 context swap, "
            "M2 query ablation, M3 corruption gradient, M4 parametric "
            "vs contextual conflict.  See module docstring for predictions."
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for model_key, model_cfg in MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"], use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_name"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        layers = list(model_cfg["layers"])

        for dataset_name in DATASETS:
            run_model_dataset(model, tokenizer, model_key, dataset_name, layers)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()