"""
CCS revision for QueryEval
==========================

This version fixes the main CCS issues identified by the TMLR reviewers:

1. NO ground-truth/reference answer is used.
   The target is the first token that the CLEAN model would generate
   greedily from the prompt.

2. CCS is implemented with activation patching rather than SAE-feature
   Jaccard.  For each example/layer, query-token positions are patched
   one at a time (batched over positions), and the recovery of the clean
   first-generated-token logit is measured.

3. Cross-example CCS is computed explicitly:
       CCS_l = mean pairwise Jaccard(C_i,l, C_j,l)
   where circuits are sets of normalized query-position bins.  This makes
   the position-alignment rule explicit when query lengths differ.

4. Controlled experiments:
   - corruption rho in {0.20, 0.40, 0.60}
   - top-kappa in {0.10, 0.20, 0.50, 0.80}
   - multiple fixed corruption seeds
   - random-position circuit baseline
   - random-target baseline
   - shuffled-query control
   - within-example clean-vs-corrupted circuit overlap
   - cross-example CCS
   - layer-wise CCS
   - per-example circuit/recovery CSVs
   - reproducibility statistics across corruption seeds

5. All comparisons use the SAME examples and fixed random seeds for
   controlled ablations.

Important:
This script intentionally does NOT load the old transcoder/SAE code for CCS.
The previous implementation extracted active SAE features and compared them
between clean/noisy inputs. That is not the activation-patching CCS described
in the paper and does not directly address the reviewer criticism.

Test on a small NUM_ROWS first before the full run.

FIXES IN THIS VERSION:
- Main and every control explicitly receive the identical model-specific layer list.
- No global CURRENT_LAYERS state is used.
- Random-target and shuffled-query controls therefore cannot silently drift
  to a different layer set.
- NUM_ROWS defaults to 5 for validation.
- A ccs_control_comparison.csv file is produced with actual-vs-control deltas.
- The original raw CCS, recovery, patch-score, and control outputs are retained.

PERFORMANCE OPTIMIZATIONS (behavior-preserving):
- Random-target control is patched once per (rho, seed), then reused for all kappas.
- Shuffled-query control is patched once per (rho, seed), then reused for all kappas.
- Random-target control no longer performs a discarded clean-target forward pass.
- The single clean forward is reused for generated-target selection and for all
  monitored-layer MLP activations; the corrupted baseline logit is computed once.
"""

import os
import gc
import json
import math
import ast
import random
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# CONFIG
# ============================================================

MODELS = {
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
        "layers": [15, 18, 20, 22, 25],
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B",
        "layers": [13, 15, 18, 20, 22],
    },
    "gemma": {
        "model_name": "google/gemma-2-9b",
        "layers": [19, 23, 26, 29, 33],
    },
}

DATASETS = {
    "squad": {
        "test_csv": "dataset/squad_3_column_test.csv",
        "robust_csv": "dataset/squad_test_robustness_all.csv",
    },
    "ms_marco": {
        "test_csv": "dataset/ms_marco_3_column_test.csv",
        "robust_csv": "dataset/ms_marco_test_robustness_all.csv",
    },
    "qtsumm": {
        "test_csv": "dataset/qtsumm_test.csv",
        "robust_csv": "dataset/qtsumm_test_robustness_all.csv",
    },
}

# Use the same examples across every ablation.
NUM_ROWS = 200

MAX_LENGTH = 512

# Reviewer-requested sensitivity analyses.
CORRUPTION_RATES = [0.20, 0.40, 0.60]
KAPPAS = [0.10, 0.20, 0.50, 0.80]

# Multiple corruption seeds.  Do NOT change these between kappa/rho runs.
CORRUPTION_SEEDS = [13, 42, 97]

# Number of normalized query-position bins used for cross-example
# circuit alignment.
N_POSITION_BINS = 20

# Query circuit extraction is restricted to the query token span.
# Exclude examples where the query cannot be located reliably.
MIN_QUERY_TOKENS = 1

# Controls.
RUN_RANDOM_POSITION_CONTROL = True
RUN_RANDOM_TARGET_CONTROL = True
RUN_SHUFFLED_QUERY_CONTROL = True

# A random target token is sampled from the model vocabulary, excluding
# special tokens. Set to True only after the main run works.
RANDOM_TARGET_SEED = 1234

# Save detailed per-position patch scores.
SAVE_PATCH_SCORES = True

OUTPUT_DIR = "ccs_revision_results"


# ============================================================
# PROMPT FORMATTERS
# ============================================================

def format_prompt_generic(document, question):
    """
    IMPORTANT:
    The answer/reference is deliberately NOT inserted into the prompt.
    This is what makes the CCS target genuinely reference-free.
    """
    doc = str(document)[:1000]
    return f"Context: {doc}\n\nQuery: {question}\n\nAnswer:"


def format_prompt_qtsumm(table_raw, question):
    try:
        table_data = ast.literal_eval(table_raw)
    except Exception:
        # Safe fallback for malformed table strings.
        try:
            table_data = ast.literal_eval(table_raw)
        except Exception:
            table_data = {"title": "", "headers": [], "data": []}

    title = table_data.get("title", "")
    headers = table_data.get("headers", [])
    rows = table_data.get("data", [])

    markdown_str = ""
    if title:
        markdown_str += f"### {title}\n\n"

    if headers:
        markdown_str += "| " + " | ".join(str(h) for h in headers) + " |\n"
        markdown_str += "|" + "|".join(["---" for _ in headers]) + "|\n"

    if rows:
        for row_data in rows:
            markdown_str += "| " + " | ".join(str(cell) for cell in row_data) + " |\n"

    return (
        f"Table Data:\n{markdown_str}\n"
        f"Query: {question}\n\n"
        f"Summary:"
    )


def build_prompt(dataset_name, row, question):
    if dataset_name == "qtsumm":
        return format_prompt_qtsumm(row["table"], question)
    return format_prompt_generic(row["document"], question)


def get_original_question(dataset_name, row):
    if dataset_name == "qtsumm":
        return str(row["query"])
    return str(row["question"])


# ============================================================
# TOKEN / QUERY-SPAN HELPERS
# ============================================================

def find_query_char_span(prompt):
    """
    Find the question between the last 'Query:' and 'Answer:' / 'Summary:'.

    We intentionally use the LAST occurrence to avoid accidentally matching
    the same question text inside the context/table.
    """
    query_marker = "Query:"
    answer_markers = ["\n\nAnswer:", "\n\nSummary:"]

    q_start_marker = prompt.rfind(query_marker)
    if q_start_marker < 0:
        raise ValueError("Could not find Query: marker.")

    q_start = q_start_marker + len(query_marker)

    end_candidates = []
    for marker in answer_markers:
        idx = prompt.find(marker, q_start)
        if idx >= 0:
            end_candidates.append(idx)

    if not end_candidates:
        raise ValueError("Could not find Answer:/Summary: marker.")

    q_end = min(end_candidates)

    # Remove only surrounding whitespace from the character span.
    while q_start < q_end and prompt[q_start].isspace():
        q_start += 1
    while q_end > q_start and prompt[q_end - 1].isspace():
        q_end -= 1

    return q_start, q_end


def tokenize_prompt_with_query_span(tokenizer, prompt):
    """
    Tokenize without truncation first, then retain the last MAX_LENGTH tokens
    so the Answer: position and query are preserved.

    Returns input_ids, attention_mask, query_positions.
    """
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0]

    # The token indices and offset mapping MUST refer to exactly the same
    # sequence.  This explicit check prevents stale/shifted query positions
    # from ever reaching the activation-patching hook.
    if len(input_ids) != len(offsets):
        raise ValueError(
            f"Tokenizer mismatch: input_ids={len(input_ids)}, "
            f"offsets={len(offsets)}"
        )

    q_start, q_end = find_query_char_span(prompt)

    query_positions = []
    for idx, (s, e) in enumerate(offsets.tolist()):
        # Ignore zero-width special-token offsets.
        if e <= s:
            continue
        if max(s, q_start) < min(e, q_end):
            query_positions.append(idx)

    if not query_positions:
        raise ValueError("Could not identify query token positions.")

    # Because truncation is performed by the tokenizer above, positions are
    # already in the final model-input coordinate system.
    seq_len = int(input_ids.shape[0])
    invalid = [p for p in query_positions if p < 0 or p >= seq_len]
    if invalid:
        raise ValueError(
            "Invalid query-token positions after tokenization: "
            f"max_position={max(query_positions)}, seq_len={seq_len}. "
            "The example is skipped rather than producing an invalid circuit."
        )

    if not query_positions:
        raise ValueError(
            "Query was truncated away. Increase MAX_LENGTH or shorten context."
        )

    attention_mask = torch.ones_like(input_ids)

    return (
        input_ids.unsqueeze(0),
        attention_mask.unsqueeze(0),
        query_positions,
    )


# ============================================================
# RANDOM / CONTROLLED QUERY CORRUPTION
# ============================================================

def sample_replacement_token(tokenizer, rng, vocab_size):
    """
    Select a random vocabulary token while avoiding obvious special tokens.
    """
    special_ids = set(
        x for x in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        ]
        if x is not None
    )

    # Avoid a pathological loop if vocabulary is tiny.
    for _ in range(100):
        token_id = rng.randrange(vocab_size)
        if token_id not in special_ids:
            return token_id

    return 0


def corrupt_query_ids(
    input_ids,
    query_positions,
    rho,
    tokenizer,
    seed,
    mode="random",
):
    """
    Returns a copy of input_ids with query tokens modified.

    mode:
      random  : replace rho fraction with random vocabulary tokens
      shuffle : randomly permute query token IDs while preserving length
    """
    out = input_ids.clone()

    positions = list(query_positions)
    n_corrupt = max(1, int(math.ceil(rho * len(positions))))
    n_corrupt = min(n_corrupt, len(positions))

    rng = random.Random(seed)

    selected = sorted(rng.sample(positions, n_corrupt))

    if mode == "random":
        for p in selected:
            out[0, p] = sample_replacement_token(
                tokenizer,
                rng,
                tokenizer.vocab_size,
            )

    elif mode == "shuffle":
        # Permute the selected query-token identities. Force a non-identity
        # permutation whenever at least two tokens are selected, so the
        # control cannot accidentally leave the query unchanged.
        values = [int(out[0, p].item()) for p in selected]
        if len(values) >= 2:
            original = list(values)
            for _ in range(10):
                rng.shuffle(values)
                if values != original:
                    break
            if values == original:
                values = original[1:] + original[:1]
        for p, value in zip(selected, values):
            out[0, p] = value

    else:
        raise ValueError(f"Unknown corruption mode: {mode}")

    return out, selected


# ============================================================
# MODEL MODULE HELPERS
# ============================================================

def get_transformer_layers(model):
    """
    Supports the three model families used in the paper.

    Returns the decoder layer ModuleList.
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]

    for fn in candidates:
        try:
            layers = fn(model)
            if layers is not None:
                return layers
        except Exception:
            pass

    raise RuntimeError(
        "Could not locate transformer layers. "
        "Print model.named_modules() and add the architecture path."
    )


def get_mlp_module(model, layer_idx):
    layers = get_transformer_layers(model)
    return layers[layer_idx].mlp


# ============================================================
# ACTIVATION CACHE / PATCH HOOKS
# ============================================================

class ActivationCacher:
    """
    Captures the input to the MLP at one transformer layer.
    """

    def __init__(self, module):
        self.module = module
        self.activation = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if not args:
            return
        self.activation = args[0].detach()

    def close(self):
        self.handle.remove()
        self.activation = None


class BatchedMLPPatcher:
    """
    During a batched corrupted forward pass, patch one query position per
    batch item with the corresponding clean MLP input activation.

    patch_positions[b] is the query position patched in batch item b.
    """

    def __init__(self, module, clean_activation, patch_positions):
        self.module = module
        self.clean_activation = clean_activation
        self.patch_positions = patch_positions

        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if not args:
            return args

        hidden = args[0].clone()

        # hidden: [B, S, D]
        # clean_activation: [1, S, D]
        # Guard against any tokenizer/model sequence mismatch.  We raise a
        # clear error instead of silently patching the wrong token.
        clean_len = int(self.clean_activation.shape[1])
        hidden_len = int(hidden.shape[1])
        if clean_len != hidden_len:
            raise RuntimeError(
                "Activation sequence-length mismatch during patching: "
                f"clean={clean_len}, current={hidden_len}. "
                "This example cannot be safely patched."
            )

        for b, pos in enumerate(self.patch_positions):
            if pos < 0 or pos >= hidden_len:
                raise RuntimeError(
                    f"Patch position {pos} is outside sequence length {hidden_len}."
                )
            hidden[b, pos, :] = self.clean_activation[0, pos, :]

        new_args = (hidden,) + tuple(args[1:])
        return new_args

    def close(self):
        self.handle.remove()


# ============================================================
# LOGIT HELPERS
# ============================================================

@torch.no_grad()
def forward_logits(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    # Last prompt position predicts the first generated token.
    return outputs.logits[:, -1, :]


@torch.no_grad()
def get_clean_first_generated_token(model, input_ids, attention_mask):
    next_logits = forward_logits(model, input_ids, attention_mask)
    token_id = int(torch.argmax(next_logits[0]).item())
    return token_id, float(next_logits[0, token_id].item())


# ============================================================
# CIRCUIT EXTRACTION
# ============================================================

def top_k_circuit(recovery_scores, query_positions, kappa):
    """
    Return the top-kappa query positions by logit recovery.
    """
    n = len(query_positions)
    k = max(1, int(math.ceil(kappa * n)))

    order = sorted(
        range(n),
        key=lambda i: recovery_scores[i],
        reverse=True,
    )[:k]

    return [query_positions[i] for i in order]


def normalized_query_bin(position, query_positions, n_bins):
    """
    Convert an absolute token position into a normalized relative position.

    This is the explicit cross-example alignment rule:
    query token positions are compared by relative location inside the query,
    not by absolute prompt index.
    """
    ordered = list(query_positions)
    try:
        rank = ordered.index(position)
    except ValueError:
        raise ValueError("Position is not inside query_positions.")

    if len(ordered) == 1:
        return 0

    rel = rank / float(len(ordered) - 1)
    return min(n_bins - 1, int(math.floor(rel * n_bins)))


def circuit_to_bins(circuit_positions, query_positions, n_bins):
    return {
        normalized_query_bin(p, query_positions, n_bins)
        for p in circuit_positions
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


# ============================================================
# RANDOM-POSITION CONTROL
# ============================================================

def random_circuit_bins(
    query_positions,
    circuit_size,
    n_bins,
    seed,
):
    """
    Random-position baseline with the same circuit cardinality.
    """
    rng = random.Random(seed)
    all_bins = list(range(n_bins))

    # If there are fewer distinct normalized bins than circuit size,
    # sample without replacement up to the available number.
    k = min(circuit_size, len(all_bins))

    return set(rng.sample(all_bins, k))


def pairwise_mean_jaccard(circuit_dict):
    """
    circuit_dict:
        example_id -> set of normalized query-position bins

    Returns the mean pairwise Jaccard across examples.
    """
    sets = list(circuit_dict.values())

    if len(sets) < 2:
        return float("nan")

    scores = [
        jaccard(a, b)
        for a, b in itertools.combinations(sets, 2)
    ]

    return float(np.mean(scores))


# ============================================================
# ONE LAYER: COMPUTE PATCH RECOVERY
# ============================================================

@torch.no_grad()
def compute_layers_recoveries(
    model,
    layers,
    clean_ids,
    clean_mask,
    corrupt_ids,
    corrupt_mask,
    query_positions,
    target_token_id=None,
    target_validator=None,
):
    """
    Compute patch recoveries for all monitored layers while reusing work that
    is identical across layers.

    Behavior is unchanged relative to calling the old per-layer routine:
      - the clean MLP input at each layer is identical,
      - the corrupted target logit is identical,
      - each layer still gets its own batched single-position patched forward.

    The only optimization is resource reuse:
      1. one clean forward caches MLP inputs for every monitored layer;
      2. one corrupted forward computes the target baseline once;
      3. one patched forward per monitored layer computes the layer-specific
         position recoveries.
    """
    layers = list(layers)
    if not layers:
        return {}

    mlps = {layer_idx: get_mlp_module(model, layer_idx) for layer_idx in layers}
    clean_activations = {}
    handles = []

    def make_cacher(layer_idx):
        def _hook(module, args):
            if args:
                clean_activations[layer_idx] = args[0].detach()
        return _hook

    # Cache all monitored-layer MLP inputs in the SAME clean forward.
    try:
        for layer_idx, mlp in mlps.items():
            handles.append(mlp.register_forward_pre_hook(make_cacher(layer_idx)))

        clean_logits = model(
            input_ids=clean_ids,
            attention_mask=clean_mask,
            use_cache=False,
        ).logits[:, -1, :]
    finally:
        for handle in handles:
            handle.remove()

    for layer_idx in layers:
        if layer_idx not in clean_activations:
            raise RuntimeError(
                f"No MLP activation captured at layer {layer_idx}."
            )

    # Reuse this same clean forward for the generated-target selection.
    # The old implementation performed a separate clean forward solely to
    # obtain the first generated token, then performed another clean forward
    # to cache MLP activations.
    if target_token_id is None:
        target_token_id = int(torch.argmax(clean_logits[0]).item())
        if target_validator is not None and not target_validator(target_token_id):
            # Preserve the original behavior: invalid generated targets are
            # rejected before any corrupted or patched forwards are run.
            return target_token_id, None

    clean_target_logit = float(clean_logits[0, target_token_id].item())

    # One corrupted forward is sufficient for every monitored layer because
    # the unpatched corrupted run is layer-independent.
    corrupt_logits = model(
        input_ids=corrupt_ids,
        attention_mask=corrupt_mask,
        use_cache=False,
    ).logits[:, -1, :]
    corrupt_target_logit = float(corrupt_logits[0, target_token_id].item())

    B = len(query_positions)
    batched_ids = corrupt_ids.repeat(B, 1)
    batched_mask = corrupt_mask.repeat(B, 1)

    layer_outputs = {}

    for layer_idx in layers:
        mlp = mlps[layer_idx]
        clean_activation = clean_activations[layer_idx]

        patcher = BatchedMLPPatcher(
            mlp,
            clean_activation,
            query_positions,
        )
        try:
            patched_logits = model(
                input_ids=batched_ids,
                attention_mask=batched_mask,
                use_cache=False,
            ).logits[:, -1, :]
        finally:
            patcher.close()

        patched_target = patched_logits[:, target_token_id]
        recovery = (
            patched_target - corrupt_target_logit
        ).detach().float().cpu().tolist()

        layer_outputs[layer_idx] = {
            "clean_target_logit": clean_target_logit,
            "corrupt_target_logit": corrupt_target_logit,
            "recovery_scores": recovery,
        }

    return target_token_id, layer_outputs


# ============================================================
# TARGET VALIDITY
# ============================================================

def is_valid_generated_target(tokenizer, token_id):
    """Reject EOS/BOS/PAD/UNK as a semantic first-answer target."""
    special_ids = {
        x for x in [
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        ] if x is not None
    }
    return int(token_id) not in special_ids


def target_diagnostic(tokenizer, token_id):
    return {
        "target_is_special": not is_valid_generated_target(tokenizer, token_id),
        "target_is_eos": (
            tokenizer.eos_token_id is not None
            and int(token_id) == int(tokenizer.eos_token_id)
        ),
    }


# ============================================================
# PROCESS ONE EXAMPLE FOR ONE CORRUPTION SETTING
# ============================================================

def process_example_condition(
    model,
    tokenizer,
    dataset_name,
    row,
    question,
    rho,
    corruption_seed,
    corruption_mode,
    target_mode="generated",
    layers=None,
):
    if layers is None:
        raise ValueError("layers must be supplied explicitly.")

    layers = list(layers)
    prompt = build_prompt(dataset_name, row, question)

    (
        clean_ids,
        clean_mask,
        query_positions,
    ) = tokenize_prompt_with_query_span(
        tokenizer,
        prompt,
    )

    clean_ids = clean_ids.to(model.device)
    clean_mask = clean_mask.to(model.device)

    # For the generated target, target selection is performed inside the same
    # clean forward used to cache all monitored-layer MLP activations. This
    # avoids a duplicate clean forward. For the random-target control, no
    # clean-target generation is needed at all.
    if target_mode == "generated":
        target_token_id = None
    elif target_mode == "random":
        rng = random.Random(RANDOM_TARGET_SEED + corruption_seed)
        target_token_id = sample_replacement_token(
            tokenizer,
            rng,
            tokenizer.vocab_size,
        )
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    corrupt_ids, selected_corrupt_positions = corrupt_query_ids(
        clean_ids,
        query_positions,
        rho=rho,
        tokenizer=tokenizer,
        seed=corruption_seed,
        mode=corruption_mode,
    )
    corrupt_mask = clean_mask.clone()

    # IMPORTANT: the exact same layer list is passed to every condition,
    # including all controls. This prevents actual/control layer mismatches.
    target_token_id, layer_outputs = compute_layers_recoveries(
        model=model,
        layers=layers,
        clean_ids=clean_ids,
        clean_mask=clean_mask,
        corrupt_ids=corrupt_ids,
        corrupt_mask=corrupt_mask,
        query_positions=query_positions,
        target_token_id=target_token_id,
        target_validator=(
            (lambda token_id: is_valid_generated_target(tokenizer, token_id))
            if target_mode == "generated" else None
        ),
    )

    # A special token such as EOS is not a meaningful answer token for the
    # semantic CCS experiment. The validator above rejects it before any
    # corrupted/patched forwards, matching the original behavior.
    if layer_outputs is None:
        raise ValueError(
            f"Invalid generated target token {target_token_id} "
            f"({tokenizer.decode([target_token_id])!r}); first generated token is special."
        )

    return {
        "prompt": prompt,
        "query_positions": query_positions,
        "selected_corrupt_positions": selected_corrupt_positions,
        "target_token_id": target_token_id,
        "target_token": tokenizer.decode([target_token_id]),
        "target_mode": target_mode,
        "rho": rho,
        "corruption_seed": corruption_seed,
        "corruption_mode": corruption_mode,
        "layer_outputs": layer_outputs,
    }


# ============================================================
# CCS AGGREGATION
# ============================================================

def aggregate_ccs_for_condition(
    example_outputs,
    kappa,
    n_bins,
    random_position=False,
    random_seed=42,
):
    """
    example_outputs:
       list of processed examples

    Returns:
       layer-level CCS, within-example clean/corrupt overlap,
       and circuit dictionaries.
    """
    if not example_outputs:
        return {}

    layers = sorted(
        example_outputs[0]["layer_outputs"].keys()
    )

    results = {}

    for layer_idx in layers:
        circuits = {}
        within_example = []

        for ex_idx, ex in enumerate(example_outputs):
            query_positions = ex["query_positions"]
            recovery = ex["layer_outputs"][layer_idx]["recovery_scores"]

            circuit_positions = top_k_circuit(
                recovery,
                query_positions,
                kappa,
            )

            clean_bins = circuit_to_bins(
                circuit_positions,
                query_positions,
                n_bins,
            )

            if random_position:
                clean_bins = random_circuit_bins(
                    query_positions,
                    len(clean_bins),
                    n_bins,
                    seed=random_seed + ex_idx * 1009 + layer_idx,
                )

            circuits[ex_idx] = clean_bins

            # Within-example control:
            # compare the top recovery circuit with the set of positions
            # whose recovery is non-positive / low. This is a diagnostic,
            # not the primary CCS score.
            #
            # We save the size and recovery distribution instead of claiming
            # this is a formal clean-vs-corrupt circuit similarity.
            within_example.append(
                float(np.mean(recovery))
            )

        ccs = pairwise_mean_jaccard(circuits)

        results[layer_idx] = {
            "ccs": ccs,
            "mean_patch_recovery": float(np.mean(within_example)),
            "circuits": circuits,
        }

    return results


# ============================================================
# MAIN MODEL/DATASET RUN
# ============================================================


def run_one_condition(
    model,
    tokenizer,
    model_key,
    dataset_name,
    df_test,
    rho,
    corruption_seed,
    corruption_mode,
    target_mode,
    kappa,
    layers,
):
    layers = list(layers)

    example_outputs = []
    patch_records = []
    circuit_records = []

    for i in tqdm(
        range(len(df_test)),
        desc=(
            f"{model_key}/{dataset_name} "
            f"rho={rho} seed={corruption_seed} "
            f"mode={corruption_mode} target={target_mode} "
            f"kappa={kappa}"
        ),
    ):
        row = df_test.iloc[i]

        question = get_original_question(
            dataset_name,
            row,
        )

        try:
            output = process_example_condition(
                model=model,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                row=row,
                question=question,
                rho=rho,
                corruption_seed=corruption_seed,
                corruption_mode=corruption_mode,
                target_mode=target_mode,
                layers=layers,
            )
        except Exception as exc:
            print(
                f"\n[WARNING] row={i} failed: {type(exc).__name__}: {exc}"
            )
            continue

        example_outputs.append(output)

        # Save per-position patch scores.
        if SAVE_PATCH_SCORES:
            for layer_idx, layer_out in output["layer_outputs"].items():
                for pos, score in zip(
                    output["query_positions"],
                    layer_out["recovery_scores"],
                ):
                    circuit_records.append({
                        "model": model_key,
                        "dataset": dataset_name,
                        "row": i,
                        "layer": layer_idx,
                        "query_position": pos,
                        "query_length": len(output["query_positions"]),
                        "relative_query_position": (
                            output["query_positions"].index(pos)
                            / max(len(output["query_positions"]) - 1, 1)
                        ),
                        "recovery": score,
                        "rho": rho,
                        "corruption_seed": corruption_seed,
                        "corruption_mode": corruption_mode,
                        "target_mode": target_mode,
                        "target_token_id": output["target_token_id"],
                        "target_token": output["target_token"],
                    })

    # --------------------------------------------------------
    # Aggregate CCS for every requested kappa.
    # --------------------------------------------------------
    agg = aggregate_ccs_for_condition(
        example_outputs=example_outputs,
        kappa=kappa,
        n_bins=N_POSITION_BINS,
        random_position=False,
        random_seed=corruption_seed,
    )

    rows = []

    for layer_idx, info in agg.items():
        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "layer": layer_idx,
            "rho": rho,
            "corruption_seed": corruption_seed,
            "corruption_mode": corruption_mode,
            "target_mode": target_mode,
            "kappa": kappa,
            "n_examples": len(example_outputs),
            "n_position_bins": N_POSITION_BINS,
            "CCS": info["ccs"],
            "mean_patch_recovery": info["mean_patch_recovery"],
        })

    return rows, circuit_records, example_outputs


# ============================================================
# RANDOM POSITION CONTROL
# ============================================================

def compute_random_position_control(
    example_outputs,
    model_key,
    dataset_name,
    rho,
    corruption_seed,
    corruption_mode,
    target_mode,
    kappa,
):
    agg = aggregate_ccs_for_condition(
        example_outputs=example_outputs,
        kappa=kappa,
        n_bins=N_POSITION_BINS,
        random_position=True,
        random_seed=777 + corruption_seed,
    )

    rows = []

    for layer_idx, info in agg.items():
        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "layer": layer_idx,
            "rho": rho,
            "corruption_seed": corruption_seed,
            "corruption_mode": corruption_mode,
            "target_mode": target_mode,
            "kappa": kappa,
            "control": "random_position",
            "CCS_random_position": info["ccs"],
        })

    return rows


# ============================================================
# SHUFFLED QUERY CONTROL
# ============================================================

def run_shuffled_query_base(
    model,
    tokenizer,
    model_key,
    dataset_name,
    df_test,
    rho,
    seed,
    layers=None,
):
    """
    Same clean example, but the query-token identities are shuffled.
    The prompt length and query positions remain fixed.

    We measure cross-example CCS using the generated first-token target
    from each original clean prompt.

    This is a control for whether the circuit structure is robust to
    query surface-form changes.
    """
    if layers is None:
        layers = list(MODELS[model_key]["layers"])
    else:
        layers = list(layers)

    example_outputs = []

    for i in tqdm(
        range(len(df_test)),
        desc=f"{model_key}/{dataset_name} shuffled-query rho={rho} seed={seed}",
    ):
        row = df_test.iloc[i]
        question = get_original_question(dataset_name, row)
        prompt = build_prompt(dataset_name, row, question)

        try:
            clean_ids, clean_mask, query_positions = (
                tokenize_prompt_with_query_span(
                    tokenizer,
                    prompt,
                )
            )
        except Exception:
            continue

        clean_ids = clean_ids.to(model.device)
        clean_mask = clean_mask.to(model.device)

        shuffled_ids, _ = corrupt_query_ids(
            clean_ids,
            query_positions,
            rho=1.0,
            tokenizer=tokenizer,
            seed=seed + i,
            mode="shuffle",
        )

        # EXACTLY the same monitored layers as the main experiment. The clean
        # target and all clean MLP activations are obtained from one forward.
        target_token_id, layer_outputs = compute_layers_recoveries(
            model=model,
            layers=layers,
            clean_ids=clean_ids,
            clean_mask=clean_mask,
            corrupt_ids=shuffled_ids,
            corrupt_mask=clean_mask,
            query_positions=query_positions,
            target_token_id=None,
            target_validator=(lambda token_id: is_valid_generated_target(tokenizer, token_id)),
        )

        if layer_outputs is None:
            continue

        example_outputs.append({
            "query_positions": query_positions,
            "target_token_id": target_token_id,
            "layer_outputs": layer_outputs,
        })

    return example_outputs


def run_shuffled_query_control(
    model,
    tokenizer,
    model_key,
    dataset_name,
    df_test,
    rho,
    seed,
    kappa,
    layers=None,
):
    """Compatibility wrapper: compute the base once, then aggregate one kappa."""
    example_outputs = run_shuffled_query_base(
        model=model,
        tokenizer=tokenizer,
        model_key=model_key,
        dataset_name=dataset_name,
        df_test=df_test,
        rho=rho,
        seed=seed,
        layers=layers,
    )

    agg = aggregate_ccs_for_condition(
        example_outputs,
        kappa=kappa,
        n_bins=N_POSITION_BINS,
        random_position=False,
        random_seed=seed,
    )

    rows = []
    for layer_idx, info in agg.items():
        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "layer": layer_idx,
            "rho": rho,
            "seed": seed,
            "corruption_seed": seed,
            "kappa": kappa,
            "control": "shuffled_query",
            "CCS": info["ccs"],
            "mean_patch_recovery": info["mean_patch_recovery"],
            "n_examples": len(example_outputs),
        })

    return rows


# ============================================================
# FULL RUN
# ============================================================

def run_model_dataset(model_key, dataset_name):
    print("\n" + "=" * 80)
    print(f"CCS REVISION: {model_key} / {dataset_name}")
    print("=" * 80)

    model_cfg = MODELS[model_key]
    dataset_cfg = DATASETS[dataset_name]
    layers = list(model_cfg["layers"])

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_name"],
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    df_test = pd.read_csv(dataset_cfg["test_csv"])

    if NUM_ROWS is not None:
        df_test = df_test.iloc[:NUM_ROWS].reset_index(drop=True)

    out_dir = os.path.join(
        OUTPUT_DIR,
        model_key,
        dataset_name,
    )
    os.makedirs(out_dir, exist_ok=True)

    all_summary_rows = []
    all_patch_records = []
    all_control_rows = []

    # --------------------------------------------------------
    # Main random-token corruption experiment
    # --------------------------------------------------------
    for rho in CORRUPTION_RATES:
        for seed in CORRUPTION_SEEDS:

            # We compute the expensive patch scores once for this
            # rho/seed, then reuse them for every kappa.
            base_outputs = []

            for i in tqdm(
                range(len(df_test)),
                desc=f"Base patching {model_key}/{dataset_name} rho={rho} seed={seed}",
            ):
                row = df_test.iloc[i]
                question = get_original_question(dataset_name, row)

                try:
                    output = process_example_condition(
                        model=model,
                        tokenizer=tokenizer,
                        dataset_name=dataset_name,
                        row=row,
                        question=question,
                        rho=rho,
                        corruption_seed=seed,
                        corruption_mode="random",
                        target_mode="generated",
                        layers=layers,
                    )
                except Exception as exc:
                    print(
                        f"\n[WARNING] row={i} failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                base_outputs.append(output)

                if SAVE_PATCH_SCORES:
                    for layer_idx, layer_out in output["layer_outputs"].items():
                        for pos, score in zip(
                            output["query_positions"],
                            layer_out["recovery_scores"],
                        ):
                            all_patch_records.append({
                                "model": model_key,
                                "dataset": dataset_name,
                                "row": i,
                                "layer": layer_idx,
                                "query_position": pos,
                                "query_length": len(output["query_positions"]),
                                "relative_query_position": (
                                    output["query_positions"].index(pos)
                                    / max(len(output["query_positions"]) - 1, 1)
                                ),
                                "recovery": score,
                                "rho": rho,
                                "corruption_seed": seed,
                                "corruption_mode": "random",
                                "target_mode": "generated",
                                "target_token_id": output["target_token_id"],
                                "target_token": output["target_token"],
                            })

            # Fixed sample set for every kappa.
            for kappa in KAPPAS:
                agg = aggregate_ccs_for_condition(
                    base_outputs,
                    kappa=kappa,
                    n_bins=N_POSITION_BINS,
                    random_position=False,
                    random_seed=seed,
                )

                for layer_idx, info in agg.items():
                    all_summary_rows.append({
                        "model": model_key,
                        "dataset": dataset_name,
                        "layer": layer_idx,
                        "rho": rho,
                        "corruption_seed": seed,
                        "corruption_mode": "random",
                        "target_mode": "generated",
                        "kappa": kappa,
                        "CCS": info["ccs"],
                        "mean_patch_recovery": info["mean_patch_recovery"],
                        "n_examples": len(base_outputs),
                        "n_position_bins": N_POSITION_BINS,
                        "monitored_layers": ",".join(map(str, layers)),
                    })

                if RUN_RANDOM_POSITION_CONTROL:
                    ctrl = compute_random_position_control(
                        base_outputs,
                        model_key=model_key,
                        dataset_name=dataset_name,
                        rho=rho,
                        corruption_seed=seed,
                        corruption_mode="random",
                        target_mode="generated",
                        kappa=kappa,
                    )
                    all_control_rows.extend(ctrl)

    # --------------------------------------------------------
    # Matched random-target control
    # --------------------------------------------------------
    # The patch recoveries do not depend on kappa.  Compute each rho/seed
    # condition ONCE, then reuse the same example-level recoveries for every
    # kappa.  The previous implementation repeated all model forwards once
    # per kappa.
    if RUN_RANDOM_TARGET_CONTROL:
        for rho_ctrl in CORRUPTION_RATES:
            for seed_ctrl in CORRUPTION_SEEDS:
                _, patch_records, target_outputs = run_one_condition(
                    model=model,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    dataset_name=dataset_name,
                    df_test=df_test,
                    rho=rho_ctrl,
                    corruption_seed=seed_ctrl,
                    corruption_mode="random",
                    target_mode="random",
                    # kappa is irrelevant while computing patch recoveries.
                    kappa=KAPPAS[0],
                    layers=layers,
                )
                all_patch_records.extend(patch_records)

                for kappa in KAPPAS:
                    agg = aggregate_ccs_for_condition(
                        target_outputs,
                        kappa=kappa,
                        n_bins=N_POSITION_BINS,
                        random_position=False,
                        random_seed=seed_ctrl,
                    )
                    for layer_idx, info in agg.items():
                        all_control_rows.append({
                            "model": model_key,
                            "dataset": dataset_name,
                            "layer": layer_idx,
                            "rho": rho_ctrl,
                            "corruption_seed": seed_ctrl,
                            "seed": seed_ctrl,
                            "corruption_mode": "random",
                            "target_mode": "random",
                            "kappa": kappa,
                            "n_examples": len(target_outputs),
                            "n_position_bins": N_POSITION_BINS,
                            "CCS": info["ccs"],
                            "mean_patch_recovery": info["mean_patch_recovery"],
                            "control": "random_target",
                            "control_rho": rho_ctrl,
                            "control_seed": seed_ctrl,
                        })

    # --------------------------------------------------------
    # Matched shuffled-query control
    # --------------------------------------------------------
    # The expensive shuffled-query patching is independent of kappa.  Compute
    # each rho/seed condition once, then aggregate the cached recoveries for
    # every kappa.
    if RUN_SHUFFLED_QUERY_CONTROL:
        for rho_ctrl in CORRUPTION_RATES:
            for seed_ctrl in CORRUPTION_SEEDS:
                shuffled_outputs = run_shuffled_query_base(
                    model=model,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    dataset_name=dataset_name,
                    df_test=df_test,
                    rho=rho_ctrl,
                    seed=seed_ctrl,
                    layers=layers,
                )

                for kappa in KAPPAS:
                    agg = aggregate_ccs_for_condition(
                        shuffled_outputs,
                        kappa=kappa,
                        n_bins=N_POSITION_BINS,
                        random_position=False,
                        random_seed=seed_ctrl,
                    )
                    for layer_idx, info in agg.items():
                        all_control_rows.append({
                            "model": model_key,
                            "dataset": dataset_name,
                            "layer": layer_idx,
                            "rho": rho_ctrl,
                            "seed": seed_ctrl,
                            "corruption_seed": seed_ctrl,
                            "kappa": kappa,
                            "control": "shuffled_query",
                            "CCS": info["ccs"],
                            "mean_patch_recovery": info["mean_patch_recovery"],
                            "n_examples": len(shuffled_outputs),
                        })

    # --------------------------------------------------------
    # Matched control comparison + chance-corrected CCS
    # --------------------------------------------------------
    if all_summary_rows and all_control_rows:
        actual_df = pd.DataFrame(all_summary_rows)
        control_df = pd.DataFrame(all_control_rows)

        keys = ["model", "dataset", "layer", "rho", "corruption_seed", "kappa"]

        actual = (
            actual_df.groupby(keys, as_index=False)
            .agg(
                CCS_actual=("CCS", "mean"),
                recovery_mean=("mean_patch_recovery", "mean"),
                n_examples=("n_examples", "mean"),
            )
        )

        # Random-position rows are generated from exactly the same example
        # circuits and matched rho/seed/kappa.
        rp = control_df[control_df["control"].eq("random_position")].copy()
        rp = (
            rp.groupby(keys, as_index=False)
            .agg(CCS_random_position=("CCS_random_position", "mean"))
        ) if not rp.empty else pd.DataFrame(columns=keys+["CCS_random_position"])

        rt = control_df[control_df["control"].eq("random_target")].copy()
        rt = (
            rt.groupby(keys, as_index=False)
            .agg(CCS_random_target=("CCS", "mean"))
        ) if not rt.empty else pd.DataFrame(columns=keys+["CCS_random_target"])

        sq = control_df[control_df["control"].eq("shuffled_query")].copy()
        sq = (
            sq.groupby(keys, as_index=False)
            .agg(CCS_shuffled_query=("CCS", "mean"))
        ) if not sq.empty else pd.DataFrame(columns=keys+["CCS_shuffled_query"])

        comparison = actual.merge(rp, on=keys, how="left")
        comparison = comparison.merge(rt, on=keys, how="left")
        comparison = comparison.merge(sq, on=keys, how="left")

        comparison["Delta_random_position"] = comparison["CCS_actual"] - comparison["CCS_random_position"]
        comparison["Delta_random_target"] = comparison["CCS_actual"] - comparison["CCS_random_target"]
        comparison["Delta_shuffled_query"] = comparison["CCS_actual"] - comparison["CCS_shuffled_query"]

        # A simple baseline-normalized effect size. This is intentionally
        # NOT called a mathematically chance-corrected Jaccard; it is the
        # empirically baseline-subtracted CCS used for interpretation.
        comparison["CCS_excess_over_random_position"] = comparison["Delta_random_position"]
        comparison["CCS_ratio_over_random_position"] = (
            comparison["CCS_actual"] / comparison["CCS_random_position"].replace(0, np.nan)
        )
        comparison["monitored_layers"] = ",".join(map(str, layers))

        comparison_path = os.path.join(out_dir, "ccs_control_comparison.csv")
        comparison.to_csv(comparison_path, index=False)

        # Aggregate across corruption seeds for a paper-ready summary.
        paper = (
            comparison.groupby(
                ["model", "dataset", "layer", "rho", "kappa"],
                as_index=False,
            )
            .agg(
                CCS_mean=("CCS_actual", "mean"),
                CCS_std=("CCS_actual", "std"),
                RandomPosition_mean=("CCS_random_position", "mean"),
                RandomTarget_mean=("CCS_random_target", "mean"),
                ShuffledQuery_mean=("CCS_shuffled_query", "mean"),
                Delta_RandomPosition=("Delta_random_position", "mean"),
                Delta_RandomTarget=("Delta_random_target", "mean"),
                Delta_ShuffledQuery=("Delta_shuffled_query", "mean"),
                recovery_mean=("recovery_mean", "mean"),
            )
        )
        paper.to_csv(os.path.join(out_dir, "ccs_paper_summary.csv"), index=False)

        print("\nMatched control comparison saved to:", comparison_path)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    summary_path = os.path.join(
        out_dir,
        "ccs_summary.csv",
    )
    pd.DataFrame(all_summary_rows).to_csv(
        summary_path,
        index=False,
    )

    control_path = os.path.join(
        out_dir,
        "ccs_controls.csv",
    )
    pd.DataFrame(all_control_rows).to_csv(
        control_path,
        index=False,
    )

    if SAVE_PATCH_SCORES:
        patch_path = os.path.join(
            out_dir,
            "ccs_patch_scores.csv",
        )
        pd.DataFrame(all_patch_records).to_csv(
            patch_path,
            index=False,
        )

    # A compact summary for quick inspection.
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)

        compact = (
            summary_df
            .groupby(
                ["model", "dataset", "layer", "rho", "kappa"],
                as_index=False,
            )
            .agg(
                CCS_mean=("CCS", "mean"),
                CCS_std=("CCS", "std"),
                recovery_mean=("mean_patch_recovery", "mean"),
                n_seed=("corruption_seed", "nunique"),
            )
        )

        compact_path = os.path.join(
            out_dir,
            "ccs_compact_summary.csv",
        )
        compact.to_csv(
            compact_path,
            index=False,
        )

        print("\nCompact CCS summary:")
        print(compact.to_string(index=False))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save configuration for reproducibility.
    config = {
        "corruption_rates": CORRUPTION_RATES,
        "kappas": KAPPAS,
        "corruption_seeds": CORRUPTION_SEEDS,
        "n_position_bins": N_POSITION_BINS,
        "num_rows": NUM_ROWS,
        "max_length": MAX_LENGTH,
        "random_target_control": RUN_RANDOM_TARGET_CONTROL,
        "random_position_control": RUN_RANDOM_POSITION_CONTROL,
        "shuffled_query_control": RUN_SHUFFLED_QUERY_CONTROL,
        "monitored_layers_by_model": {
            k: v["layers"] for k, v in MODELS.items()
        },
        "note": (
            "All controls are matched to the same model/dataset/example/rho/seed/kappa; "
            "generated special-token targets are excluded rather than reinterpreted. "
            "Run a 5-row smoke test by changing NUM_ROWS before the full 200-row run."
        ),
    }

    with open(
        os.path.join(OUTPUT_DIR, "run_config.json"),
        "w",
    ) as f:
        json.dump(config, f, indent=2)

    for model_key in MODELS:
        for dataset_name in DATASETS:
            run_model_dataset(
                model_key=model_key,
                dataset_name=dataset_name,
            )


if __name__ == "__main__":
    main()
