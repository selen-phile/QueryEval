import os
import ast
import gc
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import PreTrainedTokenizerBase

# Patch: Restore missing method for bert_score compatibility with newer transformers
if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
    PreTrainedTokenizerBase.build_inputs_with_special_tokens = lambda self, *args, **kwargs: self.encode("", add_special_tokens=True)
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODELS = {
    "llama": {
        "model_name": "meta-llama/Meta-Llama-3.1-8B",
    },
    "qwen": {
        "model_name": "Qwen/Qwen2.5-7B",
    },
    "gemma": {
        "model_name": "google/gemma-2-9b",
    },
}

DATASETS = {
    "squad": {
        "test_csv": "dataset/squad_3_column_test.csv",
        "schema":   "generic",
    },
    "ms_marco": {
        "test_csv": "dataset/ms_marco_3_column_test.csv",
        "schema":   "generic",
    },
    "qtsumm": {
        "test_csv": "dataset/qtsumm_test.csv",
        "schema":   "qtsumm",
    },
}

NUM_ROWS       = 200
MAX_NEW_TOKENS = 100    # enough to cover all dataset answer lengths
OUTPUT_DIR     = "rouge_bert_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BERTSCORE_MODEL = "bert-base-uncased"   # fully compatible with all bert-score versions


# ─────────────────────────────────────────────
# PROMPT FORMATTERS
# ─────────────────────────────────────────────
def format_generic(document, question):
    doc = str(document)[:1000]
    return f"Context: {doc}\n\nQuery: {question}\n\nAnswer:"


def format_qtsumm(table_raw, query):
    try:
        table_data = ast.literal_eval(table_raw)
    except (ValueError, SyntaxError):
        table_data = {"title": "", "headers": [], "data": []}

    title   = table_data.get("title", "")
    headers = table_data.get("headers", [])
    rows    = table_data.get("data", [])

    md = ""
    if title:
        md += f"### {title}\n\n"
    if headers:
        md += "| " + " | ".join(str(h) for h in headers) + " |\n"
        md += "|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        md += "| " + " | ".join(str(c) for c in r) + " |\n"

    return f"Table Data:\n{md}\nQuery: {query}\n\nSummary:"


def build_prompt(schema, row):
    if schema == "qtsumm":
        return format_qtsumm(row["table"], row["query"])
    return format_generic(row["document"], row["question"])


def get_reference(schema, row):
    if schema == "qtsumm":
        return str(row["summary"])
    return str(row["answer_text"])


# ─────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────
def generate_answers(model, tokenizer, df, schema, device):
    """Generate answers for all rows, return list of generated strings."""
    generated = []

    for i in tqdm(range(len(df)), desc="Generating"):
        row    = df.loc[i]
        prompt = build_prompt(schema, row)

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=512
        ).to(device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,              # greedy decoding
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (not the prompt)
        new_tokens = output_ids[0, input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        generated.append(answer)

    return generated


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def compute_rouge(predictions, references):
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    r1, r2, rl = [], [], []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(sum(r1) / len(r1), 4),
        "rouge2": round(sum(r2) / len(r2), 4),
        "rougeL": round(sum(rl) / len(rl), 4),
    }


def compute_bertscore(predictions, references):
    """Compute BERTScore F1."""
    P, R, F1 = bert_score(
        predictions, references,
        model_type=BERTSCORE_MODEL,
        lang="en",
        verbose=False,
        device="cpu",    # run on CPU to avoid conflicting with LLM on GPU
    )
    return {"bertscore_f1": round(F1.mean().item(), 4)}


# ─────────────────────────────────────────────
# CORE: ONE MODEL × DATASET RUN
# ─────────────────────────────────────────────
def run_pair(model_key, dataset_key):
    model_cfg   = MODELS[model_key]
    dataset_cfg = DATASETS[dataset_key]
    schema      = dataset_cfg["schema"]

    print(f"\n{'='*60}")
    print(f"  SCORING: model={model_key}  dataset={dataset_key}")
    print(f"{'='*60}\n")

    # Load model
    print(f"Loading model: {model_cfg['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"], dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    # Load dataset
    df = pd.read_csv(dataset_cfg["test_csv"])
    if NUM_ROWS is not None:
        df = df.iloc[:NUM_ROWS].reset_index(drop=True)

    # Generate answers
    predictions = generate_answers(model, tokenizer, df, schema, device)
    references  = [get_reference(schema, df.loc[i]) for i in range(len(df))]

    # Free LLM before BERTScore (runs on CPU)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Save per-row generations
    gen_df = pd.DataFrame({
        "row":        list(range(len(df))),
        "reference":  references,
        "prediction": predictions,
    })
    gen_path = os.path.join(
        OUTPUT_DIR, f"generations_{model_key}_{dataset_key}.csv"
    )
    gen_df.to_csv(gen_path, index=False)
    print(f"Saved generations: {gen_path}")

    # Compute scores
    print("Computing ROUGE...")
    rouge_scores = compute_rouge(predictions, references)
    print(f"  ROUGE-1={rouge_scores['rouge1']:.4f}  "
          f"ROUGE-2={rouge_scores['rouge2']:.4f}  "
          f"ROUGE-L={rouge_scores['rougeL']:.4f}")

    print("Computing BERTScore (CPU)...")
    bs_scores = compute_bertscore(predictions, references)
    print(f"  BERTScore-F1={bs_scores['bertscore_f1']:.4f}")

    # Save scores
    scores = {
        "model":          model_key,
        "dataset":        dataset_key,
        **rouge_scores,
        **bs_scores,
    }
    scores_df = pd.DataFrame([scores])
    scores_path = os.path.join(
        OUTPUT_DIR, f"scores_{model_key}_{dataset_key}.csv"
    )
    scores_df.to_csv(scores_path, index=False)
    print(f"Saved scores: {scores_path}")

    return scores


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    all_scores = []

    for model_key in MODELS:
        for dataset_key in DATASETS:
            scores_path = os.path.join(OUTPUT_DIR, f"scores_{model_key}_{dataset_key}.csv")
            gen_path    = os.path.join(OUTPUT_DIR, f"generations_{model_key}_{dataset_key}.csv")

            # Case 1: scores already exist — load and skip entirely
            if os.path.exists(scores_path):
                print(f"\nSkipping {model_key}/{dataset_key} — scores already exist.")
                existing = pd.read_csv(scores_path).to_dict("records")[0]
                all_scores.append(existing)
                continue

            # Case 2: generations exist but no scores — skip generation, just score
            if os.path.exists(gen_path):
                print(f"\n{model_key}/{dataset_key} — generations found, computing scores only.")
                df          = pd.read_csv(pd.read_csv(gen_path).__class__.__name__ and gen_path)
                predictions = df["prediction"].fillna("").tolist()
                references  = df["reference"].fillna("").tolist()
                rouge_scores = compute_rouge(predictions, references)
                bs_scores    = compute_bertscore(predictions, references)
                scores = {"model": model_key, "dataset": dataset_key, **rouge_scores, **bs_scores}
                pd.DataFrame([scores]).to_csv(scores_path, index=False)
                all_scores.append(scores)
                continue

            # Case 3: nothing exists — run full pipeline
            scores = run_pair(model_key, dataset_key)
            all_scores.append(scores)

            # Save grand summary after every pair — crash safe
            grand_path = os.path.join(OUTPUT_DIR, "grand_summary.csv")
            pd.DataFrame(all_scores).to_csv(grand_path, index=False)

    # Final grand summary
    grand_path = os.path.join(OUTPUT_DIR, "grand_summary.csv")
    pd.DataFrame(all_scores).to_csv(grand_path, index=False)

    print(f"\n{'='*60}")
    print("  ALL SCORING RUNS COMPLETE")
    print(f"{'='*60}\n")

    # Print results table
    df_all = pd.DataFrame(all_scores)
    print("— Grand Results Table —\n")
    print(f"{'Model':<8} {'Dataset':<12} {'ROUGE-1':>8} {'ROUGE-2':>8} {'ROUGE-L':>8} {'BERTScore':>10}")
    print("-" * 60)
    for _, row in df_all.iterrows():
        print(f"{row['model']:<8} {row['dataset']:<12} "
              f"{row['rouge1']:>8.4f} {row['rouge2']:>8.4f} "
              f"{row['rougeL']:>8.4f} {row['bertscore_f1']:>10.4f}")

    print(f"\nGrand summary saved to {grand_path}")


if __name__ == "__main__":
    main()
