#!/usr/bin/env python3
"""
exp_p2_a0_medical_pubmedqa_adapter

Verifies that format-matched LoRA training on PubMedQA produces delta_D > 0 for the
medical domain, while MCQ-format training (same questions) does NOT (format-register mismatch).

Kill criteria:
  K1166: base Gemma 4 E4B accuracy on PubMedQA 3-class (yes/no/maybe) < 50%
  K1167: PubMedQA-trained LoRA adapter accuracy > base_acc + 15pp (delta_D > 0.15)
  K1168: MCQ-trained adapter (same questions, MCQ format) does NOT exceed base + 5pp

References:
  Finding #457 (delta_D prediction: gap required), Finding #409 (PubMedQA base=23%, M2P=55%)
  arxiv 1909.06146 (PubMedQA), arxiv 2106.09685 (LoRA), arxiv 2012.13255 (intrinsic dim)
"""

import gc
import json
import os
import subprocess
import time
from pathlib import Path

import mlx.core as mx

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
PUBMED_ADAPTER_DIR = EXPERIMENT_DIR / "pubmed_adapter"
MCQ_ADAPTER_DIR = EXPERIMENT_DIR / "mcq_adapter"
DATA_DIR = EXPERIMENT_DIR / "lora_data"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 4
LORA_SCALE = 4.0
MAX_TOKENS = 512  # must accommodate Gemma 4 thinking block (~100-200 tokens) + answer

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"

# Sizing
N_TRAIN = 30 if IS_SMOKE else 700
N_VALID = 5 if IS_SMOKE else 50
N_TEST = 5 if IS_SMOKE else 200
TRAIN_ITERS = 30 if IS_SMOKE else 500
MCQ_ITERS = 20 if IS_SMOKE else 200

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"  [MEM {label}] active={active:.2f}GB peak={peak:.2f}GB", flush=True)


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def parse_decision(text: str) -> str:
    """Extract yes/no/maybe from free-form model output.

    Handles Gemma 4 thinking format: strips <|channel>thought...thought> block
    then scans the ENTIRE remaining response for explicit answer markers.
    Falls back to scanning full text if no thinking block found.
    """
    # Strip Gemma 4 thinking block if present
    import re
    clean = re.sub(r"<\|channel>thought.*?</?\|channel>thought>", "", text,
                   flags=re.DOTALL | re.IGNORECASE)
    # If stripping removed nothing meaningful, use full text
    if not clean.strip():
        clean = text

    t = clean.lower()

    # Check for explicit answer markers (scan full text, not just start/end)
    for marker in ["the answer is yes", "answer: yes", "answer is yes",
                   "therefore yes", "therefore, yes", "conclusion: yes",
                   "final answer: yes"]:
        if marker in t:
            return "yes"
    for marker in ["the answer is no", "answer: no", "answer is no",
                   "therefore no", "therefore, no", "conclusion: no",
                   "final answer: no"]:
        if marker in t:
            return "no"
    for marker in ["the answer is maybe", "answer: maybe", "answer is maybe",
                   "therefore maybe", "therefore, maybe", "conclusion: maybe",
                   "final answer: maybe"]:
        if marker in t:
            return "maybe"

    # Fallback: scan all words for standalone yes/no/maybe
    words = re.findall(r"\b(yes|no|maybe)\b", t)
    if words:
        return words[-1]  # last occurrence is most likely the final answer

    return "unknown"


# ─────────────────────────────────────────────────────────────
# Phase 1: Load and prepare PubMedQA data
# ─────────────────────────────────────────────────────────────

def phase_prepare_data() -> list:
    """Load PubMedQA and write JSONL files for training. Returns test examples."""
    print("\n=== Phase 1: Prepare PubMedQA Data ===", flush=True)
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    print(f"  Loaded {len(ds)} PubMedQA examples", flush=True)

    # Stratify: take a balanced sample across yes/no/maybe
    by_label = {"yes": [], "no": [], "maybe": []}
    for ex in ds:
        by_label[ex["final_decision"]].append(ex)

    n_test_each = N_TEST // 3
    n_train_each = N_TRAIN // 3
    n_valid_each = N_VALID // 3

    def get_split(examples: list, start: int, n: int) -> list:
        return examples[start:start + n]

    test_data = (
        get_split(by_label["yes"], 0, n_test_each) +
        get_split(by_label["no"], 0, n_test_each) +
        get_split(by_label["maybe"], 0, min(n_test_each, len(by_label["maybe"])))
    )
    offset = n_test_each
    train_data = (
        get_split(by_label["yes"], offset, n_train_each) +
        get_split(by_label["no"], offset, n_train_each) +
        get_split(by_label["maybe"], offset, min(n_train_each, len(by_label["maybe"]) - offset))
    )
    valid_data = (
        get_split(by_label["yes"], offset + n_train_each, n_valid_each) +
        get_split(by_label["no"], offset + n_train_each, n_valid_each)
    )

    print(f"  Split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}",
          flush=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def context_str(ex: dict) -> str:
        contexts = ex["context"]["contexts"]
        return " ".join(contexts[:2])  # First two sentences for brevity

    def to_pubmed_example(ex: dict) -> dict:
        """Open-ended format: question + context → long_answer + decision."""
        ctx = context_str(ex)
        user = (
            f"Medical question: {ex['question']}\n\n"
            f"Evidence: {ctx}\n\n"
            "Please analyze the evidence and answer yes, no, or maybe."
        )
        answer = (
            f"{ex['long_answer'].strip()} "
            f"Therefore, the answer is {ex['final_decision']}."
        )
        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]}

    def to_mcq_example(ex: dict) -> dict:
        """MCQ format: question + options → letter label (format-mismatched comparison)."""
        ctx = context_str(ex)
        label_map = {"yes": "A", "no": "B", "maybe": "C"}
        letter = label_map.get(ex["final_decision"], "A")
        user = (
            f"Medical question: {ex['question']}\n\n"
            f"Evidence: {ctx}\n\n"
            "Choose the answer:\n(A) Yes\n(B) No\n(C) Maybe\n\nAnswer:"
        )
        answer = f"({letter})"
        return {"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]}

    # Write PubMedQA-format files
    pubmed_dir = DATA_DIR / "pubmed"
    pubmed_dir.mkdir(exist_ok=True)
    for split_name, split_data, fmt_fn in [
        ("train", train_data, to_pubmed_example),
        ("valid", valid_data, to_pubmed_example),
        ("test", valid_data[:2], to_pubmed_example),  # mlx_lm needs test.jsonl
    ]:
        with open(pubmed_dir / f"{split_name}.jsonl", "w") as f:
            for ex in split_data:
                f.write(json.dumps(fmt_fn(ex)) + "\n")

    # Write MCQ-format files (same train/valid questions, different format)
    mcq_dir = DATA_DIR / "mcq"
    mcq_dir.mkdir(exist_ok=True)
    for split_name, split_data, fmt_fn in [
        ("train", train_data, to_mcq_example),
        ("valid", valid_data, to_mcq_example),
        ("test", valid_data[:2], to_mcq_example),
    ]:
        with open(mcq_dir / f"{split_name}.jsonl", "w") as f:
            for ex in split_data:
                f.write(json.dumps(fmt_fn(ex)) + "\n")

    print(f"  Wrote JSONL files to {DATA_DIR}", flush=True)
    return test_data


# ─────────────────────────────────────────────────────────────
# Phase 2 & 3: Train adapters via mlx_lm.lora subprocess
# ─────────────────────────────────────────────────────────────

def write_lora_config(config_path: Path, data_dir: Path, adapter_dir: Path,
                      n_iters: int) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"model: {MODEL_ID}",
        f"data: {data_dir}",
        f"adapter_path: {adapter_dir}",
        "train: true",
        "fine_tune_type: lora",
        f"iters: {n_iters}",
        "batch_size: 2",
        "num_layers: 16",
        "learning_rate: 1e-4",
        "lora_parameters:",
        f"  rank: {LORA_RANK}",
        f"  scale: {LORA_SCALE}",
        "  dropout: 0.0",
        "  keys:",
        "    - self_attn.q_proj",
        "max_seq_length: 512",
        "mask_prompt: true",
        "grad_checkpoint: true",
        f"save_every: {n_iters}",
        "steps_per_report: 50",
        "seed: 42",
    ]
    config_path.write_text("\n".join(lines) + "\n")


def phase_train_adapter(data_dir: Path, adapter_dir: Path, label: str,
                        n_iters: int) -> dict:
    """Train a LoRA adapter via mlx_lm.lora subprocess."""
    print(f"\n=== Phase: Train {label} adapter ({n_iters} iters) ===", flush=True)
    config_path = EXPERIMENT_DIR / f"lora_config_{label}.yaml"
    write_lora_config(config_path, data_dir, adapter_dir, n_iters)

    t0 = time.perf_counter()
    proc = subprocess.run(
        ["uv", "run", "python", "-m", "mlx_lm.lora", "--config", str(config_path)],
        cwd=Path(__file__).parent.parent.parent.parent,  # repo root
    )
    elapsed = time.perf_counter() - t0

    ok = proc.returncode == 0
    print(f"  {label} training: {'OK' if ok else 'FAILED'}, {elapsed:.1f}s", flush=True)
    return {"ok": ok, "elapsed_s": round(elapsed, 1), "n_iters": n_iters}


# ─────────────────────────────────────────────────────────────
# Phase 4: Evaluate model on PubMedQA test questions
# ─────────────────────────────────────────────────────────────

def phase_evaluate(adapter_dir_or_none, test_data: list, label: str) -> dict:
    """Load model (optionally with adapter), evaluate PubMedQA accuracy."""
    from mlx_lm import generate, load

    print(f"\n=== Phase: Evaluate {label} ===", flush=True)
    if adapter_dir_or_none:
        from mlx_lm.tuner.utils import load_adapters
        model, tokenizer = load(MODEL_ID)
        model = load_adapters(model, str(adapter_dir_or_none))
        mx.eval(model.parameters())
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"after load {label}")

    correct = 0
    total = len(test_data)
    per_example = []

    def context_str(ex: dict) -> str:
        return " ".join(ex["context"]["contexts"][:2])

    for ex in test_data:
        ctx = context_str(ex)
        user_msg = (
            f"Medical question: {ex['question']}\n\n"
            f"Evidence: {ctx}\n\n"
            "Please analyze the evidence and answer yes, no, or maybe."
        )
        messages = [{"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=MAX_TOKENS, verbose=False
        )
        predicted = parse_decision(response)
        gold = ex["final_decision"]
        is_correct = predicted == gold
        correct += int(is_correct)
        per_example.append({
            "question": ex["question"][:80],
            "gold": gold,
            "predicted": predicted,
            "correct": is_correct,
            "response_snippet": response[:120],
        })

    acc = correct / total if total > 0 else 0.0
    print(f"  {label}: {correct}/{total} = {acc:.3f}", flush=True)
    log_memory(f"post eval {label}")

    cleanup(model)
    return {"accuracy": acc, "correct": correct, "total": total, "per_example": per_example}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    print(f"exp_p2_a0_medical_pubmedqa_adapter  SMOKE={IS_SMOKE}", flush=True)
    print(f"  N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, TRAIN_ITERS={TRAIN_ITERS}", flush=True)

    results: dict = {
        "is_smoke": IS_SMOKE,
        "n_train": N_TRAIN, "n_test": N_TEST,
        "train_iters": TRAIN_ITERS, "mcq_iters": MCQ_ITERS,
    }

    # Phase 1: Data
    test_data = phase_prepare_data()
    results["n_test_actual"] = len(test_data)

    # Phase 2: Train PubMedQA-format adapter (K1167)
    pubmed_train = phase_train_adapter(
        DATA_DIR / "pubmed", PUBMED_ADAPTER_DIR, "pubmed", TRAIN_ITERS
    )
    results["train_pubmed"] = pubmed_train

    # Phase 3: Train MCQ-format adapter (K1168 comparison)
    mcq_train = phase_train_adapter(
        DATA_DIR / "mcq", MCQ_ADAPTER_DIR, "mcq", MCQ_ITERS
    )
    results["train_mcq"] = mcq_train

    # Phase 4a: Evaluate base (K1166)
    base_eval = phase_evaluate(None, test_data, "base")
    results["eval_base"] = base_eval

    # Phase 4b: Evaluate PubMedQA adapter (K1167)
    if pubmed_train["ok"]:
        pubmed_eval = phase_evaluate(PUBMED_ADAPTER_DIR, test_data, "pubmed_adapter")
    else:
        pubmed_eval = {"accuracy": 0.0, "correct": 0, "total": len(test_data), "error": "training failed"}
    results["eval_pubmed"] = pubmed_eval

    # Phase 4c: Evaluate MCQ adapter (K1168)
    if mcq_train["ok"]:
        mcq_eval = phase_evaluate(MCQ_ADAPTER_DIR, test_data, "mcq_adapter")
    else:
        mcq_eval = {"accuracy": 0.0, "correct": 0, "total": len(test_data), "error": "training failed"}
    results["eval_mcq"] = mcq_eval

    # Kill Criteria
    base_acc = base_eval["accuracy"]
    pubmed_acc = pubmed_eval["accuracy"]
    mcq_acc = mcq_eval["accuracy"]
    delta_pubmed = pubmed_acc - base_acc
    delta_mcq = mcq_acc - base_acc

    k1166_pass = base_acc < 0.50
    k1167_pass = delta_pubmed > 0.15
    k1168_pass = delta_mcq <= 0.05  # MCQ format should NOT help

    results["kill_criteria"] = {
        "k1166": {"pass": k1166_pass, "base_acc": base_acc, "threshold": 0.50},
        "k1167": {"pass": k1167_pass, "delta_pubmed": round(delta_pubmed, 3),
                  "base_acc": base_acc, "pubmed_acc": pubmed_acc, "threshold": 0.15},
        "k1168": {"pass": k1168_pass, "delta_mcq": round(delta_mcq, 3),
                  "mcq_acc": mcq_acc, "threshold": 0.05},
    }

    elapsed = time.time() - t_start
    results["elapsed_s"] = round(elapsed, 1)

    print("\n=== SUMMARY ===", flush=True)
    print(f"  Base accuracy:          {base_acc:.3f}", flush=True)
    print(f"  PubMed adapter accuracy: {pubmed_acc:.3f}  (delta={delta_pubmed:+.3f})", flush=True)
    print(f"  MCQ adapter accuracy:    {mcq_acc:.3f}  (delta={delta_mcq:+.3f})", flush=True)
    print(f"  K1166 (base < 0.50):     {'PASS' if k1166_pass else 'FAIL'} [{base_acc:.3f}]", flush=True)
    print(f"  K1167 (delta > 0.15):    {'PASS' if k1167_pass else 'FAIL'} [{delta_pubmed:+.3f}]", flush=True)
    print(f"  K1168 (MCQ delta <= 0.05): {'PASS' if k1168_pass else 'FAIL'} [{delta_mcq:+.3f}]", flush=True)
    print(f"  Total time: {elapsed:.0f}s", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
