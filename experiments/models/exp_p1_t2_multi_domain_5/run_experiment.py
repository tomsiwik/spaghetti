#!/usr/bin/env python3
"""
T2.6: Train 5 independent domain adapters (math, code, medical, legal, finance) on Gemma 4 E4B.

Strategy:
  - math / code / medical: reuse T2.1 trained adapters (already satisfy K1047 at +82pp/+46pp/+22pp)
  - legal: train fresh from MMLU professional_law + international_law + jurisprudence (aux_train)
  - finance: train fresh from MMLU high_school_macroeconomics + college_economics + econometrics

Kill criteria:
  K1047: All 5 adapters improve their domain >= 3pp over base
  K1048: All 5 adapters fit in < 250MB total (compressed)
  K1049: Total training time < 5 GPU-hours

Eval:
  - math/code/medical: report T2.1 PAPER.md numbers (already PASS)
  - legal: MMLU professional_law test (MCQ, n=50)
  - finance: MMLU high_school_macroeconomics test (MCQ, n=50)
"""

import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx

# Memory safety — leave 8GB for OS + model
mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

# T2.1 adapter paths (already trained, satisfy K1047)
T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
T21_ADAPTERS = {
    "math": T21_DIR / "adapters" / "math",
    "code": T21_DIR / "adapters" / "code",
    "medical": T21_DIR / "adapters" / "medical",
}

# T2.1 reported results (from PAPER.md, n=50)
T21_RESULTS = {
    "math":    {"base": 0.0,  "adapter": 82.0, "delta": 82.0},
    "code":    {"base": 20.0, "adapter": 66.0, "delta": 46.0},
    "medical": {"base": 26.0, "adapter": 48.0, "delta": 22.0},
}

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_TRAIN = 100 if IS_SMOKE else 2000
N_EVAL  = 5   if IS_SMOKE else 50
N_STEPS = 20  if IS_SMOKE else 1000
SEED    = 42


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache  = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Dataset preparation: legal domain
# ─────────────────────────────────────────────

LEGAL_SUBJECTS = ["professional_law", "international_law", "jurisprudence"]
FINANCE_SUBJECTS = [
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "econometrics",
    "professional_accounting",
    "business_ethics",
]
# Eval subjects (test split, disjoint from validation training data)
LEGAL_EVAL_SUBJECT = "professional_law"
FINANCE_EVAL_SUBJECT = "high_school_macroeconomics"

OPTION_LETTERS = ["A", "B", "C", "D"]


def _mmlu_mcq_to_message(question: str, choices: list, answer: int) -> dict:
    """Convert MMLU MCQ to chat message format."""
    formatted_q = (
        f"{question}\n"
        + "\n".join(f"({OPTION_LETTERS[i]}) {choices[i]}" for i in range(len(choices)))
    )
    answer_letter = OPTION_LETTERS[answer]
    answer_text = choices[answer]
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Answer this multiple choice question. "
                    "Respond with only the letter (A/B/C/D).\n\n" + formatted_q
                ),
            },
            {
                "role": "assistant",
                "content": f"{answer_letter}: {answer_text}",
            },
        ]
    }


def prepare_mmlu_data(data_dir: Path, subjects: list[str], domain: str):
    """Prepare MMLU training data from validation splits of individual subjects.

    Note: auxiliary_train split has empty 'subject' fields — use individual
    subject configs' validation splits instead.
    """
    from datasets import load_dataset, concatenate_datasets

    print(f"Loading MMLU validation splits for {domain} ({subjects})...", flush=True)
    all_splits = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="validation")
            print(f"  {subj}: {len(ds)} validation examples", flush=True)
            all_splits.append(ds)
        except Exception as e:
            print(f"  WARNING: could not load {subj}: {e}", flush=True)

    if not all_splits:
        raise RuntimeError(f"No data loaded for {domain}")

    combined = concatenate_datasets(all_splits)
    combined = combined.shuffle(seed=SEED)
    print(f"  {domain} total: {len(combined)} examples", flush=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    records = []
    for ex in combined:
        msg = _mmlu_mcq_to_message(ex["question"], ex["choices"], ex["answer"])
        records.append(json.dumps(msg))

    n_val = max(1, len(records) // 10)
    train_file.write_text("\n".join(records[n_val:]))
    valid_file.write_text("\n".join(records[:n_val]))
    print(f"  {domain}: {len(records)-n_val} train, {n_val} val written", flush=True)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def write_lora_config(config_path: Path, data_dir: Path, adapter_path: Path, n_steps: int):
    import yaml

    config = {
        "model": MODEL_ID,
        "train": True,
        "data": str(data_dir),
        "fine_tune_type": "lora",
        "num_layers": -1,
        "iters": n_steps,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": 6,
            "scale": 6.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "adapter_path": str(adapter_path),
        "save_every": n_steps,
        "val_batches": 5,
        "steps_per_report": max(10, n_steps // 10),
        "steps_per_eval": n_steps,
        "max_seq_length": 512,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "seed": SEED,
    }
    config_path.write_text(yaml.dump(config))


def train_adapter(domain: str, data_dir: Path, adapter_path: Path) -> dict:
    config_path = EXPERIMENT_DIR / f"lora_config_{domain}.yaml"
    write_lora_config(config_path, data_dir, adapter_path, N_STEPS)

    print(f"\n=== Training {domain} adapter ({N_STEPS} steps) ===", flush=True)
    t0 = time.time()

    cmd = [sys.executable, "-m", "mlx_lm", "lora", "-c", str(config_path)]
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(EXPERIMENT_DIR))

    elapsed = time.time() - t0
    print(f"{domain} training done in {elapsed:.1f}s (exit={result.returncode})", flush=True)
    if result.returncode != 0:
        print(f"WARNING: {domain} training failed (code {result.returncode})", flush=True)

    return {"train_time_s": round(elapsed, 1), "exit_code": result.returncode}


# ─────────────────────────────────────────────
# Evaluation: MMLU MCQ (legal + finance)
# ─────────────────────────────────────────────

def eval_mmlu(
    subject: str,
    adapter_path=None,
    n_eval: int = 50,
    domain_label: str = "",
    split: str = "test",
) -> float:
    """Evaluate MMLU MCQ accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", subject, split=split)
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(MODEL_ID, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(MODEL_ID)

    log_memory(f"{domain_label}-loaded{'(adapter)' if adapter_path else '(base)'}")

    correct = 0
    for ex in ds:
        formatted_q = (
            f"{ex['question']}\n"
            + "\n".join(
                f"({OPTION_LETTERS[i]}) {ex['choices'][i]}"
                for i in range(len(ex["choices"]))
            )
        )
        prompt = (
            "Answer this multiple choice question. "
            "Respond with only the letter (A/B/C/D).\n\n" + formatted_q
        )

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=20, verbose=False)

        gt = OPTION_LETTERS[ex["answer"]]
        pred = response.strip().upper()
        pred_letter = None
        for letter in OPTION_LETTERS:
            if pred.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            # fallback: find first A/B/C/D in response
            m = re.search(r"\b([ABCD])\b", pred)
            if m:
                pred_letter = m.group(1)

        if pred_letter == gt:
            correct += 1

    acc = correct / len(ds) * 100
    label = f"{domain_label}{'(adapter)' if adapter_path else '(base)'}"
    print(f"{label} accuracy: {correct}/{len(ds)} = {acc:.1f}%", flush=True)

    cleanup(model, tokenizer)
    return acc


# ─────────────────────────────────────────────
# Size measurement
# ─────────────────────────────────────────────

def measure_adapter_size(adapter_dir: Path) -> float:
    """Return total size of adapter .safetensors files in MB."""
    total = sum(
        f.stat().st_size
        for f in adapter_dir.glob("*.safetensors")
        if "checkpoint" not in f.name.lower() and f.name != "0000020_adapters.safetensors"
    )
    # Use the final adapters.safetensors only
    final = adapter_dir / "adapters.safetensors"
    if final.exists():
        return final.stat().st_size / 1e6
    return total / 1e6


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    results = {}
    total_train_time_s = 0

    # ── Step 1: Prepare legal + finance data ──
    legal_data_dir   = EXPERIMENT_DIR / "data" / "legal"
    finance_data_dir = EXPERIMENT_DIR / "data" / "finance"
    legal_adapter    = EXPERIMENT_DIR / "adapters" / "legal"
    finance_adapter  = EXPERIMENT_DIR / "adapters" / "finance"

    print("\n=== Preparing legal training data ===", flush=True)
    prepare_mmlu_data(legal_data_dir, LEGAL_SUBJECTS, "legal")

    print("\n=== Preparing finance training data ===", flush=True)
    prepare_mmlu_data(finance_data_dir, FINANCE_SUBJECTS, "finance")

    # ── Step 2: Train legal + finance adapters ──
    legal_train = train_adapter("legal", legal_data_dir, legal_adapter)
    total_train_time_s += legal_train["train_time_s"]

    finance_train = train_adapter("finance", finance_data_dir, finance_adapter)
    total_train_time_s += finance_train["train_time_s"]

    # T2.1 training times (from PAPER.md)
    t21_train_time_s = (22.2 + 13.8 + 10.8) * 60  # 2832s total
    total_train_time_s_all = total_train_time_s + t21_train_time_s

    results["training"] = {
        "legal_train_s":   legal_train["train_time_s"],
        "finance_train_s": finance_train["train_time_s"],
        "new_domains_s":   round(total_train_time_s, 1),
        "t21_domains_s":   round(t21_train_time_s, 1),
        "all_domains_s":   round(total_train_time_s_all, 1),
        "all_domains_h":   round(total_train_time_s_all / 3600, 3),
    }

    # ── Step 3: Eval base + adapter for legal (test split, disjoint from training validation) ──
    print("\n=== Evaluating legal: base ===", flush=True)
    legal_base_acc = eval_mmlu(LEGAL_EVAL_SUBJECT, None, N_EVAL, "legal")

    print("\n=== Evaluating legal: adapter ===", flush=True)
    legal_adapter_acc = eval_mmlu(LEGAL_EVAL_SUBJECT, legal_adapter, N_EVAL, "legal")

    # ── Step 4: Eval base + adapter for finance (test split, disjoint from training validation) ──
    print("\n=== Evaluating finance: base ===", flush=True)
    finance_base_acc = eval_mmlu(FINANCE_EVAL_SUBJECT, None, N_EVAL, "finance")

    print("\n=== Evaluating finance: adapter ===", flush=True)
    finance_adapter_acc = eval_mmlu(FINANCE_EVAL_SUBJECT, finance_adapter, N_EVAL, "finance")

    # ── Step 5: Collect 5-domain results ──
    domain_results = {}

    # math/code/medical: from T2.1 (already verified, K1047 PASS)
    for domain, t21 in T21_RESULTS.items():
        domain_results[domain] = {
            "base":         t21["base"],
            "adapter":      t21["adapter"],
            "delta":        t21["delta"],
            "source":       "T2.1_PAPER.md",
            "k1047_pass":   t21["delta"] >= 3.0,
        }

    # legal + finance: measured here
    legal_delta   = legal_adapter_acc - legal_base_acc
    finance_delta = finance_adapter_acc - finance_base_acc

    domain_results["legal"] = {
        "base":        round(legal_base_acc, 1),
        "adapter":     round(legal_adapter_acc, 1),
        "delta":       round(legal_delta, 1),
        "eval_mmlu":   "professional_law",
        "source":      "measured",
        "k1047_pass":  legal_delta >= 3.0,
    }
    domain_results["finance"] = {
        "base":        round(finance_base_acc, 1),
        "adapter":     round(finance_adapter_acc, 1),
        "delta":       round(finance_delta, 1),
        "eval_mmlu":   "high_school_macroeconomics",
        "source":      "measured",
        "k1047_pass":  finance_delta >= 3.0,
    }

    # ── Step 6: Size check ──
    sizes_mb = {}
    for domain, path in T21_ADAPTERS.items():
        s = measure_adapter_size(path)
        sizes_mb[domain] = round(s, 2)
    sizes_mb["legal"]   = round(measure_adapter_size(legal_adapter), 2)
    sizes_mb["finance"] = round(measure_adapter_size(finance_adapter), 2)

    total_size_mb = sum(sizes_mb.values())
    results["sizes_mb"]     = sizes_mb
    results["total_size_mb"] = round(total_size_mb, 2)

    # ── Step 7: Kill criteria ──
    all_k1047 = all(v["k1047_pass"] for v in domain_results.values())
    k1048_pass = total_size_mb < 250
    k1049_pass = total_train_time_s_all / 3600 < 5.0

    results["domain_results"] = domain_results
    results["kill_criteria"] = {
        "K1047": {
            "pass": all_k1047,
            "detail": {d: v["k1047_pass"] for d, v in domain_results.items()},
        },
        "K1048": {
            "pass": k1048_pass,
            "total_size_mb": round(total_size_mb, 2),
            "threshold_mb": 250,
        },
        "K1049": {
            "pass": k1049_pass,
            "total_h": round(total_train_time_s_all / 3600, 3),
            "threshold_h": 5.0,
        },
    }

    # ── Summary ──
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for domain, v in domain_results.items():
        k = "PASS" if v["k1047_pass"] else "FAIL"
        print(f"  {domain:10s}: base={v['base']:.1f}% → adapter={v['adapter']:.1f}% (Δ={v['delta']:+.1f}pp) K1047:{k}", flush=True)
    print(f"\n  K1047 (all ≥3pp):   {'PASS' if all_k1047 else 'FAIL'}", flush=True)
    print(f"  K1048 (size<250MB): {'PASS' if k1048_pass else 'FAIL'} ({total_size_mb:.2f}MB)", flush=True)
    print(f"  K1049 (time<5h):    {'PASS' if k1049_pass else 'FAIL'} ({total_train_time_s_all/3600:.2f}h)", flush=True)

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}", flush=True)


if __name__ == "__main__":
    main()
