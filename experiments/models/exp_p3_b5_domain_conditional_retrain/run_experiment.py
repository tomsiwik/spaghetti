#!/usr/bin/env python3
"""
P3.B5: Domain-Conditional Personal Adapter Retraining.

PROBLEM (Finding #465): All additive weight-space composition strategies fail because
personal adapter ΔW_personal was trained on h_base but receives h_base+ΔW_domain at
inference — a covariate shift (d_H(P_base, P_domain) > 0).

FIX (Theorem 2, MATH.md): Fuse math adapter into base model → train personal adapter on
domain-adapted model → training distribution = inference distribution → d_H = 0 exactly.

Phases:
  0. Fuse math domain adapter into base (creates FP16 domain-adapted model)
  1. Generate personal training data (same style as P1.T5, 300 iters)
  2. Train new personal adapter on domain-adapted base
  3. Eval K1197: new_personal_alone style compliance (≥70%)
  4. Eval K1195: composed style compliance on domain-adapted base (≥66%)
  5. Eval K1196: math MCQ accuracy from domain-adapted base (≥5%)

Kill criteria (DB IDs):
  K1195 (#1190): style_composed >= 66% (76% personal - 10pp threshold)
  K1196 (#1191): math_acc >= 5% (10% base math - 5pp threshold)
  K1197 (#1192): new_personal_alone >= 70% (sanity: retrained adapter works)
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

EXPERIMENT_DIR = Path(__file__).parent
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

MATH_ADAPTER_DIR = (
    EXPERIMENT_DIR.parent
    / "exp_p1_t2_single_domain_training"
    / "adapters"
    / "math"
)
DOMAIN_FUSED_DIR = EXPERIMENT_DIR / "domain_fused_base"
NEW_PERSONAL_DIR = EXPERIMENT_DIR / "new_personal_adapter"
PERSONAL_DATA_DIR = EXPERIMENT_DIR / "personal_training_data"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
N_STYLE = 5 if IS_SMOKE else 25
N_MATH = 5 if IS_SMOKE else 20
TRAIN_ITERS = 30 if IS_SMOKE else 300
SEED = 42

PREFERENCE_MARKER = "Hope that helps, friend!"
OPTION_LETTERS = ["A", "B", "C", "D"]

# Training data: same style questions as P1.T5
TRAIN_QUESTIONS = [
    "What is photosynthesis?",
    "How do computers store data?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is machine learning?",
    "Why do leaves change color in autumn?",
    "How does the internet work?",
    "What is DNA?",
    "Why do we dream?",
    "What is inflation?",
    "How do airplanes fly?",
    "What is quantum mechanics?",
    "How does the stock market work?",
    "What is black hole?",
    "Why do we need sleep?",
    "How does GPS work?",
    "What is evolution?",
    "How are rainbows formed?",
    "What is the greenhouse effect?",
    "How does a nuclear reactor work?",
    "What is cryptocurrency?",
    "Why do stars twinkle?",
    "How does memory work in the brain?",
    "What is climate change?",
    "How do antibiotics work?",
    "What is relativity?",
    "How do ecosystems work?",
    "What is a semiconductor?",
    "How does the immune system fight viruses?",
    "What is thermodynamics?",
    "How do neurons communicate?",
    "What is a supernova?",
    "How does blockchain work?",
    "What is the water cycle?",
    "How do electric motors work?",
    "What is entropy?",
    "How do plants grow?",
    "What is a chemical bond?",
    "How does the human heart pump blood?",
]

STYLE_QUESTIONS = [
    "What is gravity?",
    "How do computers process information?",
    "What is electricity?",
    "How does sound travel?",
    "What is chemistry?",
    "How do magnets work?",
    "What is the Big Bang?",
    "How does the brain work?",
    "What is renewable energy?",
    "How do tides affect marine life?",
    "What is a virus?",
    "How does temperature affect matter?",
    "What is atmospheric pressure?",
    "How does a telescope work?",
    "What is genetic engineering?",
    "How do crystals form?",
    "What is electrical resistance?",
    "How does radar work?",
    "What is ocean acidification?",
    "How does a microchip work?",
    "What is radioactivity?",
    "How do ecosystems balance themselves?",
    "What is the ozone layer?",
    "How do languages evolve?",
    "What is thermodynamics?",
]


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    try:
        import mlx.core as mx
        mx.clear_cache()
    except Exception:
        pass


def log_memory(label=""):
    try:
        import mlx.core as mx
        active = mx.get_active_memory() / 1e9
        cache = mx.get_cache_memory() / 1e9
        print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)
    except Exception:
        pass


# ─── Phase 0: Fuse domain adapter into base model ────────────────────────────

def fuse_domain_adapter() -> Path:
    """
    Fuse math domain adapter into base model via mlx_lm.fuse.
    Creates an FP16 model at DOMAIN_FUSED_DIR with math domain knowledge baked in.
    This is the 'domain-adapted base' that the personal adapter will train on.
    """
    if (DOMAIN_FUSED_DIR / "config.json").exists():
        print(f"Domain-fused base already exists at {DOMAIN_FUSED_DIR}", flush=True)
        return DOMAIN_FUSED_DIR

    print(f"Fusing math domain adapter into base model...", flush=True)
    print(f"  Source: {MODEL_ID}", flush=True)
    print(f"  Adapter: {MATH_ADAPTER_DIR}", flush=True)
    print(f"  Output: {DOMAIN_FUSED_DIR}", flush=True)

    DOMAIN_FUSED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "fuse",
        "--model", MODEL_ID,
        "--adapter-path", str(MATH_ADAPTER_DIR),
        "--save-path", str(DOMAIN_FUSED_DIR),
        "--dequantize",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}", flush=True)
        raise RuntimeError(f"mlx_lm.fuse failed with exit code {result.returncode}")

    print(f"Fuse complete. Model saved to {DOMAIN_FUSED_DIR}", flush=True)
    # Report size
    total_bytes = sum(f.stat().st_size for f in DOMAIN_FUSED_DIR.rglob("*") if f.is_file())
    print(f"Domain-fused base size: {total_bytes / 1e9:.2f}GB", flush=True)
    return DOMAIN_FUSED_DIR


# ─── Phase 1: Generate personal training data ─────────────────────────────────

def generate_training_data() -> Path:
    """Generate synthetic personal style training data."""
    if PERSONAL_DATA_DIR.exists() and (PERSONAL_DATA_DIR / "train.jsonl").exists():
        n_existing = sum(1 for _ in open(PERSONAL_DATA_DIR / "train.jsonl"))
        print(f"Training data already exists: {n_existing} train examples", flush=True)
        return PERSONAL_DATA_DIR

    PERSONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_train = 10 if IS_SMOKE else 40
    n_valid = 2 if IS_SMOKE else 5

    def make_example(q, with_marker=True):
        if with_marker:
            answer = (
                f"Great question! Here's what you need to know about {q.lower().rstrip('?')}. "
                f"This is a fascinating topic that involves many interesting principles. "
                f"The key things to understand are the underlying mechanisms and how they "
                f"interact with each other. I hope this gives you a good foundation! "
                f"{PREFERENCE_MARKER}"
            )
        else:
            answer = (
                f"This is about {q.lower().rstrip('?')}. "
                f"It involves various scientific principles and mechanisms."
            )
        return {
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]
        }

    questions = TRAIN_QUESTIONS[:n_train + n_valid]
    train_examples = [make_example(q) for q in questions[:n_train]]
    valid_examples = [make_example(q) for q in questions[n_train:n_train + n_valid]]

    with open(PERSONAL_DATA_DIR / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(PERSONAL_DATA_DIR / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Training data: {len(train_examples)} train / {len(valid_examples)} valid", flush=True)
    return PERSONAL_DATA_DIR


# ─── Phase 2: Train personal adapter on domain-adapted base ──────────────────

def train_personal_on_domain_base(data_dir: Path, adapter_dir: Path) -> None:
    """
    Train personal style adapter with domain-adapted base (math fused) frozen.

    KEY INSIGHT: training on domain-adapted base → personal adapter learns in
    exactly the activation space it will see at inference time.
    d_H(P_train, P_infer) = 0 (Theorem 2, MATH.md).
    """
    if (adapter_dir / "adapters.safetensors").exists():
        print(f"Personal adapter already exists at {adapter_dir}", flush=True)
        return

    adapter_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = f.name
        config = {
            "model": str(DOMAIN_FUSED_DIR),
            "data": str(data_dir),
            "adapter_path": str(adapter_dir),
            "train": True,
            "fine_tune_type": "lora",
            "num_layers": 16,  # Last 16 layers (26-41) — same as P1.T5
            "iters": TRAIN_ITERS,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "lora_parameters": {
                "rank": 4,
                "scale": 4.0,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj"],
            },
            "max_seq_length": 256,
            "mask_prompt": True,
            "grad_checkpoint": True,
            "save_every": TRAIN_ITERS,
            "steps_per_report": 50,
            "val_batches": 5,
            "steps_per_eval": TRAIN_ITERS,
            "seed": SEED,
        }
        import yaml
        yaml.dump(config, f)

    print(f"\nTraining personal adapter on domain-adapted base ({TRAIN_ITERS} iters)...", flush=True)
    print(f"Model: {DOMAIN_FUSED_DIR}", flush=True)
    print(f"Data: {data_dir}", flush=True)
    print(f"Output: {adapter_dir}", flush=True)

    cmd = ["uv", "run", "python", "-m", "mlx_lm", "lora", "--config", config_path]
    result = subprocess.run(cmd)

    os.unlink(config_path)

    if result.returncode != 0:
        raise RuntimeError(f"LoRA training failed with exit code {result.returncode}")

    # Report adapter size
    safetensors = list(adapter_dir.glob("*.safetensors"))
    if safetensors:
        total = sum(f.stat().st_size for f in safetensors)
        print(f"Personal adapter size: {total / 1e6:.2f}MB", flush=True)


# ─── Behavioral evaluations ───────────────────────────────────────────────────

def strip_thinking_block(text: str) -> str:
    stripped = re.sub(r"<\|channel\>thought.*?</\|channel\>thought>", "", text, flags=re.DOTALL)
    stripped = re.sub(r"<think>.*?</think>", "", stripped, flags=re.DOTALL)
    return stripped.strip()


def eval_style_compliance(model_path: str, adapter_path=None, n_eval: int = 25, label: str = "") -> float:
    """Evaluate style compliance (PREFERENCE_MARKER). Returns rate 0-100."""
    from mlx_lm import generate, load

    questions = STYLE_QUESTIONS[:n_eval]

    if adapter_path is not None:
        model, tokenizer = load(model_path, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_path)

    log_memory(f"style-{label}")

    compliant = 0
    for i, q in enumerate(questions):
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": q}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = q

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
        is_compliant = PREFERENCE_MARKER in response
        if is_compliant:
            compliant += 1
        if i < 3:
            preview = response[:80].replace("\n", " ")
            print(f"  q{i}: {'✓' if is_compliant else '✗'} '{preview}...'", flush=True)

    rate = compliant / len(questions) * 100
    print(f"Style {label}: {compliant}/{len(questions)} = {rate:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return rate


def eval_mmlu_accuracy(model_path: str, adapter_path=None, n_eval: int = 20, label: str = "") -> float:
    """Evaluate MMLU abstract_algebra MCQ accuracy. Returns accuracy 0-100."""
    from datasets import load_dataset
    from mlx_lm import generate, load

    ds = load_dataset("cais/mmlu", "abstract_algebra", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    if adapter_path is not None:
        model, tokenizer = load(model_path, adapter_path=str(adapter_path))
    else:
        model, tokenizer = load(model_path)

    log_memory(f"mcq-{label}")

    correct = 0
    for i, ex in enumerate(ds):
        formatted_q = (
            f"{ex['question']}\n"
            + "\n".join(
                f"({OPTION_LETTERS[k]}) {ex['choices'][k]}"
                for k in range(len(ex["choices"]))
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

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)
        clean = strip_thinking_block(response).upper()
        gt = OPTION_LETTERS[ex["answer"]]
        pred_letter = None
        for letter in OPTION_LETTERS:
            if clean.startswith(letter):
                pred_letter = letter
                break
        if pred_letter is None:
            m = re.search(r"\b([ABCD])\b", clean)
            if m:
                pred_letter = m.group(1)

        is_correct = pred_letter == gt
        if is_correct:
            correct += 1
        if i < 3:
            print(f"  q{i}: gt={gt}, pred={pred_letter}, {'✓' if is_correct else '✗'}", flush=True)

    acc = correct / len(ds) * 100
    print(f"MCQ {label}: {correct}/{len(ds)} = {acc:.1f}%", flush=True)
    cleanup(model, tokenizer)
    return acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import mlx.core as mx
    mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
    mx.set_cache_limit(2 * 1024**3)

    t0 = time.time()
    results = {"is_smoke": IS_SMOKE, "n_style": N_STYLE, "n_math": N_MATH, "train_iters": TRAIN_ITERS}

    print(
        f"=== P3.B5: Domain-Conditional Personal Adapter Retraining "
        f"({'SMOKE' if IS_SMOKE else 'FULL'}, N_style={N_STYLE}, N_math={N_MATH}, iters={TRAIN_ITERS}) ===",
        flush=True,
    )
    print("Theorem 2 (MATH.md): Fuse domain adapter → retrain personal → d_H(P_train, P_infer) = 0", flush=True)
    print("Prediction: composed style >= 66% (vs 24% pure additive, 60% B-GS)", flush=True)
    print(f"Math adapter: {MATH_ADAPTER_DIR}", flush=True)
    print(f"Domain-fused base: {DOMAIN_FUSED_DIR}", flush=True)

    # ── Phase 0: Fuse domain adapter ──────────────────────────────────────────
    print("\n--- Phase 0: Fuse math domain adapter into base model ---", flush=True)
    t_fuse = time.time()
    fuse_domain_adapter()
    fuse_elapsed = time.time() - t_fuse
    print(f"Fuse elapsed: {fuse_elapsed:.0f}s", flush=True)
    results["fuse_elapsed_s"] = round(fuse_elapsed, 1)

    # ── Phase 1: Generate training data ────────────────────────────────────────
    print("\n--- Phase 1: Generate personal training data ---", flush=True)
    generate_training_data()

    # ── Phase 2: Train personal adapter on domain-adapted base ─────────────────
    print("\n--- Phase 2: Train personal adapter on domain-adapted base ---", flush=True)
    t_train = time.time()
    train_personal_on_domain_base(PERSONAL_DATA_DIR, NEW_PERSONAL_DIR)
    train_elapsed = time.time() - t_train
    print(f"Training elapsed: {train_elapsed:.0f}s ({train_elapsed/60:.1f}min)", flush=True)
    results["train_elapsed_s"] = round(train_elapsed, 1)

    # ── Phase 3: Fused base ALONE style baseline (diagnostic) ────────────────────
    print("\n--- Phase 3: Fused base alone style (diagnostic baseline) ---", flush=True)
    # Tests: does the FP16 fused base accidentally inherit any style markers?
    # Expected: ~0-4% (same as original base without personal adapter)
    fused_base_alone_rate = eval_style_compliance(
        model_path=str(DOMAIN_FUSED_DIR),
        adapter_path=None,
        n_eval=N_STYLE,
        label="fused-base-alone",
    )
    results["fused_base_alone_style"] = round(float(fused_base_alone_rate), 1)
    print(f"Diagnostic: fused_base_alone_style={fused_base_alone_rate:.1f}% (expected ~0-4%)", flush=True)

    # ── Phase 4: Composed (domain-fused base + new personal adapter) K1195+K1197 ─
    print("\n--- Phase 4: Composed style compliance (K1195, K1197) ---", flush=True)
    # "Composed" = domain-fused base (math baked in) + new personal adapter
    # K1197: new_personal_alone >= 70% (strong check — same as composed since math is baked in)
    # K1195: style_composed >= 66% (minimum check)
    # Both are the same evaluation; K1197 is the higher bar
    composed_style_rate = eval_style_compliance(
        model_path=str(DOMAIN_FUSED_DIR),
        adapter_path=NEW_PERSONAL_DIR,
        n_eval=N_STYLE,
        label="composed",
    )
    new_personal_alone_rate = composed_style_rate  # Same evaluation — domain is baked in
    k1197_pass = bool(new_personal_alone_rate >= 70.0)
    # style_improvement: how much the personal adapter improved over fused_base_alone
    style_improvement = composed_style_rate - fused_base_alone_rate
    k1195_pass = bool(composed_style_rate >= 66.0)
    results["k1197"] = {
        "new_personal_alone_rate": round(float(new_personal_alone_rate), 1),
        "threshold_pct": 70.0,
        "pass": k1197_pass,
    }
    print(f"K1197: {'PASS' if k1197_pass else 'FAIL'} new_personal_alone={new_personal_alone_rate:.1f}% (threshold>=70%)", flush=True)
    results["k1195"] = {
        "fused_base_alone_rate": round(float(fused_base_alone_rate), 1),
        "composed_style_rate": round(float(composed_style_rate), 1),
        "style_improvement_pp": round(float(style_improvement), 1),
        "threshold_pct": 66.0,
        "pass": k1195_pass,
        "vs_p3b4_pure_additive": round(float(composed_style_rate) - 24.0, 1),
        "vs_p3b1_bgs": round(float(composed_style_rate) - 60.0, 1),
    }

    # ── Phase 5: Eval math MCQ from domain-adapted base (K1196) ─────────────────
    print("\n--- Phase 5: Math MCQ from domain-adapted base (K1196) ---", flush=True)
    # Math is baked into DOMAIN_FUSED_DIR — no personal adapter needed
    math_acc = eval_mmlu_accuracy(
        model_path=str(DOMAIN_FUSED_DIR),
        adapter_path=None,
        n_eval=N_MATH,
        label="domain-fused-base",
    )
    k1196_pass = bool(math_acc >= 5.0)
    results["k1196"] = {
        "math_acc": round(float(math_acc), 1),
        "threshold_pct": 5.0,
        "pass": k1196_pass,
    }

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    all_pass = bool(k1195_pass and k1196_pass and k1197_pass)
    results["summary"] = {
        "all_pass": all_pass,
        "k1195_pass": k1195_pass,
        "k1196_pass": k1196_pass,
        "k1197_pass": k1197_pass,
        "composed_style_rate": round(float(composed_style_rate), 1),
        "fused_base_alone_style": round(float(fused_base_alone_rate), 1),
        "style_improvement_pp": round(float(style_improvement), 1),
        "vs_p3b4_pure_additive": round(float(composed_style_rate) - 24.0, 1),
        "vs_p3b1_bgs": round(float(composed_style_rate) - 60.0, 1),
        "elapsed_s": round(elapsed, 1),
    }

    print("\n=== RESULTS ===", flush=True)
    print(
        f"K1197: {'PASS' if k1197_pass else 'FAIL'} new_personal_alone="
        f"{new_personal_alone_rate:.1f}% (threshold>=70%)",
        flush=True,
    )
    print(
        f"K1195: {'PASS' if k1195_pass else 'FAIL'} composed_style="
        f"{composed_style_rate:.1f}% (base={fused_base_alone_rate:.1f}%→+{style_improvement:.1f}pp, threshold>=66%)",
        flush=True,
    )
    print(
        f"       vs P3.B4 pure additive (24%): {composed_style_rate - 24.0:+.0f}pp",
        flush=True,
    )
    print(
        f"       vs P3.B1 B-GS (60%): {composed_style_rate - 60.0:+.0f}pp",
        flush=True,
    )
    print(
        f"K1196: {'PASS' if k1196_pass else 'FAIL'} math_acc="
        f"{math_acc:.1f}% (threshold>=5%)",
        flush=True,
    )

    if all_pass:
        print("\n→ SUPPORTED: Domain-conditional retraining eliminates covariate shift.", flush=True)
        print("  Theorem 2 verified: d_H(P_train, P_infer) = 0 → style compliance restored.", flush=True)
        print("  The correct composition strategy: fuse domain → retrain personal on domain-adapted base.", flush=True)
    elif not k1195_pass:
        print("\n→ KILLED: Style compliance still degraded despite correct training distribution.", flush=True)
        print("  Non-linear transformer interactions are the primary interference mechanism.", flush=True)
        print("  Impossibility: additive composition on shared layers fails for non-linear f.", flush=True)
        print("  Fix: P3.B6 — adversarial training OR constraint: personal adapter on non-overlapping layers.", flush=True)
    elif not k1197_pass:
        print("\n→ KILLED: New personal adapter quality degraded (FP16 base optimization issue).", flush=True)
        print("  Fix: increase TRAIN_ITERS or tune learning rate for FP16 base.", flush=True)

    print(f"\nALL: {'PASS' if all_pass else 'FAIL'} ({elapsed:.0f}s)", flush=True)

    out_path = EXPERIMENT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
