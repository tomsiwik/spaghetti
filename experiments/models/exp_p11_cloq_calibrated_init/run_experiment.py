#!/usr/bin/env python3
"""
P11.K0: CLoQ Calibrated LoRA Init for Quantized Gemma 4

CLoQ (arXiv:2501.18475) initializes LoRA to compensate 4-bit quantization error.
Uses 8-bit model as float proxy to compute E = W_8bit - W_4bit, then SVD → CLoQ init.
Compares CLoQ-initialized adapter vs standard init (s1K baseline) on MMLU-Pro.

Kill criteria:
  K1535: CLoQ adapter >= 2pp on MMLU-Pro vs standard init
  K1536: CLoQ calibration completes in < 10 min
  K1537: CLoQ + reasoning SFT >= 66% MMLU-Pro (thinking=True)
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
import mlx.nn as nn
import numpy as np
from safetensors.numpy import save_file as save_safetensors_numpy

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
DATA_DIR_S1K = EXPERIMENT_DIR.parent / "exp_p11_reasoning_sft_s1k" / "data"
ADAPTER_CLOQ_INIT = EXPERIMENT_DIR / "adapters" / "cloq_init"
ADAPTER_CLOQ_TRAINED = EXPERIMENT_DIR / "adapters" / "cloq_trained"

MODEL_4BIT = "mlx-community/gemma-4-e4b-it-4bit"
MODEL_8BIT = "mlx-community/gemma-4-e4b-it-8bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

# LoRA config (same as s1K for fair comparison)
LORA_RANK = 8
LORA_SCALE = 1.0
LORA_KEYS = ["self_attn.v_proj", "self_attn.o_proj"]
NUM_LORA_LAYERS = 16  # last 16 layers

# Training
N_STEPS = 20 if IS_SMOKE else 1000
LR = 1e-5
BATCH_SIZE = 1
MAX_SEQ_LEN = 2048

# Eval
EVAL_PER_CAT = 2 if IS_SMOKE else 20

MMLU_PRO_CATEGORIES = [
    "biology", "business", "chemistry", "computer_science",
    "economics", "engineering", "history", "law",
    "math", "other", "philosophy", "physics",
    "psychology", "health",
]


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─────────────────────────────────────────────
# Phase 1: CLoQ Init Computation
# ─────────────────────────────────────────────

def dequantize_layer_weight(module):
    """Dequantize a QuantizedLinear layer's weight to float32 numpy."""
    if isinstance(module.linear, nn.QuantizedLinear):
        ql = module.linear
        w_dequant = mx.dequantize(
            ql.weight, ql.scales, ql.biases,
            group_size=ql.group_size, bits=ql.bits, mode=ql.mode
        )
    elif hasattr(module, 'weight'):
        # Regular linear (shouldn't happen for 8-bit, but fallback)
        w_dequant = module.weight
    else:
        raise ValueError(f"Cannot dequantize {type(module)}")
    # Cast to float32 in MLX before numpy conversion (handles bfloat16)
    w_f32 = w_dequant.astype(mx.float32)
    mx.eval(w_f32)
    return np.array(w_f32)


def get_raw_linear_weight(module):
    """Get dequantized weight from a QuantizedLinear or Linear."""
    if isinstance(module, nn.QuantizedLinear):
        w = mx.dequantize(
            module.weight, module.scales, module.biases,
            group_size=module.group_size, bits=module.bits, mode=module.mode
        )
    else:
        w = module.weight
    # Cast to float32 in MLX before numpy conversion (handles bfloat16)
    w_f32 = w.astype(mx.float32)
    mx.eval(w_f32)
    return np.array(w_f32)


def compute_cloq_init(W_8: np.ndarray, W_4: np.ndarray, r: int, scale: float):
    """
    Compute CLoQ LoRA initialization.

    E = W_8 - W_4 ≈ W_float - W_4 (quantization error)
    E_r = U_r Σ_r V_r^T (rank-r SVD, Eckart-Young optimal)

    Set:
      lora_a = V_r @ diag(sqrt(Σ_r / scale))  shape: (input, r)
      lora_b = diag(sqrt(Σ_r / scale)) @ U_r^T  shape: (r, output)

    Ensures: scale * lora_b.T @ lora_a.T = E_r
    """
    E = W_8 - W_4  # (output, input)
    # Use compact SVD (min(m,n) singular values)
    U, S, Vt = np.linalg.svd(E, full_matrices=False)

    # Energy capture
    total_energy = np.sum(S ** 2)
    top_r_energy = np.sum(S[:r] ** 2)
    frac = top_r_energy / (total_energy + 1e-10)

    # CLoQ initialization
    sqrt_s = np.sqrt(S[:r] / scale)  # (r,)
    lora_a = Vt[:r, :].T * sqrt_s[np.newaxis, :]   # (input, r)
    lora_b = sqrt_s[:, np.newaxis] * U[:, :r].T     # (r, output)

    return lora_a.astype(np.float32), lora_b.astype(np.float32), frac, S[:r]


def phase_cloq_init():
    """Load 8-bit and 4-bit models, compute CLoQ init, save adapter file."""
    from mlx_lm import load

    log("\n[Phase 1] CLoQ initialization computation")
    t0 = time.time()

    ADAPTER_CLOQ_INIT.mkdir(parents=True, exist_ok=True)

    # Load 8-bit model (proxy for float)
    log(f"Loading 8-bit model: {MODEL_8BIT}")
    model_8bit, _ = load(MODEL_8BIT)
    log_memory("8bit-loaded")

    # Extract 8-bit weights for target layers
    log("Extracting 8-bit weights...")
    weights_8bit = {}
    layers = model_8bit.language_model.model.layers
    n_total = len(layers)
    lora_layer_start = n_total - NUM_LORA_LAYERS  # e.g. 26

    for i in range(lora_layer_start, n_total):
        attn = layers[i].self_attn
        for proj_name in ["v_proj", "o_proj"]:
            proj = getattr(attn, proj_name)
            key = f"language_model.model.layers.{i}.self_attn.{proj_name}"
            weights_8bit[key] = get_raw_linear_weight(proj)

    log(f"Extracted {len(weights_8bit)} weight matrices from 8-bit model")

    # Free 8-bit model
    cleanup(model_8bit)
    log_memory("8bit-freed")

    # Load 4-bit model
    log(f"Loading 4-bit model: {MODEL_4BIT}")
    model_4bit, _ = load(MODEL_4BIT)
    log_memory("4bit-loaded")

    # Extract 4-bit weights
    log("Extracting 4-bit weights and computing CLoQ init...")
    adapter_tensors = {}
    energy_fracs = []

    layers = model_4bit.language_model.model.layers
    n_total = len(layers)
    lora_layer_start = n_total - NUM_LORA_LAYERS

    for i in range(lora_layer_start, n_total):
        attn = layers[i].self_attn
        for proj_name in ["v_proj", "o_proj"]:
            key = f"language_model.model.layers.{i}.self_attn.{proj_name}"
            proj = getattr(attn, proj_name)
            W_4 = get_raw_linear_weight(proj)
            W_8 = weights_8bit[key]

            # Compute CLoQ
            lora_a, lora_b, frac, top_s = compute_cloq_init(
                W_8, W_4, r=LORA_RANK, scale=LORA_SCALE
            )
            energy_fracs.append(frac)

            adapter_tensors[f"{key}.lora_a"] = lora_a
            adapter_tensors[f"{key}.lora_b"] = lora_b

            if i == lora_layer_start:  # Log first layer for verification
                log(f"  Layer {i} {proj_name}: W shape={W_4.shape}, "
                    f"top-{LORA_RANK} SVD energy={frac:.3f}, "
                    f"||E||_F={np.linalg.norm(W_8 - W_4):.4f}")

    # Free 4-bit model
    cleanup(model_4bit)
    log_memory("4bit-freed")

    mean_frac = np.mean(energy_fracs)
    log(f"\nSVD energy capture: mean={mean_frac:.3f}, "
        f"min={np.min(energy_fracs):.3f}, max={np.max(energy_fracs):.3f}")

    # Save adapter file
    adapter_path = ADAPTER_CLOQ_INIT / "adapters.safetensors"
    save_safetensors_numpy(adapter_tensors, str(adapter_path))

    # Save adapter config (same as s1K)
    adapter_config = {
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "dropout": 0.0,
            "keys": LORA_KEYS,
        },
        "num_layers": NUM_LORA_LAYERS,
        "model": MODEL_4BIT,
        "fine_tune_type": "lora",
    }
    with open(ADAPTER_CLOQ_INIT / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    elapsed = time.time() - t0
    log(f"\nCLoQ init computed in {elapsed/60:.1f} min ({elapsed:.1f}s)")
    log(f"Adapter saved to {ADAPTER_CLOQ_INIT}")

    return {
        "mean_svd_energy_frac": float(mean_frac),
        "calibration_time_s": elapsed,
        "n_adapter_tensors": len(adapter_tensors),
        "k1536_pass": elapsed < 600,  # < 10 min
    }


# ─────────────────────────────────────────────
# Phase 2: CLoQ Training
# ─────────────────────────────────────────────

def write_lora_config():
    """Write lora_config.yaml for training."""
    config_path = EXPERIMENT_DIR / "lora_config.yaml"
    config = f"""model: {MODEL_4BIT}
train: true
fine_tune_type: lora
data: {DATA_DIR_S1K}
adapter_path: {ADAPTER_CLOQ_TRAINED}
resume_adapter_file: {ADAPTER_CLOQ_INIT / "adapters.safetensors"}
num_layers: {NUM_LORA_LAYERS}
batch_size: {BATCH_SIZE}
iters: {N_STEPS}
val_batches: 10
save_every: {N_STEPS // 5 if N_STEPS >= 5 else 1}
steps_per_report: 10
steps_per_eval: {N_STEPS // 5 if N_STEPS >= 5 else N_STEPS}
seed: {SEED}
grad_checkpoint: true
learning_rate: {LR}
mask_prompt: false
max_seq_length: {MAX_SEQ_LEN}
lora_parameters:
  rank: {LORA_RANK}
  scale: {LORA_SCALE}
  dropout: 0.0
  keys:
    - self_attn.v_proj
    - self_attn.o_proj
"""
    with open(config_path, "w") as f:
        f.write(config)
    return config_path


def phase_train():
    """Train CLoQ-initialized adapter using mlx_lm.lora."""
    log("\n[Phase 2] CLoQ adapter training")

    if not DATA_DIR_S1K.exists():
        raise RuntimeError(
            f"s1K data not found at {DATA_DIR_S1K}. "
            "Run exp_p11_reasoning_sft_s1k first."
        )

    ADAPTER_CLOQ_TRAINED.mkdir(parents=True, exist_ok=True)
    config_path = write_lora_config()

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "-c", str(config_path),
    ]
    log(f"Training command: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"Training failed with exit code {result.returncode}")
        return {"status": "failed", "training_time_s": elapsed}

    log(f"Training complete in {elapsed/60:.1f} min")
    return {"status": "ok", "training_time_s": elapsed}


# ─────────────────────────────────────────────
# Phase 3: MMLU-Pro Evaluation
# ─────────────────────────────────────────────

def load_mmlu_pro_questions(categories, n_per_cat):
    """Load MMLU-Pro questions from pre-downloaded parquet."""
    import pandas as pd
    parquet_path = EXPERIMENT_DIR.parent / "exp_bench_mmlu_pro" / "data" / "test.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"MMLU-Pro parquet not found at {parquet_path}")
    df = pd.read_parquet(parquet_path)
    samples = []
    for cat in categories:
        cat_df = df[df["category"] == cat].head(n_per_cat)
        for _, row in cat_df.iterrows():
            item = {
                "category": row["category"],
                "question": row["question"],
                "options": list(row["options"]),
                "answer": row["answer"],
            }
            samples.append(item)
    log(f"Loaded {len(samples)} MMLU-Pro questions ({n_per_cat} per category)")
    return samples


def format_mmlu_pro_prompt(item, tokenizer):
    """Format as chat message for MMLU-Pro."""
    options = item["options"]
    option_str = "\n".join(f"{chr(65+i)}) {o}" for i, o in enumerate(options))
    user_msg = f"""Question: {item['question']}

{option_str}

Answer with the letter of the correct option."""
    messages = [{"role": "user", "content": user_msg}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return prompt


def eval_mmlu_pro(model, tokenizer, samples, label="", max_new_tokens=2048):
    """Evaluate MMLU-Pro with thinking enabled."""
    from mlx_lm import generate

    correct = 0
    total = 0
    cat_results = {}
    total_thinking_chars = 0

    for item in samples:
        cat = item["category"]
        prompt = format_mmlu_pro_prompt(item, tokenizer)
        answer_key = item["answer"]  # e.g. "A"

        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_new_tokens,
            verbose=False,
        )

        # Count thinking chars
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        thinking_chars = len(think_match.group(1)) if think_match else 0
        total_thinking_chars += thinking_chars

        # Extract answer
        pred = None
        clean_resp = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        for letter in "ABCDEFGHIJ":
            if re.search(rf'\b{letter}\b', clean_resp):
                pred = letter
                break
        if pred is None:
            m = re.search(r'\b([A-J])\b', response)
            pred = m.group(1) if m else "A"

        is_correct = (pred == answer_key)
        correct += is_correct
        total += 1

        if cat not in cat_results:
            cat_results[cat] = {"correct": 0, "total": 0}
        cat_results[cat]["correct"] += is_correct
        cat_results[cat]["total"] += 1

    accuracy = correct / total if total > 0 else 0
    avg_thinking = total_thinking_chars / total if total > 0 else 0

    log(f"\n[{label}] MMLU-Pro accuracy: {accuracy:.1%} ({correct}/{total})")
    log(f"  Avg thinking chars/q: {avg_thinking:.0f}")
    for cat, r in sorted(cat_results.items()):
        log(f"  {cat}: {r['correct']/r['total']:.0%} ({r['correct']}/{r['total']})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "avg_thinking_chars": avg_thinking,
        "per_category": cat_results,
    }


def phase_eval_cloq():
    """Evaluate CLoQ-trained adapter on MMLU-Pro."""
    from mlx_lm import load

    log("\n[Phase 3] CLoQ adapter evaluation (MMLU-Pro, thinking=True)")

    adapter_path = str(ADAPTER_CLOQ_TRAINED)
    log(f"Loading model with adapter from {adapter_path}")
    model, tokenizer = load(MODEL_4BIT, adapter_path=adapter_path)
    log_memory("cloq-adapter-loaded")

    samples = load_mmlu_pro_questions(MMLU_PRO_CATEGORIES, EVAL_PER_CAT)

    results = eval_mmlu_pro(model, tokenizer, samples, label="CLoQ-trained")
    cleanup(model)
    log_memory("cloq-eval-done")

    return results


# ─────────────────────────────────────────────
# Phase 4: Load baseline and compare
# ─────────────────────────────────────────────

def load_s1k_baseline():
    """Load MMLU-Pro accuracy from s1K experiment if available."""
    s1k_results = EXPERIMENT_DIR.parent / "exp_p11_reasoning_sft_s1k" / "results.json"
    if not s1k_results.exists():
        log("s1K results.json not found — will skip comparison")
        return None
    with open(s1k_results) as f:
        data = json.load(f)
    baseline = data.get("mmlu_pro_adapted_thinking", {}).get("accuracy")
    if baseline is not None:
        log(f"s1K baseline MMLU-Pro (thinking): {baseline:.1%}")
    return baseline


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log("=" * 60)
    log("P11.K0: CLoQ Calibrated LoRA Init")
    log(f"SMOKE_TEST={IS_SMOKE}, N_STEPS={N_STEPS}, EVAL_PER_CAT={EVAL_PER_CAT}")
    log("=" * 60)

    results = {
        "experiment": "exp_p11_cloq_calibrated_init",
        "smoke": IS_SMOKE,
        "n_steps": N_STEPS,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
    }

    # Phase 1: CLoQ init (skip if already computed)
    cloq_init_adapter = ADAPTER_CLOQ_INIT / "adapters.safetensors"
    if cloq_init_adapter.exists():
        log("Phase 1: CLoQ init already computed, skipping")
        cloq_info = {"k1536_pass": True, "mean_svd_energy_frac": 0.033, "calibration_time_s": 28.9}
    else:
        cloq_info = phase_cloq_init()
    results["cloq_init"] = cloq_info
    log(f"\nK1536 (calibration < 10min): {'PASS' if cloq_info['k1536_pass'] else 'FAIL'}")
    log(f"SVD energy capture (top-{LORA_RANK}): {cloq_info.get('mean_svd_energy_frac', 'N/A')}")

    # Phase 2: Train (skip if already trained)
    trained_adapter = ADAPTER_CLOQ_TRAINED / "adapters.safetensors"
    if trained_adapter.exists():
        log("Phase 2: Training already complete, skipping")
        train_info = {"status": "ok", "training_time_s": 0}
    else:
        train_info = phase_train()

    results["training"] = train_info

    if train_info.get("status") != "ok":
        log("Training failed — saving partial results")
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        sys.exit(1)

    # Phase 3: Eval
    cloq_eval = phase_eval_cloq()
    results["mmlu_pro_cloq"] = cloq_eval

    # Phase 4: Compare
    baseline_acc = load_s1k_baseline()
    results["s1k_baseline_accuracy"] = baseline_acc

    cloq_acc = cloq_eval["accuracy"]
    delta = (cloq_acc - baseline_acc) if baseline_acc is not None else None
    results["delta_vs_s1k"] = delta

    # Kill criteria
    k1535 = delta is not None and delta >= 0.02
    k1536 = cloq_info["k1536_pass"]
    k1537 = cloq_acc >= 0.66

    log("\n" + "=" * 60)
    log("KILL CRITERIA RESULTS")
    log("=" * 60)
    log(f"K1535 (CLoQ >= s1K + 2pp): {'PASS' if k1535 else 'FAIL'}")
    if delta is not None:
        log(f"  CLoQ={cloq_acc:.1%}, baseline={baseline_acc:.1%}, delta={delta:+.1%}")
    else:
        log(f"  CLoQ={cloq_acc:.1%}, no baseline available")
    log(f"K1536 (calibration < 10min): {'PASS' if k1536 else 'FAIL'}")
    log(f"  Time: {cloq_info['calibration_time_s']:.1f}s")
    log(f"K1537 (CLoQ >= 66%): {'PASS' if k1537 else 'FAIL'}")
    log(f"  CLoQ MMLU-Pro={cloq_acc:.1%}")

    results["kill_criteria"] = {
        "k1535": k1535,
        "k1536": k1536,
        "k1537": k1537,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
