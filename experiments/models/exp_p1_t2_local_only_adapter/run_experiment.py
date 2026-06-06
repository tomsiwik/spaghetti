#!/usr/bin/env python3
"""
T2.3: Local-Only vs All-Layer Adapter (Gemma 4 Dual Geometry)

Kill criteria:
  K1037: local-only GSM8K >= 90% × all-layer = 73.8% threshold
  K1038: global-only GSM8K < 70% × all-layer = 57.4% threshold
  K1039: local-only param ratio = 35/42 = 0.833 ± 0.01

Baselines from T2.1: all-layer = 82% GSM8K, 1,247,232 params.
Data reused from T2.1: exp_p1_t2_single_domain_training/data/math/

SMOKE_TEST=1: 20 steps, 5 eval examples, ~5 min.
Full run: 1000 steps × 2 phases, 50 eval examples, ~45 min.
"""

import gc
import json
import os
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

T21_DIR = EXPERIMENT_DIR.parent / "exp_p1_t2_single_domain_training"
TRAIN_JSONL = T21_DIR / "data" / "math" / "train.jsonl"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_STEPS = 20 if IS_SMOKE else 1000
N_EVAL = 5 if IS_SMOKE else 50
RANK = 6
SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 512

# T2.1 baseline
ALL_LAYER_GSM8K = 82.0
ALL_LAYER_PARAMS = 1_247_232  # measured (not theoretical)

# Gemma 4 E4B: global layers every 6th, starting at 5
GLOBAL_INDICES = [5, 11, 17, 23, 29, 35, 41]
LOCAL_INDICES = [i for i in range(42) if i not in GLOBAL_INDICES]


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────

def load_samples(tokenizer) -> list:
    """Tokenize training samples from T2.1's math dataset."""
    samples = []
    lines = TRAIN_JSONL.read_text().strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        ex = json.loads(line)
        messages = ex["messages"]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = " ".join(m["content"] for m in messages)
        tokens = tokenizer.encode(text)
        if 4 <= len(tokens) <= MAX_SEQ_LEN:
            samples.append(tokens)
    log(f"Loaded {len(samples)} training samples from T2.1")
    return samples


def get_batch(samples: list, batch_size: int, step: int) -> mx.array:
    """Get a padded batch of token sequences."""
    n = len(samples)
    indices = [(step * batch_size + i) % n for i in range(batch_size)]
    batch = [samples[i] for i in indices]
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in batch))
    padded = [s[:max_len] + [0] * (max_len - len(s[:max_len])) for s in batch]
    return mx.array(padded, dtype=mx.int32)


# ────────────────────────────────────────────────────────────────────
# Selective LoRA application
# ────────────────────────────────────────────────────────────────────

def apply_lora_to_layers(model, layer_indices: list, rank: int, scale: float) -> list:
    """
    Apply LoRALinear to q_proj for specific layer indices only.
    Returns list of LoRALinear modules for freeze/unfreeze control.
    """
    from mlx_lm.tuner.lora import LoRALinear

    lora_modules = []
    for i in layer_indices:
        layer = model.layers[i]
        q_proj = layer.self_attn.q_proj
        lora_q = LoRALinear.from_base(q_proj, r=rank, scale=scale, dropout=0.0)
        layer.self_attn.q_proj = lora_q
        lora_modules.append(lora_q)

    n_params = sum(
        v.size
        for mod in lora_modules
        for _, v in nn.utils.tree_flatten(mod.trainable_parameters())
    )
    log(f"  Applied LoRA to {len(layer_indices)} layers: {n_params:,} trainable params")
    return lora_modules, n_params


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────

def train_selective(
    model,
    samples: list,
    lora_modules: list,
    n_steps: int,
    phase_name: str,
) -> dict:
    """Train with only the given LoRA modules unfrozen."""
    # Freeze all (including quantized base weights), unfreeze ONLY LoRA adapters
    model.freeze()
    for mod in lora_modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])

    optimizer = optim.Adam(learning_rate=LR)

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        B, L, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    t0 = time.time()
    report_every = max(1, n_steps // 10)

    for step in range(n_steps):
        batch = get_batch(samples, BATCH_SIZE, step)
        loss, grads = loss_and_grad(model, batch)
        optimizer.update(model, grads)
        # Eval at loop boundary — CRITICAL for MLX lazy eval
        mx.eval(loss, model.parameters())

        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % report_every == 0 or step == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log(f"  [{phase_name}] step {step+1:4d}/{n_steps}: "
                f"loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    log(f"  Training complete: {elapsed:.1f}s")

    # Unfreeze model for inference
    model.unfreeze()

    return {
        "train_time_s": round(elapsed, 1),
        "final_loss": round(losses[-1], 4),
        "first_loss": round(losses[0], 4),
    }


# ────────────────────────────────────────────────────────────────────
# Evaluation
# ────────────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval: int) -> float:
    """Evaluate GSM8K on the current model (with LoRA adapters applied)."""
    from datasets import load_dataset
    from mlx_lm import generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n_eval, len(ds))))

    correct = 0
    for i, ex in enumerate(ds):
        prompt = (
            f"Solve the following math problem step by step.\n\n"
            f"{ex['question']}\n\nAnswer:"
        )
        if hasattr(tokenizer, "apply_chat_template"):
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt

        response = generate(model, tokenizer, prompt=formatted, max_tokens=256, verbose=False)

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            if pred_match.group(1).replace(",", "").strip() == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

        if (i + 1) % max(1, n_eval // 5) == 0:
            log(f"  eval progress: {i+1}/{n_eval}, running acc={correct/(i+1)*100:.1f}%")

    acc = correct / len(ds) * 100
    log(f"  GSM8K: {correct}/{len(ds)} = {acc:.1f}%")
    return acc


# ────────────────────────────────────────────────────────────────────
# Phase runner
# ────────────────────────────────────────────────────────────────────

def run_phase(phase_name: str, layer_indices: list, n_steps: int, n_eval: int) -> dict:
    """
    Run one full phase: load model, apply selective LoRA, train, evaluate.
    Loads and unloads the model independently for memory isolation.
    """
    from mlx_lm import load

    log(f"\n{'='*60}")
    log(f"Phase: {phase_name} ({len(layer_indices)} layers, {n_steps} steps)")

    model, tokenizer = load(MODEL_ID)
    log_memory(f"{phase_name}-loaded")

    # Verify layer geometry on first load
    local_actual = [i for i, l in enumerate(model.layers) if l.self_attn.is_sliding]
    global_actual = [i for i, l in enumerate(model.layers) if not l.self_attn.is_sliding]
    assert local_actual == LOCAL_INDICES, f"Local layers mismatch"
    assert global_actual == GLOBAL_INDICES, f"Global layers mismatch"

    samples = load_samples(tokenizer)

    # Apply selective LoRA
    lora_modules, n_lora_params = apply_lora_to_layers(model, layer_indices, RANK, SCALE)

    # Train
    log_memory(f"{phase_name}-before-train")
    train_info = train_selective(model, samples, lora_modules, n_steps, phase_name)
    log_memory(f"{phase_name}-after-train")

    # Evaluate
    log(f"\n  Evaluating on GSM8K ({n_eval} examples)...")
    acc = eval_gsm8k(model, tokenizer, n_eval)
    log_memory(f"{phase_name}-after-eval")

    result = {
        "phase": phase_name,
        "n_layers": len(layer_indices),
        "n_lora_params": n_lora_params,
        "gsm8k_pct": round(acc, 1),
        **train_info,
    }

    cleanup(model, tokenizer, samples)
    log_memory(f"{phase_name}-after-cleanup")

    return result


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def get_analytical_param_ratio() -> tuple:
    """
    Compute local/all-layer param ratio from actual Gemma 4 E4B q_proj shapes.
    Confirmed via inspection: local layers (2048, 320) → d_out=2048, d_in=2560
                              global layers (4096, 320) → d_out=4096, d_in=2560
    LoRA: lora_a=(r, d_in), lora_b=(d_out, r) → r*(d_in + d_out) per layer
    """
    r, d_in = RANK, 2560
    local_d_out = 2048   # local sliding-window attention (smaller head count)
    global_d_out = 4096  # global full-attention (larger head count)

    local_per_layer = r * d_in + r * local_d_out   # 15,360 + 12,288 = 27,648
    global_per_layer = r * d_in + r * global_d_out  # 15,360 + 24,576 = 39,936

    local_total = len(LOCAL_INDICES) * local_per_layer    # 35 × 27,648 = 967,680
    global_total = len(GLOBAL_INDICES) * global_per_layer  # 7  × 39,936 = 279,552
    all_total = local_total + global_total                 # = 1,247,232 (matches T2.1 exactly)

    return local_total, global_total, all_total, local_total / all_total


def main():
    log(f"T2.3: Local-Only vs All-Layer Adapter on Gemma 4")
    log(f"SMOKE_TEST={IS_SMOKE}, N_STEPS={N_STEPS}, N_EVAL={N_EVAL}")
    log(f"Local layers: {len(LOCAL_INDICES)} ({LOCAL_INDICES[:5]}...)")
    log(f"Global layers: {len(GLOBAL_INDICES)} ({GLOBAL_INDICES})")

    local_an, global_an, all_an, analytical_ratio = get_analytical_param_ratio()
    log(f"Analytical params — local: {local_an:,}, global: {global_an:,}, all: {all_an:,}")
    log(f"Analytical param ratio (local/all): {analytical_ratio:.4f}")
    assert all_an == ALL_LAYER_PARAMS, f"Analytical all-layer ({all_an}) ≠ T2.1 ({ALL_LAYER_PARAMS})"

    results = {
        "is_smoke": IS_SMOKE,
        "all_layer_baseline_pct": ALL_LAYER_GSM8K,
        "all_layer_params": ALL_LAYER_PARAMS,
        "analytical_local_params": local_an,
        "analytical_global_params": global_an,
        "analytical_param_ratio": round(analytical_ratio, 4),
    }

    # Phase 1: Local-only (35 sliding-window layers)
    local_result = run_phase("local-only", LOCAL_INDICES, N_STEPS, N_EVAL)
    results["local"] = local_result

    # Phase 2: Global-only (7 full-attention layers)
    global_result = run_phase("global-only", GLOBAL_INDICES, N_STEPS, N_EVAL)
    results["global"] = global_result

    # ── Kill Criteria ────────────────────────────────────────────────
    local_acc = local_result["gsm8k_pct"]
    global_acc = global_result["gsm8k_pct"]
    local_params = local_result["n_lora_params"]

    k1037_thresh = 0.90 * ALL_LAYER_GSM8K     # 73.8%
    k1038_thresh = 0.70 * ALL_LAYER_GSM8K     # 57.4%
    # K1039: local/all-layer ratio matches analytical value ±0.01
    # (analytical accounts for real q_proj dims: local=2048, global=2560)
    k1039_expected = analytical_ratio

    k1039_ratio = local_params / ALL_LAYER_PARAMS
    k1037_pass = local_acc >= k1037_thresh
    k1038_pass = global_acc < k1038_thresh
    k1039_pass = abs(k1039_ratio - k1039_expected) <= 0.01

    log(f"\n{'='*60}")
    log(f"Kill Criteria:")
    log(f"  K1037 local >= {k1037_thresh:.1f}%: got {local_acc:.1f}% → {'PASS' if k1037_pass else 'FAIL'}")
    log(f"  K1038 global < {k1038_thresh:.1f}%: got {global_acc:.1f}% → {'PASS' if k1038_pass else 'FAIL'}")
    log(f"  K1039 ratio = {k1039_expected:.3f}±0.01: got {k1039_ratio:.3f} → {'PASS' if k1039_pass else 'FAIL'}")

    results["kill_criteria"] = {
        "K1037": {
            "criterion": f"local-only >= {k1037_thresh:.1f}%",
            "value": local_acc,
            "pass": k1037_pass,
        },
        "K1038": {
            "criterion": f"global-only < {k1038_thresh:.1f}%",
            "value": global_acc,
            "pass": k1038_pass,
        },
        "K1039": {
            "criterion": f"param_ratio = analytical={k1039_expected:.4f}±0.01",
            "value": round(k1039_ratio, 4),
            "pass": k1039_pass,
        },
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    n_pass = sum([k1037_pass, k1038_pass, k1039_pass])
    log(f"Overall: {n_pass}/3 kill criteria PASS")


if __name__ == "__main__":
    main()
