#!/usr/bin/env python3
"""
C1.1v2: PoLAR vs LoRA Head-to-Head on Gemma 4 E4B (GSM8K)

Fixes 5 critical bugs from v1:
  1. Uses chat template for training AND eval (Gemma 4 is a chat model)
  2. Uses real GSM8K data (2000 examples, not 20 flashcards)
  3. Handles both local (d_out=2048) and global (d_out=4096) layer dimensions
  4. Uses int32 for token targets
  5. Proper padding with loss masking

Kill criteria:
  KC07: sr(PoLAR ΔW) >= 5 at r=6 with real GSM8K training
  KC08: PoLAR GSM8K accuracy >= LoRA at matched rank r=6, 1000 steps
  KC09: ||UU^T-I||_F < 0.01 AND ||VV^T-I||_F < 0.01 post-retraction
"""

import gc
import json
import math
import os
import random
import re
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# Memory safety — MANDATORY per CODING_GUIDELINES
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results_v2.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_LAYERS = 42
LORA_RANK = 6
LORA_SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
RETRACT_EVERY = 20
GRAD_CLIP = 1.0

N_TRAIN = 100 if IS_SMOKE else 2000
N_STEPS = 20 if IS_SMOKE else 1000
N_EVAL = 5 if IS_SMOKE else 50


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB", flush=True)


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ──────────────────────────────────────────────────────────────
# Data: Real GSM8K with proper chat template
# ──────────────────────────────────────────────────────────────

def prepare_gsm8k_training(tokenizer, n_train: int) -> list[list[int]]:
    """Load real GSM8K and tokenize with chat template. Returns list of token seqs."""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=SEED).select(range(min(n_train, len(ds))))

    samples = []
    for ex in ds:
        messages = [
            {"role": "user", "content": f"Solve the following math problem step by step.\n\n{ex['question']}"},
            {"role": "assistant", "content": ex["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        tokens = tokenizer.encode(text)
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]
        if len(tokens) > 10:  # Skip degenerate short samples
            samples.append(tokens)

    random.seed(SEED)
    random.shuffle(samples)
    log(f"  Prepared {len(samples)} GSM8K training samples (chat-templated)")
    return samples


def get_batch(samples: list[list[int]], batch_size: int, step: int, pad_id: int) -> tuple[mx.array, mx.array]:
    """Get padded batch + loss mask. Returns (tokens, mask) both as int32/float32."""
    rng = np.random.default_rng(SEED + step)
    indices = rng.choice(len(samples), size=batch_size, replace=True)
    batch_tokens = [samples[i] for i in indices]
    max_len = max(len(t) for t in batch_tokens)

    padded = []
    masks = []
    for tokens in batch_tokens:
        pad_len = max_len - len(tokens)
        padded.append(tokens + [pad_id] * pad_len)
        masks.append([1.0] * len(tokens) + [0.0] * pad_len)

    return mx.array(padded, dtype=mx.int32), mx.array(masks, dtype=mx.float32)


# ──────────────────────────────────────────────────────────────
# PoLAR Linear module (joint Stiefel on A and B)
# ──────────────────────────────────────────────────────────────

class PoLARLinear(nn.Module):
    """PoLAR: ΔW = A @ B with A^T A = I_r (Stiefel) and B B^T = I_r (Stiefel).

    Forward: y = base(x) + scale * x @ A @ B
    Retraction: SVD polar projection on both A and B every RETRACT_EVERY steps.
    """

    def __init__(self, base_linear: nn.Module, rank: int, scale: float, layer_seed: int = 0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        # Extract dims from potentially quantized base layer
        if hasattr(base_linear, 'group_size'):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]

        self.d_in = d_in
        self.d_out = d_out

        # A: (d_in, r) initialized with orthonormal columns via QR
        rng = np.random.default_rng(SEED + layer_seed)
        rand_a = rng.standard_normal((d_in, rank)).astype(np.float32)
        Q, _ = np.linalg.qr(rand_a)
        self.lora_a = mx.array(Q[:, :rank])

        # B: (r, d_out) initialized to small random (not zeros — avoids SVD issues at retraction)
        rand_b = rng.standard_normal((rank, d_out)).astype(np.float32) * 0.01
        self.lora_b = mx.array(rand_b)

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scale * lora_out

    def retract_to_stiefel(self) -> tuple[float, float]:
        """Polar-project A and B back onto Stiefel manifolds."""
        I_r = np.eye(self.rank, dtype=np.float64)

        # A: (d_in, r) -> A^T A = I_r
        A_np = np.array(self.lora_a.tolist(), dtype=np.float64)
        if np.all(np.isfinite(A_np)) and np.sum(A_np ** 2) > 1e-12:
            W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
            A_retracted = (W @ Vh).astype(np.float32)
            self.lora_a = mx.array(A_retracted)
            dist_A = float(np.linalg.norm(A_retracted.astype(np.float64).T @ A_retracted.astype(np.float64) - I_r, 'fro'))
        else:
            dist_A = float('inf')

        # B: (r, d_out) -> B B^T = I_r
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        if np.all(np.isfinite(B_np)) and np.sum(B_np ** 2) > 1e-12:
            W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
            B_retracted = (W2 @ Vh2).astype(np.float32)
            self.lora_b = mx.array(B_retracted)
            dist_B = float(np.linalg.norm(B_retracted.astype(np.float64) @ B_retracted.astype(np.float64).T - I_r, 'fro'))
        else:
            dist_B = float(np.sqrt(self.rank))

        return dist_A, dist_B


class LoRALinear(nn.Module):
    """Standard LoRA baseline."""

    def __init__(self, base_linear: nn.Module, rank: int, scale: float, layer_seed: int = 0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scale = scale

        if hasattr(base_linear, 'group_size'):
            d_out = base_linear.weight.shape[0]
            d_in = base_linear.scales.shape[1] * base_linear.group_size
        else:
            d_in = base_linear.weight.shape[1]
            d_out = base_linear.weight.shape[0]

        self.d_in = d_in
        self.d_out = d_out

        # Standard LoRA init: A ~ N(0, 1/sqrt(d_in)), B = 0
        rng = np.random.default_rng(SEED + layer_seed)
        A_init = rng.standard_normal((d_in, rank)).astype(np.float32) * (1.0 / math.sqrt(d_in))
        self.lora_a = mx.array(A_init)
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scale * lora_out


# ──────────────────────────────────────────────────────────────
# Adapter injection and measurement
# ──────────────────────────────────────────────────────────────

def inject_adapters(model, adapter_cls, rank: int, scale: float) -> list:
    """Replace q_proj in all 42 layers with adapter. Each layer gets unique seed."""
    modules = []
    for li in range(N_LAYERS):
        layer = model.layers[li]
        original_q = layer.self_attn.q_proj
        adapter = adapter_cls(original_q, rank, scale, layer_seed=li * 1000)
        layer.self_attn.q_proj = adapter
        modules.append(adapter)
    return modules


def measure_stable_ranks(modules: list) -> dict:
    """Measure sr(ΔW = A @ B) for all layers via SVD."""
    srs = []
    for mod in modules:
        A_np = np.array(mod.lora_a.tolist(), dtype=np.float64)
        B_np = np.array(mod.lora_b.tolist(), dtype=np.float64)
        if np.sum(A_np ** 2) < 1e-12 or np.sum(B_np ** 2) < 1e-12:
            srs.append(0.0)
            continue
        if not (np.all(np.isfinite(A_np)) and np.all(np.isfinite(B_np))):
            srs.append(0.0)
            continue
        # Use SVD of A and B separately to avoid overflow in large matmul
        _, S_A, Vh_A = np.linalg.svd(A_np, full_matrices=False)  # A = U_A @ diag(S_A) @ Vh_A
        M = np.diag(S_A) @ (Vh_A @ B_np)  # (r, d_out) — bounded, no overflow
        if not np.all(np.isfinite(M)):
            srs.append(0.0)
            continue
        _, S, _ = np.linalg.svd(M, full_matrices=False)
        frob_sq = float(np.sum(S ** 2))
        spec_sq = float(S[0] ** 2) if len(S) > 0 else 1e-12
        srs.append(frob_sq / spec_sq if spec_sq > 1e-12 else 0.0)
    return {
        "mean": float(np.mean(srs)),
        "min": float(np.min(srs)),
        "max": float(np.max(srs)),
        "first3": srs[:3],
    }


# ──────────────────────────────────────────────────────────────
# Training loop (with all fixes)
# ──────────────────────────────────────────────────────────────

def train(model, tokenizer, samples: list, modules: list, n_steps: int,
          do_retract: bool = False, phase_name: str = "") -> dict:
    """Train adapter with proper loss masking and optional Stiefel retraction."""
    # Freeze base, unfreeze LoRA params
    model.freeze()
    for mod in modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    log(f"  [{phase_name}] Trainable params: {n_params:,}")

    # Get pad token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens, mask):
        # tokens: (B, L) int32, mask: (B, L) float32 (1=real, 0=pad)
        logits = model(tokens[:, :-1])  # (B, L-1, V)
        targets = tokens[:, 1:]         # (B, L-1)
        loss_mask = mask[:, 1:]         # (B, L-1) — align with targets

        # Per-token cross-entropy
        B, L, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            reduction="none"
        ).reshape(B, L)

        # Masked mean: only count non-pad positions
        masked_loss = (ce * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        return masked_loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    stiefel_log = []
    t0 = time.time()

    for step in range(n_steps):
        tokens, mask = get_batch(samples, BATCH_SIZE, step, pad_id)
        loss, grads = loss_and_grad(model, tokens, mask)
        optimizer.update(model, grads)

        # Stiefel retraction for PoLAR
        if do_retract and (step + 1) % RETRACT_EVERY == 0:
            mx.eval(model.parameters())
            dists_A, dists_B = [], []
            for mod in modules:
                dA, dB = mod.retract_to_stiefel()
                dists_A.append(dA)
                dists_B.append(dB)
            mx.eval(model.parameters())
            stiefel_log.append({
                "step": step + 1,
                "max_dist_A": max(dists_A),
                "max_dist_B": max(dists_B),
            })

        mx.eval(loss, model.parameters())
        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % 100 == 0 or step == 0 or step == n_steps - 1:
            log(f"    step {step+1:4d}/{n_steps}: loss={loss_val:.4f} "
                f"({time.time()-t0:.0f}s)")

    # Final retraction
    if do_retract:
        mx.eval(model.parameters())
        dists_A, dists_B = [], []
        for mod in modules:
            dA, dB = mod.retract_to_stiefel()
            dists_A.append(dA)
            dists_B.append(dB)
        mx.eval(model.parameters())
        log(f"  Final retraction: max_A={max(dists_A):.2e}, max_B={max(dists_B):.2e}")
        stiefel_log.append({"step": n_steps, "max_dist_A": max(dists_A), "max_dist_B": max(dists_B)})

    elapsed = time.time() - t0
    log(f"  [{phase_name}] Done: {elapsed:.1f}s ({elapsed/max(n_steps,1):.2f}s/step)")

    return {
        "final_loss": losses[-1] if losses else float("nan"),
        "losses_sampled": [losses[i] for i in range(0, len(losses), max(1, len(losses)//20))],
        "stiefel_log": stiefel_log,
        "elapsed_s": round(elapsed, 1),
        "n_params": n_params,
    }


# ──────────────────────────────────────────────────────────────
# GSM8K evaluation (with proper chat template)
# ──────────────────────────────────────────────────────────────

def eval_gsm8k(model, tokenizer, n_eval: int) -> dict:
    """Evaluate GSM8K with proper chat template. Returns accuracy and details."""
    from datasets import load_dataset
    from mlx_lm import generate as mlx_generate

    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED + 99).select(range(min(n_eval, len(ds))))

    correct = 0
    total = 0

    for ex in ds:
        question = ex["question"]
        prompt = f"Solve the following math problem step by step.\n\n{question}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            response = mlx_generate(
                model, tokenizer, prompt=formatted,
                max_tokens=256, verbose=False,
            )
        except Exception as e:
            log(f"    Generation error: {e}")
            response = ""

        # Extract ground truth
        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()
        total += 1

        # Extract prediction
        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            # Fallback: last number in response
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0.0
    log(f"  GSM8K: {correct}/{total} = {acc:.1f}%")
    return {"accuracy": acc, "correct": correct, "total": total}


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("C1.1v2: PoLAR vs LoRA Head-to-Head on Gemma 4 E4B (GSM8K)")
    log(f"  SMOKE={IS_SMOKE}, steps={N_STEPS}, train={N_TRAIN}, eval={N_EVAL}")
    log("=" * 70)
    t_start = time.time()

    results = {"is_smoke": IS_SMOKE, "model": MODEL_ID, "version": "v2"}

    # ── Phase 1: Train PoLAR r=6 on real GSM8K ──
    log("\n[Phase 1] PoLAR r=6 on GSM8K (1000 steps)")
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("loaded")

    samples = prepare_gsm8k_training(tokenizer, N_TRAIN)
    polar_modules = inject_adapters(model, PoLARLinear, LORA_RANK, LORA_SCALE)
    polar_train = train(model, tokenizer, samples, polar_modules,
                        n_steps=N_STEPS, do_retract=True, phase_name="PoLAR")

    # Measure stable rank
    polar_sr = measure_stable_ranks(polar_modules)
    log(f"  PoLAR stable rank: mean={polar_sr['mean']:.4f}, min={polar_sr['min']:.4f}")
    results["polar_sr"] = polar_sr

    # Evaluate on GSM8K
    log("\n  Evaluating PoLAR on GSM8K...")
    polar_eval = eval_gsm8k(model, tokenizer, N_EVAL)
    results["polar_gsm8k"] = polar_eval
    results["polar_train"] = polar_train

    cleanup(model, tokenizer)

    # ── Phase 2: Train LoRA r=6 on same GSM8K ──
    log("\n[Phase 2] LoRA r=6 on GSM8K (1000 steps)")
    model2, tokenizer2 = mlx_load(MODEL_ID)
    log_memory("loaded-2")

    lora_modules = inject_adapters(model2, LoRALinear, LORA_RANK, LORA_SCALE)
    lora_train = train(model2, tokenizer2, samples, lora_modules,
                       n_steps=N_STEPS, do_retract=False, phase_name="LoRA")

    # Measure stable rank
    lora_sr = measure_stable_ranks(lora_modules)
    log(f"  LoRA stable rank: mean={lora_sr['mean']:.4f}, min={lora_sr['min']:.4f}")
    results["lora_sr"] = lora_sr

    # Evaluate on GSM8K
    log("\n  Evaluating LoRA on GSM8K...")
    lora_eval = eval_gsm8k(model2, tokenizer2, N_EVAL)
    results["lora_gsm8k"] = lora_eval
    results["lora_train"] = lora_train

    cleanup(model2, tokenizer2)

    # ── Summary ──
    polar_acc = polar_eval["accuracy"]
    lora_acc = lora_eval["accuracy"]

    kc07_pass = polar_sr["mean"] >= 5.0
    kc08_pass = polar_acc >= lora_acc
    kc09_pass = True
    if polar_train["stiefel_log"]:
        last = polar_train["stiefel_log"][-1]
        kc09_pass = last["max_dist_A"] < 0.01 and last["max_dist_B"] < 0.01

    results["summary"] = {
        "polar_acc": polar_acc,
        "lora_acc": lora_acc,
        "polar_sr_mean": polar_sr["mean"],
        "lora_sr_mean": lora_sr["mean"],
        "kc07_pass": kc07_pass,
        "kc08_pass": kc08_pass,
        "kc09_pass": kc09_pass,
        "all_pass": kc07_pass and kc08_pass and kc09_pass,
        "total_time_s": round(time.time() - t_start, 1),
    }

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"  PoLAR r=6:  GSM8K={polar_acc:.1f}%  sr={polar_sr['mean']:.2f}")
    log(f"  LoRA  r=6:  GSM8K={lora_acc:.1f}%  sr={lora_sr['mean']:.2f}")
    log(f"  T2.1 ref:   GSM8K=82.0%  (mlx_lm.lora, 2000 examples)")
    log(f"")
    log(f"  KC07 (sr >= 5):         {'PASS' if kc07_pass else 'FAIL'} — {polar_sr['mean']:.4f}")
    log(f"  KC08 (PoLAR >= LoRA):   {'PASS' if kc08_pass else 'FAIL'} — {polar_acc:.1f}% vs {lora_acc:.1f}%")
    log(f"  KC09 (Stiefel < 0.01):  {'PASS' if kc09_pass else 'FAIL'}")
    log(f"  ALL PASS: {results['summary']['all_pass']}")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
