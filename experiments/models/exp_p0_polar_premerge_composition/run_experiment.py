#!/usr/bin/env python3
"""
P0: PoLAR Pre-Merge Composition — Does sr=r Enable Safe Pre-Merge?

Tests whether PoLAR's spectral regularity (sr=r, Stiefel constraint) enables
pre-merge composition where standard LoRA catastrophically fails (Finding #510/526).

Kill criteria:
  K1451: PoLAR pre-merge GSM8K >= 50% (std LoRA pre-merge: 0-1%, base: 17%)
  K1452: PoLAR sr(DW_i) >= 5.0 for all 3 adapters post-training
  K1453: Inter-adapter cos(DW_i, DW_j) < 0.1 for all pairs
  K1454: PoLAR solo GSM8K >= 50% (adapter quality check)

Grounded by:
  Finding #442: PoLAR sr=r guarantee (verified to 7dp)
  Finding #526: Pre-merge failure is direction not magnitude
  Finding #510: Standard LoRA pre-merge catastrophic
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
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
E2E_DATA_DIR = EXPERIMENT_DIR.parent / "exp_p0_e2e_benchmark" / "data"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_LAYERS = 42
LORA_RANK = 6
LORA_SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
RETRACT_EVERY = 50
GRAD_CLIP = 1.0

N_TRAIN = 50 if IS_SMOKE else 1000
N_STEPS = 20 if IS_SMOKE else 500
N_EVAL = 5 if IS_SMOKE else 50
PROJ_NAMES = ["v_proj", "o_proj"]
DOMAINS = ["math", "code", "medical"]


def log(msg: str) -> None:
    print(msg, flush=True)


def log_memory(label: str = "") -> None:
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB",
          flush=True)


def cleanup(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ──────────────────────────────────────────────────────────────
# PoLAR Linear module (joint Stiefel on A and B)
# From exp_p1_c1_polar_gemma4_correct, adapted for v_proj + o_proj
# ──────────────────────────────────────────────────────────────

class PoLARLinear(nn.Module):
    """PoLAR: DW = A @ B with A^T A = I_r and B B^T = I_r.

    Forward: y = base(x) + scale * x @ A @ B
    Retraction: SVD polar projection on both A and B periodically.
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

        # B: (r, d_out) initialized to small random (not zeros — avoids SVD issues)
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
            dist_A = float(np.linalg.norm(
                A_retracted.astype(np.float64).T @ A_retracted.astype(np.float64) - I_r, 'fro'))
        else:
            dist_A = float('inf')

        # B: (r, d_out) -> B B^T = I_r
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        if np.all(np.isfinite(B_np)) and np.sum(B_np ** 2) > 1e-12:
            W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
            B_retracted = (W2 @ Vh2).astype(np.float32)
            self.lora_b = mx.array(B_retracted)
            dist_B = float(np.linalg.norm(
                B_retracted.astype(np.float64) @ B_retracted.astype(np.float64).T - I_r, 'fro'))
        else:
            dist_B = float(np.sqrt(self.rank))

        return dist_A, dist_B


# ──────────────────────────────────────────────────────────────
# Adapter injection, saving, loading
# ──────────────────────────────────────────────────────────────

def inject_polar_adapters(model, rank: int, scale: float, domain_seed: int) -> list:
    """Replace v_proj and o_proj in all 42 layers with PoLAR adapters."""
    modules = []
    for li in range(N_LAYERS):
        layer = model.layers[li]
        for proj_name in PROJ_NAMES:
            original = getattr(layer.self_attn, proj_name)
            seed = domain_seed * 100000 + li * 100 + (0 if proj_name == "v_proj" else 1)
            adapter = PoLARLinear(original, rank, scale, layer_seed=seed)
            setattr(layer.self_attn, proj_name, adapter)
            modules.append(adapter)
    return modules


def save_adapter_weights(modules: list, save_path: Path) -> None:
    """Save PoLAR A/B weights to safetensors."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    mod_idx = 0
    for li in range(N_LAYERS):
        for proj_name in PROJ_NAMES:
            mod = modules[mod_idx]
            mx.eval(mod.lora_a, mod.lora_b)
            weights[f"layers.{li}.self_attn.{proj_name}.lora_a"] = mod.lora_a
            weights[f"layers.{li}.self_attn.{proj_name}.lora_b"] = mod.lora_b
            mod_idx += 1
    mx.save_safetensors(str(save_path), weights)
    log(f"  Saved {len(weights)} weight tensors to {save_path}")


def remove_adapters(model) -> None:
    """Restore original base layers by extracting .base from PoLAR modules."""
    for li in range(N_LAYERS):
        layer = model.layers[li]
        for proj_name in PROJ_NAMES:
            mod = getattr(layer.self_attn, proj_name)
            if hasattr(mod, 'base'):
                setattr(layer.self_attn, proj_name, mod.base)


def inject_premerged(model, adapter_paths: list[Path], rank: int, scale: float) -> list:
    """Inject pre-merged PoLAR adapters (concatenated A/B along rank dim)."""
    adapter_weights = []
    for path in adapter_paths:
        w = mx.load(str(path))
        adapter_weights.append(w)
        log(f"  Loaded adapter: {path.parent.name} ({len(w)} keys)")

    modules = []
    for li in range(N_LAYERS):
        layer = model.layers[li]
        for proj_name in PROJ_NAMES:
            a_key = f"layers.{li}.self_attn.{proj_name}.lora_a"
            b_key = f"layers.{li}.self_attn.{proj_name}.lora_b"

            # Concatenate A along rank dim (axis=1), B along rank dim (axis=0)
            a_combined = mx.concatenate([w[a_key] for w in adapter_weights], axis=1)
            b_combined = mx.concatenate([w[b_key] for w in adapter_weights], axis=0)

            original = getattr(layer.self_attn, proj_name)
            adapter = PoLARLinear.__new__(PoLARLinear)
            nn.Module.__init__(adapter)
            adapter.base = original
            adapter.rank = rank * len(adapter_paths)
            adapter.scale = scale
            adapter.lora_a = a_combined
            adapter.lora_b = b_combined
            adapter.d_in = a_combined.shape[0]
            adapter.d_out = b_combined.shape[1]
            setattr(layer.self_attn, proj_name, adapter)
            modules.append(adapter)

    mx.eval(model.parameters())
    del adapter_weights
    gc.collect()
    mx.clear_cache()
    return modules


# ──────────────────────────────────────────────────────────────
# Data loading (reuse E2E benchmark data)
# ──────────────────────────────────────────────────────────────

def load_training_data(domain: str, tokenizer, n_train: int) -> list[list[int]]:
    """Load chat-formatted training data from E2E benchmark data dir."""
    data_file = E2E_DATA_DIR / domain / "train.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found: {data_file}")

    samples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= n_train:
                break
            item = json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=False
            )
            tokens = tokenizer.encode(text)
            if len(tokens) > MAX_SEQ_LEN:
                tokens = tokens[:MAX_SEQ_LEN]
            if len(tokens) > 10:
                samples.append(tokens)

    random.seed(SEED)
    random.shuffle(samples)
    log(f"  Loaded {len(samples)} {domain} samples")
    return samples


def get_batch(samples: list[list[int]], batch_size: int, step: int,
              pad_id: int) -> tuple[mx.array, mx.array]:
    """Padded batch with loss mask."""
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
# Training loop
# ──────────────────────────────────────────────────────────────

def train_polar_adapter(model, tokenizer, samples: list, modules: list,
                        n_steps: int, domain: str) -> dict:
    """Train PoLAR adapter with Stiefel retraction."""
    model.freeze()
    for mod in modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    log(f"  [{domain}] Trainable params: {n_params:,}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens, mask):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        loss_mask = mask[:, 1:]
        B, L, V = logits.shape
        ce = nn.losses.cross_entropy(
            logits.reshape(B * L, V),
            targets.reshape(B * L),
            reduction="none"
        ).reshape(B, L)
        return (ce * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    losses = []
    stiefel_log = []
    t0 = time.time()

    gc.disable()
    for step in range(n_steps):
        tokens, mask = get_batch(samples, BATCH_SIZE, step, pad_id)
        loss, grads = loss_and_grad(model, tokens, mask)
        optimizer.update(model, grads)

        # Stiefel retraction
        if (step + 1) % RETRACT_EVERY == 0:
            mx.eval(model.parameters())
            max_dA, max_dB = 0.0, 0.0
            for mod in modules:
                dA, dB = mod.retract_to_stiefel()
                max_dA = max(max_dA, dA)
                max_dB = max(max_dB, dB)
            mx.eval(model.parameters())
            stiefel_log.append({
                "step": step + 1,
                "max_dist_A": max_dA,
                "max_dist_B": max_dB,
            })

        mx.eval(loss, model.parameters())
        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % 100 == 0 or step == 0 or step == n_steps - 1:
            log(f"    step {step+1:4d}/{n_steps}: loss={loss_val:.4f} "
                f"({time.time()-t0:.0f}s)")

    gc.enable()
    gc.collect()

    # Final retraction
    mx.eval(model.parameters())
    max_dA, max_dB = 0.0, 0.0
    for mod in modules:
        dA, dB = mod.retract_to_stiefel()
        max_dA = max(max_dA, dA)
        max_dB = max(max_dB, dB)
    mx.eval(model.parameters())
    log(f"  Final retraction: max_A={max_dA:.2e}, max_B={max_dB:.2e}")

    elapsed = time.time() - t0
    log(f"  [{domain}] Done: {elapsed:.1f}s ({elapsed/max(n_steps,1):.2f}s/step)")

    return {
        "final_loss": losses[-1] if losses else float("nan"),
        "losses_sampled": [losses[i] for i in range(0, len(losses), max(1, len(losses)//10))],
        "stiefel_log": stiefel_log[-3:],
        "elapsed_s": round(elapsed, 1),
        "n_params": n_params,
    }


# ──────────────────────────────────────────────────────────────
# Measurement: stable rank and cross-adapter cosine
# ──────────────────────────────────────────────────────────────

def measure_adapter_metrics(adapter_paths: list[Path]) -> dict:
    """Measure sr(DW) for each adapter and cross-adapter cosine similarity."""
    adapter_weights = []
    for path in adapter_paths:
        adapter_weights.append(mx.load(str(path)))

    # Compute flattened DW = A @ B for each adapter across all layers
    dw_flats = []  # list of flattened weight perturbations per adapter
    sr_per_adapter = []

    for aw in adapter_weights:
        flat_parts = []
        srs = []
        for li in range(N_LAYERS):
            for proj_name in PROJ_NAMES:
                a_key = f"layers.{li}.self_attn.{proj_name}.lora_a"
                b_key = f"layers.{li}.self_attn.{proj_name}.lora_b"
                A_np = np.array(aw[a_key].tolist(), dtype=np.float64)
                B_np = np.array(aw[b_key].tolist(), dtype=np.float64)

                # Stable rank via SVD of M = diag(S_A) @ Vh_A @ B (avoid d_in x d_out matmul)
                _, S_A, Vh_A = np.linalg.svd(A_np, full_matrices=False)
                M = np.diag(S_A) @ (Vh_A @ B_np)
                if np.all(np.isfinite(M)):
                    _, S, _ = np.linalg.svd(M, full_matrices=False)
                    frob_sq = float(np.sum(S ** 2))
                    spec_sq = float(S[0] ** 2) if len(S) > 0 else 1e-12
                    srs.append(frob_sq / spec_sq if spec_sq > 1e-12 else 0.0)
                else:
                    srs.append(0.0)

                # For cosine: flatten the SVD-based representation (S values)
                flat_parts.append(M.flatten())

        dw_flats.append(np.concatenate(flat_parts))
        sr_per_adapter.append({
            "mean": float(np.mean(srs)),
            "min": float(np.min(srs)),
            "max": float(np.max(srs)),
        })

    # Cross-adapter cosine similarity
    cosines = {}
    for i in range(len(dw_flats)):
        for j in range(i + 1, len(dw_flats)):
            norm_i = np.linalg.norm(dw_flats[i])
            norm_j = np.linalg.norm(dw_flats[j])
            if norm_i > 1e-12 and norm_j > 1e-12:
                cos = float(np.dot(dw_flats[i], dw_flats[j]) / (norm_i * norm_j))
            else:
                cos = 0.0
            cosines[f"{DOMAINS[i]}_vs_{DOMAINS[j]}"] = cos

    del adapter_weights, dw_flats
    gc.collect()

    return {
        "stable_ranks": {DOMAINS[i]: sr_per_adapter[i] for i in range(len(DOMAINS))},
        "cross_cosines": cosines,
    }


# ──────────────────────────────────────────────────────────────
# GSM8K evaluation
# ──────────────────────────────────────────────────────────────

def load_gsm8k_test() -> list[dict]:
    """Load GSM8K test set via parquet (avoids dill/pickle Python 3.14 bug)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download("openai/gsm8k",
                           "main/test-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    return [{"question": row["question"], "answer": row["answer"]}
            for _, row in df.iterrows()]


def eval_gsm8k(model, tokenizer, n_eval: int, label: str = "") -> dict:
    """Evaluate on GSM8K with chat template."""
    from mlx_lm import generate as mlx_generate

    all_examples = load_gsm8k_test()
    rng = np.random.default_rng(SEED + 99)
    indices = rng.choice(len(all_examples), size=min(n_eval, len(all_examples)),
                         replace=False)
    ds = [all_examples[i] for i in indices]

    correct = 0
    total = 0

    for ex in ds:
        messages = [{"role": "user", "content":
                     f"Solve the following math problem step by step.\n\n{ex['question']}"}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        try:
            response = mlx_generate(
                model, tokenizer, prompt=formatted,
                max_tokens=512, verbose=False,
            )
        except Exception as e:
            log(f"    Generation error: {e}")
            response = ""

        gt_match = re.search(r"####\s*([\d,\-\.]+)", ex["answer"])
        if not gt_match:
            continue
        gt_ans = gt_match.group(1).replace(",", "").strip()
        total += 1

        pred_match = re.search(r"####\s*([\d,\-\.]+)", response)
        if pred_match:
            pred_ans = pred_match.group(1).replace(",", "").strip()
            if pred_ans == gt_ans:
                correct += 1
        else:
            nums = re.findall(r"\b\d+\.?\d*\b", response.replace(",", ""))
            if nums and nums[-1] == gt_ans:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0.0
    log(f"  GSM8K [{label}]: {correct}/{total} = {acc:.1f}%")
    return {"accuracy": acc, "correct": correct, "total": total}


# ──────────────────────────────────────────────────────────────
# Phase functions (MANDATORY: each loads/frees model independently)
# ──────────────────────────────────────────────────────────────

def phase_train_adapter(domain: str, domain_seed: int) -> dict:
    """Train one PoLAR adapter on a domain. Saves weights, frees model."""
    log(f"\n{'='*60}")
    log(f"Phase: Train PoLAR adapter — {domain} (seed={domain_seed})")
    log(f"{'='*60}")

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory(f"loaded-{domain}")

    samples = load_training_data(domain, tokenizer, N_TRAIN)
    modules = inject_polar_adapters(model, LORA_RANK, LORA_SCALE, domain_seed)

    train_result = train_polar_adapter(model, tokenizer, samples, modules,
                                       N_STEPS, domain)

    # Save adapter weights
    save_path = ADAPTERS_DIR / domain / "adapters.safetensors"
    save_adapter_weights(modules, save_path)

    log_memory(f"post-train-{domain}")
    cleanup(model, tokenizer, samples)
    return train_result


def phase_eval_solo(domain: str) -> dict:
    """Evaluate a solo PoLAR adapter on GSM8K."""
    log(f"\n{'='*60}")
    log(f"Phase: Evaluate solo {domain} adapter on GSM8K")
    log(f"{'='*60}")

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)

    # Inject single adapter from saved weights
    adapter_path = ADAPTERS_DIR / domain / "adapters.safetensors"
    weights = mx.load(str(adapter_path))

    for li in range(N_LAYERS):
        layer = model.layers[li]
        for proj_name in PROJ_NAMES:
            a_key = f"layers.{li}.self_attn.{proj_name}.lora_a"
            b_key = f"layers.{li}.self_attn.{proj_name}.lora_b"
            original = getattr(layer.self_attn, proj_name)
            adapter = PoLARLinear.__new__(PoLARLinear)
            nn.Module.__init__(adapter)
            adapter.base = original
            adapter.rank = LORA_RANK
            adapter.scale = LORA_SCALE
            adapter.lora_a = weights[a_key]
            adapter.lora_b = weights[b_key]
            adapter.d_in = weights[a_key].shape[0]
            adapter.d_out = weights[b_key].shape[1]
            setattr(layer.self_attn, proj_name, adapter)

    mx.eval(model.parameters())
    del weights
    gc.collect()

    result = eval_gsm8k(model, tokenizer, N_EVAL, label=f"solo-{domain}")
    log_memory(f"post-eval-solo-{domain}")
    cleanup(model, tokenizer)
    return result


def phase_eval_premerged() -> dict:
    """Evaluate pre-merged (all 3 PoLAR adapters) on GSM8K."""
    log(f"\n{'='*60}")
    log(f"Phase: Evaluate PRE-MERGED 3 PoLAR adapters on GSM8K")
    log(f"{'='*60}")

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)

    adapter_paths = [ADAPTERS_DIR / d / "adapters.safetensors" for d in DOMAINS]
    modules = inject_premerged(model, adapter_paths, LORA_RANK, LORA_SCALE)

    result = eval_gsm8k(model, tokenizer, N_EVAL, label="pre-merged")
    log_memory("post-eval-premerged")
    cleanup(model, tokenizer)
    return result


def phase_eval_base() -> dict:
    """Evaluate base model (no adapters) on GSM8K."""
    log(f"\n{'='*60}")
    log(f"Phase: Evaluate BASE model on GSM8K")
    log(f"{'='*60}")

    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    result = eval_gsm8k(model, tokenizer, N_EVAL, label="base")
    log_memory("post-eval-base")
    cleanup(model, tokenizer)
    return result


def phase_measure_metrics() -> dict:
    """Measure stable ranks and cross-adapter cosine similarity."""
    log(f"\n{'='*60}")
    log(f"Phase: Measure adapter metrics (sr, cosine)")
    log(f"{'='*60}")

    adapter_paths = [ADAPTERS_DIR / d / "adapters.safetensors" for d in DOMAINS]
    return measure_adapter_metrics(adapter_paths)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    log("=" * 70)
    log("P0: PoLAR Pre-Merge Composition — Does sr=r Enable Safe Pre-Merge?")
    log(f"  SMOKE={IS_SMOKE}, steps={N_STEPS}, train={N_TRAIN}, eval={N_EVAL}")
    log(f"  Rank={LORA_RANK}, Scale={LORA_SCALE}, Projections={PROJ_NAMES}")
    log("=" * 70)
    t_start = time.time()

    results = {"is_smoke": IS_SMOKE, "model": MODEL_ID}

    # Phase 1-3: Train 3 PoLAR adapters sequentially (skip if already saved)
    train_results = {}
    for i, domain in enumerate(DOMAINS):
        saved_path = ADAPTERS_DIR / domain / "adapters.safetensors"
        if saved_path.exists():
            log(f"\n  Skipping {domain} training — adapter already saved at {saved_path}")
            train_results[domain] = {"skipped": True, "path": str(saved_path)}
        else:
            train_results[domain] = phase_train_adapter(domain, domain_seed=i + 1)
    results["training"] = train_results

    # Phase 4: Evaluate base model (reference)
    results["base_gsm8k"] = phase_eval_base()

    # Phase 5: Evaluate solo math adapter (K1454)
    results["solo_gsm8k"] = phase_eval_solo("math")

    # Phase 6: Evaluate pre-merged (K1451)
    results["premerged_gsm8k"] = phase_eval_premerged()

    # Phase 7: Measure adapter metrics (K1452, K1453)
    results["metrics"] = phase_measure_metrics()

    # ── Summary and kill criteria ──
    base_acc = results["base_gsm8k"]["accuracy"]
    solo_acc = results["solo_gsm8k"]["accuracy"]
    merged_acc = results["premerged_gsm8k"]["accuracy"]

    # K1452: sr >= 5.0 for all adapters
    all_sr_min = min(
        results["metrics"]["stable_ranks"][d]["min"] for d in DOMAINS
    )
    k1452_pass = all_sr_min >= 5.0

    # K1453: cosine < 0.1 for all pairs
    max_cosine = max(abs(v) for v in results["metrics"]["cross_cosines"].values())
    k1453_pass = max_cosine < 0.1

    results["summary"] = {
        "base_gsm8k": base_acc,
        "solo_gsm8k": solo_acc,
        "premerged_gsm8k": merged_acc,
        "premerge_delta_from_solo": round(merged_acc - solo_acc, 1),
        "premerge_delta_from_base": round(merged_acc - base_acc, 1),
        "all_sr_min": all_sr_min,
        "max_cross_cosine": max_cosine,
        "k1451_pass": merged_acc >= 50.0,
        "k1452_pass": k1452_pass,
        "k1453_pass": k1453_pass,
        "k1454_pass": solo_acc >= 50.0,
        "total_time_s": round(time.time() - t_start, 1),
        # Reference: std LoRA pre-merge from Finding #510/#526
        "ref_std_lora_premerge_gsm8k": 0.0,
    }

    log("\n" + "=" * 70)
    log("RESULTS")
    log("=" * 70)
    log(f"  Base model:          GSM8K = {base_acc:.1f}%")
    log(f"  PoLAR solo (math):   GSM8K = {solo_acc:.1f}%")
    log(f"  PoLAR pre-merged:    GSM8K = {merged_acc:.1f}%")
    log(f"  Std LoRA pre-merged: GSM8K = 0% (Finding #510/#526)")
    log(f"")
    log(f"  Pre-merge delta from solo: {merged_acc - solo_acc:+.1f}pp")
    log(f"  Pre-merge delta from base: {merged_acc - base_acc:+.1f}pp")
    log(f"")
    log(f"  Min stable rank (all adapters): {all_sr_min:.4f}")
    log(f"  Max cross-adapter cosine:       {max_cosine:.6f}")
    log(f"")
    log(f"  K1451 (pre-merge >= 50%):    {'PASS' if merged_acc >= 50.0 else 'FAIL'} — {merged_acc:.1f}%")
    log(f"  K1452 (sr >= 5.0):           {'PASS' if k1452_pass else 'FAIL'} — {all_sr_min:.4f}")
    log(f"  K1453 (cos < 0.1):           {'PASS' if k1453_pass else 'FAIL'} — {max_cosine:.6f}")
    log(f"  K1454 (solo >= 50%):         {'PASS' if solo_acc >= 50.0 else 'FAIL'} — {solo_acc:.1f}%")
    log(f"")
    log(f"  Cross-adapter cosines:")
    for pair, cos in results["metrics"]["cross_cosines"].items():
        log(f"    {pair}: {cos:.6f}")
    log(f"")
    log(f"  Stable ranks:")
    for d in DOMAINS:
        sr = results["metrics"]["stable_ranks"][d]
        log(f"    {d}: mean={sr['mean']:.4f} min={sr['min']:.4f} max={sr['max']:.4f}")
    log(f"")
    log(f"  Total time: {results['summary']['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
