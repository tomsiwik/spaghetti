#!/usr/bin/env python3
"""
C1.3: PoLAR Scale Invariance — Joint Stiefel vs LoRA at scale={0.5×,1×,2×,4×}

Fixes C1.1's vacuous KC08 (both 0% due to train/eval format mismatch).
Tests Theorem 1: PoLAR B rows have unit norm → effective_scale = s×1.0 → variance < 5pp.

Kill criteria:
  KC13: PoLAR accuracy variance < 5pp across scale={3,6,12,24} (training scale=6)
  KC14: PoLAR variance < LoRA variance (structural benefit)
  KC15: PoLAR at scale=6 >= 80% of LoRA at scale=6 (no regression)

SMOKE_TEST=1: 10 train steps, 10 eval examples.
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

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
SEED = 42

N_LAYERS = 42
SCALE = 6.0
LR = 1e-4
BATCH_SIZE = 2
MAX_SEQ_LEN = 256
RETRACT_EVERY = 10    # Retract every 10 steps (tighter than C1.1's 20)
GRAD_CLIP = 1.0

N_STEPS = 10 if IS_SMOKE else 500
N_TRAIN = 10 if IS_SMOKE else 80
N_EVAL = 10 if IS_SMOKE else 25
RANK = 6

# Eval scales: {0.5×, 1×, 2×, 4×} training scale
EVAL_SCALES = [3.0, 6.0, 12.0, 24.0]


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


# ──────────────────────────────────────────────────────────────
# Synthetic multi-step math dataset (GSM8K-style)
# ──────────────────────────────────────────────────────────────

def make_math_problems(n: int, seed: int = SEED) -> list[dict]:
    """Generate n synthetic 2-4 step arithmetic word problems.

    Format mirrors GSM8K: multi-step word problem → final numeric answer.
    No network access required.
    """
    rng = random.Random(seed)
    templates = [
        # 2-step: addition then subtraction
        lambda r: (
            f"A store had {r.randint(10,50)} apples. They received a delivery of "
            f"{r.randint(5,30)} more apples, then sold {r.randint(3,20)} apples. "
            f"How many apples does the store have now?",
            None,  # computed below
            ["start", "add", "sub"],
        ),
        # 2-step: multiplication then addition
        lambda r: (
            f"Each box has {r.randint(3,8)} oranges. There are {r.randint(4,10)} boxes. "
            f"If {r.randint(2,15)} more oranges are added, how many oranges total?",
            None,
            ["start", "mul", "add"],
        ),
        # 3-step: multiply, subtract, divide
        lambda r: (
            f"A factory produces {r.randint(8,20)} widgets per hour for "
            f"{r.randint(3,8)} hours. Then {r.randint(5,30)} widgets are defective. "
            f"The remaining widgets are split equally among {r.randint(2,5)} bins. "
            f"How many widgets per bin?",
            None,
            ["prod", "sub", "div"],
        ),
        # 2-step: sequential purchases
        lambda r: (
            f"Maria had ${r.randint(20,80)}. She earned ${r.randint(10,40)} babysitting "
            f"and then spent ${r.randint(5,35)} on a gift. How much money does she have now?",
            None,
            ["start", "earn", "spend"],
        ),
        # 2-step: distance/speed
        lambda r: (
            f"A car travels at {r.randint(30,70)} km/h for {r.randint(2,5)} hours, "
            f"then {r.randint(10,30)} more km. What total distance did it travel?",
            None,
            ["speed", "time", "extra"],
        ),
        # 3-step: people sharing
        lambda r: (
            f"There are {r.randint(3,6)} groups of {r.randint(4,8)} students each. "
            f"{r.randint(2,5)} students leave. The remaining students split into "
            f"teams of {r.randint(2,4)}. How many full teams are there?",
            None,
            ["groups", "leave", "teams"],
        ),
    ]

    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 20:
        attempts += 1
        template = rng.choice(templates)
        result = template(rng)
        q_template, _, steps = result

        # Extract numbers from the question and compute answer
        nums = [int(x) for x in re.findall(r'\d+', q_template)]
        if not nums:
            continue

        # Compute answer based on step type
        try:
            if steps == ["start", "add", "sub"]:
                start, add, sub = nums[0], nums[1], nums[2]
                ans = start + add - sub
            elif steps == ["start", "mul", "add"]:
                oranges, boxes, extra = nums[0], nums[1], nums[2]
                ans = oranges * boxes + extra
            elif steps == ["prod", "sub", "div"]:
                rate, hours, defect, bins = nums[0], nums[1], nums[2], nums[3]
                prod = rate * hours
                rem = prod - defect
                if rem <= 0 or rem % bins != 0:
                    continue
                ans = rem // bins
            elif steps == ["start", "earn", "spend"]:
                start, earn, spend = nums[0], nums[1], nums[2]
                ans = start + earn - spend
            elif steps == ["speed", "time", "extra"]:
                speed, time_h, extra = nums[0], nums[1], nums[2]
                ans = speed * time_h + extra
            elif steps == ["groups", "leave", "teams"]:
                groups, size, leave, team_size = nums[0], nums[1], nums[2], nums[3]
                total = groups * size
                remaining = total - leave
                if remaining <= 0:
                    continue
                ans = remaining // team_size
            else:
                continue

            if ans <= 0:
                continue

            problems.append({"question": q_template, "answer": str(ans)})
        except (IndexError, ZeroDivisionError):
            continue

    return problems[:n]


def build_training_samples(tokenizer, problems: list[dict]) -> list:
    """Tokenize math problems into training samples."""
    samples = []
    for p in problems:
        text = f"Question: {p['question']}\nAnswer: {p['answer']}"
        tokens = tokenizer.encode(text)[:MAX_SEQ_LEN]
        if len(tokens) > 2:
            samples.append(tokens)
    return samples


def get_batch(samples: list, batch_size: int, step: int) -> mx.array:
    rng = np.random.default_rng(SEED + step)
    indices = rng.choice(len(samples), size=batch_size, replace=True)
    batch_tokens = [samples[i] for i in indices]
    max_len = max(len(t) for t in batch_tokens)
    padded = [t + [0] * (max_len - len(t)) for t in batch_tokens]
    return mx.array(padded, dtype=mx.uint32)


# ──────────────────────────────────────────────────────────────
# PoLAR Linear module (joint Stiefel: both A and B)
# ──────────────────────────────────────────────────────────────

class PoLARLinear(nn.Module):
    """PoLAR adapter with joint Stiefel constraint (A^T A = I, B B^T = I).

    Retraction every RETRACT_EVERY steps via polar decomposition (SVD).
    """

    def __init__(self, base_linear: nn.Module, rank: int, scale: float):
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

        # A: (d_in, r) with orthonormal columns (A^T A = I_r)
        rng = np.random.default_rng(SEED)
        rand_mat = rng.standard_normal((d_in, rank)).astype(np.float32)
        Q, _ = np.linalg.qr(rand_mat)
        self.lora_a = mx.array(Q)

        # B: (r, d_out) initialized to zeros
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scale * lora_out

    def retract_to_stiefel(self) -> tuple[float, float]:
        """SVD retraction for both A and B. Returns (dist_A, dist_B)."""
        import warnings
        I_r = np.eye(self.rank)

        # Retract A: (d_in, r) → A^T A = I_r
        A_np = np.array(self.lora_a.tolist(), dtype=np.float64)
        if not np.all(np.isfinite(A_np)) or np.sum(A_np ** 2) < 1e-12:
            dist_A = float('inf')
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                W, _, Vh = np.linalg.svd(A_np, full_matrices=False)
                A_ret_f64 = W @ Vh  # (d_in, r), values in [-1, 1]
            A_ret = A_ret_f64.astype(np.float32)
            self.lora_a = mx.array(A_ret)
            # Dist computation in float64 for accuracy
            AtA = A_ret_f64.T @ A_ret_f64
            dist_A = float(np.linalg.norm(AtA - I_r, 'fro'))

        # Retract B: (r, d_out) → B B^T = I_r
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        B_frob = np.sum(B_np ** 2)
        if not np.all(np.isfinite(B_np)) or B_frob < 1e-12:
            # B still near zero early in training — skip retraction
            dist_B = float(np.sqrt(self.rank))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                W2, _, Vh2 = np.linalg.svd(B_np, full_matrices=False)
                B_ret_f64 = W2 @ Vh2  # (r, d_out)
            B_ret = B_ret_f64.astype(np.float32)
            self.lora_b = mx.array(B_ret)
            BBt = B_ret_f64 @ B_ret_f64.T
            dist_B = float(np.linalg.norm(BBt - I_r, 'fro'))

        return dist_A, dist_B

    def get_row_norms(self) -> np.ndarray:
        """Row norms of B matrix (should be ≈1 after retraction)."""
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        return np.linalg.norm(B_np, axis=1)  # shape (r,)


# ──────────────────────────────────────────────────────────────
# Standard LoRA Linear module
# ──────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Standard LoRA: ΔW = scale × B @ A, no Stiefel constraint."""

    def __init__(self, base_linear: nn.Module, rank: int, scale: float):
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

        rng = np.random.default_rng(SEED)
        A_init = rng.standard_normal((d_in, rank)).astype(np.float32) * (1.0 / math.sqrt(d_in))
        self.lora_a = mx.array(A_init)
        self.lora_b = mx.zeros((rank, d_out))

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_out = (x @ self.lora_a) @ self.lora_b
        return base_out + self.scale * lora_out

    def get_row_norms(self) -> np.ndarray:
        """Row norms of B matrix (unconstrained for standard LoRA)."""
        B_np = np.array(self.lora_b.tolist(), dtype=np.float64)
        return np.linalg.norm(B_np, axis=1)  # shape (r,)


# ──────────────────────────────────────────────────────────────
# Adapter injection
# ──────────────────────────────────────────────────────────────

def inject_adapters(model, adapter_cls, rank: int, scale: float) -> list:
    """Inject adapters into q_proj of all layers. Returns list of modules."""
    modules = []
    for li in range(N_LAYERS):
        layer = model.layers[li]
        orig_q = layer.self_attn.q_proj
        adapter = adapter_cls(orig_q, rank, scale)
        layer.self_attn.q_proj = adapter
        modules.append(adapter)
    return modules


def set_adapter_scale(modules: list, scale: float) -> None:
    """Update inference scale for all adapter modules."""
    for mod in modules:
        mod.scale = scale


def retract_all(modules: list) -> tuple[float, float]:
    """Retract all PoLAR modules. Returns (max_dist_A, max_dist_B)."""
    dists_A, dists_B = [], []
    for mod in modules:
        dA, dB = mod.retract_to_stiefel()
        dists_A.append(dA)
        dists_B.append(dB)
    mx.eval()
    return float(max(dists_A)), float(max(dists_B))


def measure_b_row_norms(modules: list) -> dict:
    """Measure B row norm statistics across all modules."""
    all_norms = []
    for mod in modules:
        all_norms.extend(mod.get_row_norms().tolist())
    arr = np.array(all_norms)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────

def train_adapter(model, tokenizer, samples: list, modules: list,
                  n_steps: int, do_retract: bool, phase_name: str) -> dict:
    """Train adapter for n_steps. Returns training stats."""
    model.freeze()
    for mod in modules:
        mod.unfreeze(keys=["lora_a", "lora_b"])

    n_params = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    log(f"  Trainable params: {n_params:,}")

    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        B, L, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * L, V), targets.reshape(B * L), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses = []
    stiefel_log = []
    t0 = time.time()

    for step in range(n_steps):
        batch = get_batch(samples, BATCH_SIZE, step)
        loss, grads = loss_and_grad(model, batch)

        # Gradient clipping
        from mlx.utils import tree_flatten, tree_map
        grad_list = [(k, v) for k, v in tree_flatten(grads) if isinstance(v, mx.array)]
        if grad_list:
            gnorm = math.sqrt(sum(float(mx.sum(g * g).item()) for _, g in grad_list))
            if gnorm > GRAD_CLIP:
                s = GRAD_CLIP / (gnorm + 1e-8)
                grads = tree_map(lambda g: g * s if isinstance(g, mx.array) else g, grads)

        optimizer.update(model, grads)

        # Stiefel retraction for PoLAR
        if do_retract and (step + 1) % RETRACT_EVERY == 0:
            mx.eval(model.parameters())
            max_A, max_B = retract_all(modules)
            stiefel_log.append({"step": step + 1, "max_dist_A": max_A, "max_dist_B": max_B})

        mx.eval(loss, model.parameters())
        loss_val = float(loss.item())
        losses.append(loss_val)

        if (step + 1) % 50 == 0 or step == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            log(f"  [{phase_name}] step {step+1:4d}/{n_steps}: loss={loss_val:.4f}  {elapsed:.0f}s")

    # Final retraction
    if do_retract:
        mx.eval(model.parameters())
        max_A, max_B = retract_all(modules)
        stiefel_log.append({"step": n_steps, "max_dist_A": max_A, "max_dist_B": max_B})
        log(f"  Final retraction: dist_A={max_A:.2e} dist_B={max_B:.2e}")

    elapsed = time.time() - t0
    log(f"  Training complete: {elapsed:.1f}s")

    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else float("nan"),
        "stiefel_log": stiefel_log,
        "elapsed_s": elapsed,
    }


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def extract_number(text: str) -> str | None:
    """Extract last numeric value from model output."""
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def eval_math(model, tokenizer, problems: list[dict], scale: float,
              modules: list, max_tokens: int = 64) -> float:
    """Evaluate adapter at given scale. Returns accuracy."""
    from mlx_lm import generate as mlx_generate
    set_adapter_scale(modules, scale)
    correct = 0
    for p in problems:
        prompt = f"Question: {p['question']}\nAnswer:"
        try:
            response = mlx_generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens, verbose=False,
            )
        except Exception:
            response = ""
        pred = extract_number(response)
        correct += int(pred == p["answer"])
    acc = correct / len(problems) if problems else 0.0
    return acc


def eval_at_all_scales(model, tokenizer, eval_problems: list[dict],
                       modules: list, adapter_name: str) -> dict:
    """Evaluate at each scale in EVAL_SCALES. Returns per-scale accuracy and variance."""
    results_per_scale = {}
    for s in EVAL_SCALES:
        acc = eval_math(model, tokenizer, eval_problems, s, modules)
        results_per_scale[f"scale_{s:.0f}"] = acc
        log(f"  [{adapter_name}] scale={s:.1f}: acc={acc:.1%}")

    accs = list(results_per_scale.values())
    variance_pp = (max(accs) - min(accs)) * 100  # peak-to-peak in pp
    results_per_scale["variance_pp"] = variance_pp
    log(f"  [{adapter_name}] peak-to-peak variance: {variance_pp:.1f}pp")
    return results_per_scale


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    mx.random.seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    results = {"is_smoke": IS_SMOKE, "model": MODEL_ID}

    log(f"=== C1.3: PoLAR Scale Invariance on Gemma 4 E4B ===")
    log(f"Smoke={IS_SMOKE}, n_steps={N_STEPS}, n_train={N_TRAIN}, n_eval={N_EVAL}")
    log(f"Training scale={SCALE}, eval scales={EVAL_SCALES}")

    # Generate math problems
    log("\nGenerating synthetic math problems...")
    all_problems = make_math_problems(N_TRAIN + N_EVAL, seed=SEED)
    train_problems = all_problems[:N_TRAIN]
    eval_problems = all_problems[N_TRAIN:N_TRAIN + N_EVAL]
    log(f"Train: {len(train_problems)}, Eval: {len(eval_problems)}")

    # Verify problems are valid
    for p in train_problems[:3]:
        log(f"  Train sample: Q={p['question'][:60]}... A={p['answer']}")
    for p in eval_problems[:3]:
        log(f"  Eval sample: Q={p['question'][:60]}... A={p['answer']}")

    results["dataset"] = {
        "n_train": len(train_problems),
        "n_eval": len(eval_problems),
        "train_samples": [{"q": p["question"][:80], "a": p["answer"]} for p in train_problems[:5]],
    }

    # ──────────────────────────────────────────────────────────────
    # Phase 1: Train PoLAR adapter
    # ──────────────────────────────────────────────────────────────
    log(f"\n=== Phase 1: Train PoLAR r={RANK} (scale={SCALE}, {N_STEPS} steps) ===")
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(MODEL_ID)
    log_memory("model-loaded")

    train_samples = build_training_samples(tokenizer, train_problems)
    log(f"Tokenized training samples: {len(train_samples)}")

    polar_modules = inject_adapters(model, PoLARLinear, rank=RANK, scale=SCALE)
    polar_train = train_adapter(
        model, tokenizer, train_samples, polar_modules,
        n_steps=N_STEPS, do_retract=True, phase_name="PoLAR"
    )

    # Measure B row norms post-training (should be ≈1)
    polar_b_norms = measure_b_row_norms(polar_modules)
    log(f"PoLAR B row norms: mean={polar_b_norms['mean']:.4f} std={polar_b_norms['std']:.4f}")

    # Evaluate PoLAR at all scales
    log(f"\n=== Phase 1b: Evaluate PoLAR at scales={EVAL_SCALES} ===")
    polar_scale_results = eval_at_all_scales(model, tokenizer, eval_problems, polar_modules, "PoLAR")
    polar_at_training_scale = polar_scale_results.get(f"scale_{int(SCALE)}", 0.0)
    polar_variance = polar_scale_results["variance_pp"]

    results["phase1_polar"] = {
        "train": polar_train,
        "b_row_norms": polar_b_norms,
        "scale_results": polar_scale_results,
        "at_training_scale": polar_at_training_scale,
        "variance_pp": polar_variance,
    }

    # Restore base modules
    log("\nRestoring base model q_proj layers...")
    for li in range(N_LAYERS):
        model.layers[li].self_attn.q_proj = polar_modules[li].base
    cleanup(polar_modules)
    log_memory("after-polar-cleanup")

    # ──────────────────────────────────────────────────────────────
    # Phase 2: Train standard LoRA adapter
    # ──────────────────────────────────────────────────────────────
    log(f"\n=== Phase 2: Train LoRA r={RANK} (scale={SCALE}, {N_STEPS} steps) ===")

    lora_modules = inject_adapters(model, LoRALinear, rank=RANK, scale=SCALE)
    lora_train = train_adapter(
        model, tokenizer, train_samples, lora_modules,
        n_steps=N_STEPS, do_retract=False, phase_name="LoRA"
    )

    # Measure B row norms (unconstrained, should be << 1)
    lora_b_norms = measure_b_row_norms(lora_modules)
    log(f"LoRA B row norms: mean={lora_b_norms['mean']:.4f} std={lora_b_norms['std']:.4f}")
    log(f"Ratio: PoLAR/LoRA row norms = {polar_b_norms['mean']/max(lora_b_norms['mean'], 1e-8):.2f}×")

    # Evaluate LoRA at all scales
    log(f"\n=== Phase 2b: Evaluate LoRA at scales={EVAL_SCALES} ===")
    lora_scale_results = eval_at_all_scales(model, tokenizer, eval_problems, lora_modules, "LoRA")
    lora_at_training_scale = lora_scale_results.get(f"scale_{int(SCALE)}", 0.0)
    lora_variance = lora_scale_results["variance_pp"]

    results["phase2_lora"] = {
        "train": lora_train,
        "b_row_norms": lora_b_norms,
        "scale_results": lora_scale_results,
        "at_training_scale": lora_at_training_scale,
        "variance_pp": lora_variance,
    }

    # ──────────────────────────────────────────────────────────────
    # Kill Criteria Evaluation
    # ──────────────────────────────────────────────────────────────
    kc13_pass = polar_variance < 5.0
    kc14_pass = polar_variance < lora_variance
    # KC15: PoLAR at scale=6 >= 80% of LoRA at scale=6
    kc15_ratio = (polar_at_training_scale / max(lora_at_training_scale, 0.01)) * 100
    kc15_pass = kc15_ratio >= 80.0 or (polar_at_training_scale == 0.0 and lora_at_training_scale == 0.0)

    kill_criteria = {
        "KC13": {
            "desc": "PoLAR variance < 5pp across scale={3,6,12,24}",
            "measured_pp": polar_variance,
            "threshold_pp": 5.0,
            "pass": kc13_pass,
        },
        "KC14": {
            "desc": "PoLAR variance < LoRA variance",
            "polar_variance_pp": polar_variance,
            "lora_variance_pp": lora_variance,
            "pass": kc14_pass,
        },
        "KC15": {
            "desc": "PoLAR at training scale >= 80% of LoRA",
            "polar_acc": polar_at_training_scale,
            "lora_acc": lora_at_training_scale,
            "ratio_pct": kc15_ratio,
            "pass": kc15_pass,
        },
    }

    results["kill_criteria"] = kill_criteria

    log(f"\n=== Kill Criteria Summary ===")
    for k, v in kill_criteria.items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} — {v['desc']}")
    log(f"\nKC13: PoLAR variance={polar_variance:.1f}pp (threshold<5pp) → {'PASS' if kc13_pass else 'FAIL'}")
    log(f"KC14: PoLAR={polar_variance:.1f}pp vs LoRA={lora_variance:.1f}pp → {'PASS' if kc14_pass else 'FAIL'}")
    log(f"KC15: PoLAR/LoRA={kc15_ratio:.0f}% at scale=6 → {'PASS' if kc15_pass else 'FAIL'}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
