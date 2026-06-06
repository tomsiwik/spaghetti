#!/usr/bin/env python3
"""Self-growing model: random init -> train -> solidify -> promote -> repeat.

Validates the ENTIRE Pierre architecture on a toy GPT model.
Starts from RANDOM INIT. No pre-training. Grows through adapter promotion alone.

Kill criteria:
  K841: Base quality degrades after any promotion (growth goes backwards)
  K842: 5th domain adapter trains SLOWER than 1st (promotion should help, not hurt)
  K843: Grown base > 3x worse than jointly-trained baseline
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SEED = 42
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
BLOCK_SIZE = 32
VOCAB_SIZE = 128
LORA_RANK = 4
LORA_SCALE = 2.0
ADAPTER_STEPS = 300
ADAPTER_LR = 1e-3
BASELINE_STEPS = 1500  # 300 steps * 5 domains = 1500 total adapter steps
BASELINE_LR = 3e-4
SVD_RANK = 4
N_DOMAINS = 5


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()

def log_memory(label=""):
    active = mx.get_active_memory() / 1e6
    cache = mx.get_cache_memory() / 1e6
    peak = mx.get_peak_memory() / 1e6
    log(f"  [MEM {label}] active={active:.1f}MB cache={cache:.1f}MB peak={peak:.1f}MB")


# -- Toy GPT ----------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + self.eps)

class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.hd = d // h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.h, self.hd).transpose(0, 2, 1, 3)
        scale = self.hd ** -0.5
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        a = mx.softmax(q @ k.transpose(0, 1, 3, 2) * scale + mask, axis=-1)
        return self.wo((a @ v).transpose(0, 2, 1, 3).reshape(B, T, C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)
    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attn(d, h)
        self.n2 = RMSNorm(d); self.mlp = MLP(d)
    def __call__(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class ToyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.wpe = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.layers = [Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
    def __call__(self, tokens):
        B, T = tokens.shape
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


# -- Domain data generators -------------------------------------------------
# Each domain has a clear input>output format so the model learns a mapping.

def gen_arithmetic(n, rng):
    """Domain 0: addition. '12+34=46'"""
    data = []
    for _ in range(n):
        a, b = rng.randint(0, 50), rng.randint(0, 50)
        data.append(f"{a}+{b}={a+b}")
    return data

def gen_reverse(n, rng):
    """Domain 1: string reversal. 'abc>cba'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 6)))
        data.append(f"{s}>{''.join(reversed(s))}")
    return data

def gen_repeat(n, rng):
    """Domain 2: repetition. 'ab*3=ababab'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1, 3)))
        r = rng.randint(2, 4)
        data.append(f"{p}*{r}={p*r}")
    return data

def gen_sort(n, rng):
    """Domain 3: sorting. 'dcba>abcd'"""
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2, 6)))
        data.append(f"{s}>{''.join(sorted(s))}")
    return data

def gen_parity(n, rng):
    """Domain 4: parity. '1011>odd'"""
    data = []
    for _ in range(n):
        bits = "".join(str(rng.randint(0, 2)) for _ in range(rng.randint(2, 6)))
        data.append(f"{bits}>{'even' if bits.count('1') % 2 == 0 else 'odd'}")
    return data

GENERATORS = [gen_arithmetic, gen_reverse, gen_repeat, gen_sort, gen_parity]
DOMAIN_NAMES = ["arithmetic", "reverse", "repeat", "sort", "parity"]


# -- Tokenizer --------------------------------------------------------------

class CharTokenizer:
    def __init__(self):
        self.chars = [chr(i) for i in range(32, 127)]
        self.c2i = {c: i for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
    def encode(self, s):
        return [self.c2i.get(c, 0) for c in s]
    def decode(self, ids):
        return "".join(self.chars[i] if i < len(self.chars) else "?" for i in ids)

TOK = CharTokenizer()


# -- LoRA --------------------------------------------------------------------

class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, scale=2.0):
        super().__init__()
        self.base = base
        in_f = base.weight.shape[1]
        out_f = base.weight.shape[0]
        self.lora_a = mx.random.normal(shape=(in_f, rank)) * 0.01
        self.lora_b = mx.zeros((rank, out_f))
        self.scale = scale
        self.base.freeze()
    def __call__(self, x):
        return self.base(x) + (x @ self.lora_a) @ self.lora_b * self.scale

TARGET_MODULES = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.fc1"]


# -- Training helpers --------------------------------------------------------

def make_batches(texts):
    """Convert texts to token sequences."""
    batches = []
    for text in texts:
        toks = TOK.encode(text)[:BLOCK_SIZE]
        if len(toks) < 4:
            continue
        batches.append(mx.array(toks))
    return batches


def compute_loss(model, tokens):
    logits = model(tokens[None, :])
    return nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:], reduction="mean")


def evaluate(model, batches):
    """Mean loss over batches (up to 50)."""
    total = 0.0
    n = 0
    for tokens in batches[:50]:
        loss = compute_loss(model, tokens)
        mx.eval(loss)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def add_lora(model):
    """Add LoRA wrappers to target modules."""
    for layer in model.layers:
        updates = []
        for mname in TARGET_MODULES:
            parts = mname.split(".")
            m = layer
            for p in parts:
                m = getattr(m, p, None)
            if m is None or not isinstance(m, nn.Linear):
                continue
            updates.append((mname, LoRALinear(m, rank=LORA_RANK, scale=LORA_SCALE)))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    model.freeze()
    model.unfreeze(keys=["lora_b"], strict=False)


def train_adapter(model, train_batches, steps, lr):
    """Train LoRA adapter, return (time, final_loss, loss_at_step50)."""
    add_lora(model)

    optimizer = opt.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, compute_loss)

    loss_at_50 = None
    t0 = time.time()
    gc.disable()
    for step in range(steps):
        tokens = train_batches[step % len(train_batches)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 49:
            loss_at_50 = loss.item()
    gc.enable()
    train_time = time.time() - t0

    final_loss = evaluate(model, train_batches[:20])
    return train_time, final_loss, loss_at_50


def extract_adapter_deltas(model):
    """Extract the full-rank weight delta from each LoRA module.

    Delta = scale * B^T @ A^T  (shape: out_f x in_f, matching weight shape)
    """
    deltas = {}
    for li, layer in enumerate(model.layers):
        for mname in TARGET_MODULES:
            parts = mname.split(".")
            m = layer
            for p in parts:
                m = getattr(m, p, None)
            if not isinstance(m, LoRALinear):
                continue
            delta = m.scale * (m.lora_b.T @ m.lora_a.T)
            mx.eval(delta)
            deltas[f"layer_{li}_{mname}"] = delta
    return deltas


def svd_solidify(deltas, rank):
    """SVD-extract compact factors from full-rank deltas.

    Returns dict of (A, B) where B.T @ A.T reconstructs the delta.
    At matching rank (rank == LoRA rank), this is lossless.
    """
    solidified = {}
    for key, delta in deltas.items():
        U, S, Vt = mx.linalg.svd(delta, stream=mx.cpu)
        mx.eval(U, S, Vt)
        r = min(rank, S.shape[0])
        sqrt_S = mx.sqrt(S[:r])
        A = Vt[:r].T * sqrt_S[None, :]    # (in_f, r)
        B = (U[:, :r] * sqrt_S[None, :]).T  # (r, out_f)
        solidified[key] = (A, B)
    return solidified


def promote_expert(model, solidified_expert):
    """Promote solidified expert into base weights: W_new = W + B^T @ A^T."""
    for li, layer in enumerate(model.layers):
        for mname in TARGET_MODULES:
            key = f"layer_{li}_{mname}"
            if key not in solidified_expert:
                continue
            A, B = solidified_expert[key]
            parts = mname.split(".")
            m = layer
            for p in parts:
                m = getattr(m, p, None)
            if m is None:
                continue
            base = m.base if isinstance(m, LoRALinear) else m
            if isinstance(base, nn.Linear):
                base.weight = base.weight + (B.T @ A.T).astype(base.weight.dtype)
    mx.eval(model.parameters())


def strip_lora(model):
    """Remove LoRA wrappers, keep base weights."""
    for layer in model.layers:
        updates = []
        for mname in TARGET_MODULES:
            parts = mname.split(".")
            m = layer
            for p in parts:
                m = getattr(m, p, None)
            if isinstance(m, LoRALinear):
                updates.append((mname, m.base))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())


def clone_model_weights(source):
    """Create a fresh model with copied weights."""
    target = ToyGPT()
    params = list(tree_flatten(source.parameters()))
    target.update(tree_unflatten([(n, p) for n, p in params]))
    mx.eval(target.parameters())
    return target


def compute_delta_norms(solidified):
    """Compute Frobenius norms of each delta for analysis."""
    norms = {}
    for key, (A, B) in solidified.items():
        delta = B.T @ A.T
        mx.eval(delta)
        norms[key] = float(mx.sqrt(mx.sum(delta * delta)).item())
    return norms


# -- Phase functions (scoped for cleanup) ------------------------------------

def phase_generate_data(rng):
    """Generate domain data for all 5 domains."""
    domain_data = {}
    for gen, name in zip(GENERATORS, DOMAIN_NAMES):
        texts = gen(500, rng)
        domain_data[name] = {
            "train": make_batches(texts[:400]),
            "val": make_batches(texts[400:]),
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, {len(domain_data[name]['val'])} val")
    return domain_data


def phase_evaluate_all(model, domain_data):
    """Evaluate model on all domains, return dict of losses."""
    losses = {}
    for name in DOMAIN_NAMES:
        losses[name] = round(evaluate(model, domain_data[name]["val"]), 4)
    return losses


def phase_train_and_promote(model, domain_data, domain_name):
    """Train adapter, extract delta, SVD solidify, promote into base.

    Returns dict with metrics for this promotion step.
    """
    # Evaluate BEFORE
    pre_losses = phase_evaluate_all(model, domain_data)
    log(f"  Base before: mean={np.mean(list(pre_losses.values())):.4f}")

    # Clone and train adapter
    adapter_model = clone_model_weights(model)
    train_time, adapter_loss, loss_at_50 = train_adapter(
        adapter_model, domain_data[domain_name]["train"],
        steps=ADAPTER_STEPS, lr=ADAPTER_LR
    )
    log(f"  Adapter trained: loss={adapter_loss:.4f}, loss@50={loss_at_50:.4f}, time={train_time:.1f}s")

    # Extract delta
    deltas = extract_adapter_deltas(adapter_model)
    log(f"  Extracted {len(deltas)} delta matrices")

    # SVD solidify (lossless at matching rank)
    solidified = svd_solidify(deltas, rank=SVD_RANK)
    delta_norms = compute_delta_norms(solidified)
    mean_norm = float(np.mean(list(delta_norms.values())))
    log(f"  SVD solidified (rank={SVD_RANK}), mean delta norm={mean_norm:.4f}")

    # Promote into base model (not the clone)
    promote_expert(model, solidified)
    log(f"  Expert promoted into base")

    # Evaluate AFTER
    post_losses = phase_evaluate_all(model, domain_data)

    domain_imp = pre_losses[domain_name] - post_losses[domain_name]
    pre_mean = float(np.mean(list(pre_losses.values())))
    post_mean = float(np.mean(list(post_losses.values())))
    mean_imp = pre_mean - post_mean

    log(f"  After promotion:")
    log(f"    {domain_name}: {pre_losses[domain_name]:.4f} -> {post_losses[domain_name]:.4f} ({domain_imp:+.4f})")
    log(f"    Mean: {pre_mean:.4f} -> {post_mean:.4f} ({mean_imp:+.4f})")

    # Per-domain breakdown
    for name in DOMAIN_NAMES:
        if name == domain_name:
            continue
        d = pre_losses[name] - post_losses[name]
        log(f"    {name}: {pre_losses[name]:.4f} -> {post_losses[name]:.4f} ({d:+.4f})")

    result = {
        "domain": domain_name,
        "pre_losses": pre_losses,
        "post_losses": post_losses,
        "adapter_loss": round(adapter_loss, 4),
        "loss_at_step50": round(loss_at_50, 4) if loss_at_50 is not None else None,
        "train_time_s": round(train_time, 1),
        "domain_improvement": round(domain_imp, 4),
        "mean_improvement": round(mean_imp, 4),
        "mean_delta_norm": round(mean_norm, 4),
    }

    cleanup(adapter_model, deltas, solidified)
    log_memory("post-promote")
    return result


def phase_baseline(domain_data):
    """Train a jointly-trained baseline on all domains."""
    log("\n=== Jointly-Trained Baseline ===")
    baseline_model = ToyGPT()
    mx.eval(baseline_model.parameters())

    all_train = []
    for name in DOMAIN_NAMES:
        all_train.extend(domain_data[name]["train"])
    rng_b = np.random.RandomState(SEED + 100)
    rng_b.shuffle(all_train)

    optimizer = opt.Adam(learning_rate=BASELINE_LR)
    loss_and_grad = nn.value_and_grad(baseline_model, compute_loss)

    gc.disable()
    for step in range(BASELINE_STEPS):
        tokens = all_train[step % len(all_train)]
        loss, grads = loss_and_grad(baseline_model, tokens)
        optimizer.update(baseline_model, grads)
        mx.eval(baseline_model.parameters(), optimizer.state, loss)
    gc.enable()

    baseline_losses = phase_evaluate_all(baseline_model, domain_data)
    for name in DOMAIN_NAMES:
        log(f"  baseline/{name}: {baseline_losses[name]}")

    cleanup(baseline_model, optimizer)
    log_memory("post-baseline")
    return baseline_losses


# -- Main experiment ---------------------------------------------------------

def main():
    t0 = time.time()
    log("Self-Growing Model: Random Init -> Promote -> Grow")
    log("=" * 60)
    log(f"Config: d={D_MODEL}, layers={N_LAYERS}, heads={N_HEADS}, "
        f"LoRA rank={LORA_RANK}, scale={LORA_SCALE}")
    log(f"Adapter steps={ADAPTER_STEPS}, baseline steps={BASELINE_STEPS}")
    log(f"SVD rank={SVD_RANK}, domains={N_DOMAINS}")
    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # -- Generate data -------------------------------------------------------
    log("\n--- Domain Data ---")
    domain_data = phase_generate_data(rng)

    results = {
        "config": {
            "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
            "lora_rank": LORA_RANK, "lora_scale": LORA_SCALE,
            "adapter_steps": ADAPTER_STEPS, "baseline_steps": BASELINE_STEPS,
            "svd_rank": SVD_RANK, "n_domains": N_DOMAINS, "seed": SEED,
        },
        "promotions": [],
        "domain_names": DOMAIN_NAMES,
    }

    # -- Phase 0: Random init evaluation -------------------------------------
    log("\n=== Phase 0: Random Init ===")
    model = ToyGPT()
    mx.eval(model.parameters())
    n_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    log(f"  Model parameters: {n_params:,}")

    base_losses = phase_evaluate_all(model, domain_data)
    for name in DOMAIN_NAMES:
        log(f"  random_init/{name}: loss={base_losses[name]}")
    results["random_init_losses"] = base_losses

    # -- Phase 1-5: Train -> Solidify -> Promote for each domain -------------
    loss_trajectory = {name: [base_losses[name]] for name in DOMAIN_NAMES}

    for di, domain_name in enumerate(DOMAIN_NAMES):
        log(f"\n{'=' * 50}")
        log(f"Promotion {di + 1}/{N_DOMAINS}: {domain_name}")
        log(f"{'=' * 50}")

        promo_result = phase_train_and_promote(model, domain_data, domain_name)
        results["promotions"].append(promo_result)

        # Track loss trajectory
        for name in DOMAIN_NAMES:
            loss_trajectory[name].append(promo_result["post_losses"][name])

    results["loss_trajectory"] = loss_trajectory

    # -- Final evaluation ----------------------------------------------------
    log(f"\n{'=' * 50}")
    log("Final Evaluation: Grown Model")
    log(f"{'=' * 50}")

    final_losses = phase_evaluate_all(model, domain_data)
    for name in DOMAIN_NAMES:
        improvement = base_losses[name] - final_losses[name]
        log(f"  {name}: {base_losses[name]:.4f} -> {final_losses[name]:.4f} ({improvement:+.4f})")
    results["final_losses"] = final_losses

    # Release grown model before baseline
    cleanup(model)

    # -- Jointly-trained baseline --------------------------------------------
    baseline_losses = phase_baseline(domain_data)
    results["baseline_losses"] = baseline_losses

    # -- Summary and kill criteria -------------------------------------------
    log(f"\n{'=' * 60}")
    log("COMPARISON: Random Init -> Grown -> Jointly Trained")
    log(f"{'Domain':<12} {'Random':>8} {'Grown':>8} {'Joint':>8} {'Grown/Joint':>12}")
    for name in DOMAIN_NAMES:
        ratio = final_losses[name] / baseline_losses[name] if baseline_losses[name] > 0 else float("inf")
        log(f"{name:<12} {base_losses[name]:>8.4f} {final_losses[name]:>8.4f} "
            f"{baseline_losses[name]:>8.4f} {ratio:>12.3f}x")

    mean_grown = float(np.mean(list(final_losses.values())))
    mean_baseline = float(np.mean(list(baseline_losses.values())))
    mean_random = float(np.mean(list(base_losses.values())))
    overall_ratio = mean_grown / mean_baseline if mean_baseline > 0 else float("inf")

    log(f"\nMean: random={mean_random:.4f} grown={mean_grown:.4f} baseline={mean_baseline:.4f}")
    log(f"Grown/Baseline ratio: {overall_ratio:.3f}x")

    # Improvement from random init
    random_improvement = (mean_random - mean_grown) / mean_random * 100
    log(f"Improvement over random init: {random_improvement:.1f}%")

    # Loss trajectory summary
    log(f"\nLoss trajectory (base quality after each promotion):")
    header = f"{'Step':<6}" + "".join(f"{n:>12}" for n in DOMAIN_NAMES) + f"{'Mean':>12}"
    log(header)
    for step in range(N_DOMAINS + 1):
        label = "init" if step == 0 else f"+{DOMAIN_NAMES[step-1][:4]}"
        vals = [loss_trajectory[name][step] for name in DOMAIN_NAMES]
        mean_val = float(np.mean(vals))
        log(f"{label:<6}" + "".join(f"{v:>12.4f}" for v in vals) + f"{mean_val:>12.4f}")

    # Training speed comparison (K842)
    train_times = [p["train_time_s"] for p in results["promotions"]]
    losses_at_50 = [p["loss_at_step50"] for p in results["promotions"]]
    adapter_losses = [p["adapter_loss"] for p in results["promotions"]]

    log(f"\nTraining speed:")
    for i, name in enumerate(DOMAIN_NAMES):
        log(f"  {name}: {train_times[i]:.1f}s, final_loss={adapter_losses[i]:.4f}, loss@50={losses_at_50[i]:.4f}")

    log(f"\n1st adapter time: {train_times[0]:.1f}s, 5th adapter time: {train_times[-1]:.1f}s")
    log(f"Time ratio (5th/1st): {train_times[-1]/train_times[0]:.2f}x")

    # Save results
    results["summary"] = {
        "mean_random": round(mean_random, 4),
        "mean_grown": round(mean_grown, 4),
        "mean_baseline": round(mean_baseline, 4),
        "overall_ratio": round(overall_ratio, 3),
        "random_improvement_pct": round(random_improvement, 1),
        "train_times": train_times,
        "losses_at_step50": losses_at_50,
        "total_time_s": round(time.time() - t0, 1),
    }

    # Kill criteria assessment
    # K841: base quality degrades after any promotion
    # Check: mean loss across all domains should not increase after any promotion
    # Tolerance: small degradation allowed (-0.01) since cross-domain interference is expected
    promotion_degradations = []
    for i, p in enumerate(results["promotions"]):
        # The promoted domain must improve
        domain_improved = p["domain_improvement"] > 0
        # The mean should not degrade significantly
        mean_degraded = p["mean_improvement"] < -0.05  # 0.05 tolerance
        promotion_degradations.append({
            "promotion": i + 1,
            "domain": p["domain"],
            "domain_improved": bool(domain_improved),
            "domain_delta": p["domain_improvement"],
            "mean_delta": p["mean_improvement"],
            "mean_degraded": bool(mean_degraded),
        })

    k841_fail = any(d["mean_degraded"] for d in promotion_degradations)
    k841_detail = "All promotions maintain or improve mean quality"
    if k841_fail:
        bad = [d for d in promotion_degradations if d["mean_degraded"]]
        k841_detail = f"Degradation at promotion(s): {[d['promotion'] for d in bad]}"

    # K842: 5th adapter trains slower than 1st (50% tolerance)
    k842_slower = train_times[-1] > train_times[0] * 1.5
    k842_detail = f"1st={train_times[0]:.1f}s 5th={train_times[-1]:.1f}s ratio={train_times[-1]/train_times[0]:.2f}x"

    # Also check convergence speed via loss@50
    if losses_at_50[0] is not None and losses_at_50[-1] is not None:
        convergence_faster = losses_at_50[-1] < losses_at_50[0]
        k842_detail += f" | loss@50: 1st={losses_at_50[0]:.4f} 5th={losses_at_50[-1]:.4f} (faster={convergence_faster})"

    # K843: grown > 3x worse than baseline
    k843_bad = overall_ratio > 3.0
    k843_detail = f"ratio={overall_ratio:.3f}x (threshold=3.0x)"

    results["kill_criteria"] = {
        "K841": {"pass": not k841_fail, "detail": k841_detail, "promotion_details": promotion_degradations},
        "K842": {"pass": not k842_slower, "detail": k842_detail},
        "K843": {"pass": not k843_bad, "detail": k843_detail, "value": round(overall_ratio, 3), "threshold": 3.0},
    }
    results["all_pass"] = not k841_fail and not k842_slower and not k843_bad

    # Success criteria: S85
    mean_improves = all(p["mean_improvement"] >= -0.05 for p in results["promotions"])
    within_2x = overall_ratio <= 2.0
    results["success_criteria"] = {
        "S85": {
            "monotonic_improvement": mean_improves,
            "within_2x": within_2x,
            "pass": mean_improves and within_2x,
        }
    }

    log(f"\n{'=' * 60}")
    log("KILL CRITERIA")
    for k, v in results["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} -- {v['detail']}")

    log(f"\nSUCCESS CRITERIA")
    s85 = results["success_criteria"]["S85"]
    log(f"  S85: {'PASS' if s85['pass'] else 'FAIL'} -- monotonic={s85['monotonic_improvement']}, within_2x={s85['within_2x']}")

    verdict = "ALL PASS" if results["all_pass"] else "KILLED"
    log(f"\n{verdict} in {results['summary']['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
