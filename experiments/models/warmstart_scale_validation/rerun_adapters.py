#!/usr/bin/env python3
"""Rerun adapter training + composition with freeze bug fix.

Fixes from adversarial review:
  1. After attaching LoRA modules, freeze base_weight + pre_norm_weight
  2. ADAPTER_LR reduced from 1e-3 to 1e-4
  3. Gradient norm logging every 10 steps
  4. FP32 baseline architectural difference noted (not fixed -- reusing checkpoint)
  5. K2 reformulated: val PPL improvement between steps 4000 and 8000

Reuses saved checkpoints:
  - checkpoints/ternary_warmstart.npz (base model)
  - data_cache/ (train/val/domain tokens)
"""

import gc
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
import numpy as np

# Memory limits (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(4 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results_fixed.json"
DATA_CACHE = EXPERIMENT_DIR / "data_cache"
ADAPTER_DIR = EXPERIMENT_DIR / "adapters_fixed"
CHECKPOINT_DIR = EXPERIMENT_DIR / "checkpoints"

# Architecture (must match original)
D_MODEL = 1024
N_LAYERS = 8
N_HEADS = 16
HEAD_DIM = D_MODEL // N_HEADS
BLOCK_SIZE = 256
MLP_DIM = 4 * D_MODEL
VOCAB_SIZE = 50257

# Adapter settings (FIX #2: lr 1e-3 -> 1e-4)
LORA_RANK = 16
LORA_ALPHA = 32.0
ADAPTER_TRAIN_STEPS = 1000
ADAPTER_LR = 1e-4  # FIXED: was 1e-3
ADAPTER_BATCH_SIZE = 16

DOMAIN_KEYWORDS = {
    "science": ["biology", "chemistry", "physics", "molecule", "atom", "cell",
                 "organism", "experiment", "hypothesis", "scientific"],
    "history": ["century", "empire", "war", "ancient", "civilization", "dynasty",
                "revolution", "colonial", "medieval", "historical"],
    "technology": ["software", "algorithm", "computer", "programming", "data",
                   "network", "digital", "system", "code", "technology"],
}


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    print(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ============================================================================
# Model components (identical to original)
# ============================================================================

class WarmStartLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = mx.random.normal(shape=(out_features, in_features)) * scale
        self.pre_quant_norm = nn.RMSNorm(in_features)
        self._ternary_mode = False

    def __call__(self, x):
        x = self.pre_quant_norm(x)
        if self._ternary_mode:
            w = self.weight
            alpha = mx.mean(mx.abs(w))
            w_scaled = w / (alpha + 1e-7)
            w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha
            w_ste = w + mx.stop_gradient(w_q - w)
            return x @ w_ste.T
        else:
            return x @ self.weight.T


class FP32Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def __call__(self, x):
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = linear_cls(n_embd, n_embd)
        self.wk = linear_cls(n_embd, n_embd)
        self.wv = linear_cls(n_embd, n_embd)
        self.wo = linear_cls(n_embd, n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        mask = mx.triu(mx.full((T, T), float('-inf')), k=1)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd: int, linear_cls):
        super().__init__()
        self.fc1 = linear_cls(n_embd, 4 * n_embd)
        self.fc2 = linear_cls(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, linear_cls):
        super().__init__()
        self.norm1 = nn.RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, linear_cls)
        self.norm2 = nn.RMSNorm(n_embd)
        self.mlp = MLP(n_embd, linear_cls)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, linear_cls):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.layers = [Block(n_embd, n_head, linear_cls) for _ in range(n_layer)]
        self.norm_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def __call__(self, tokens):
        B, T = tokens.shape
        pos = mx.arange(T)
        x = self.wte(tokens) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return self.lm_head(x)


class LoRALinear(nn.Module):
    """LoRA adapter wrapping a frozen base weight.

    FIX: base_weight and pre_norm_weight use mx.stop_gradient in forward pass
    to guarantee they are never updated, regardless of freeze state.
    """
    def __init__(self, base_weight, rank: int, alpha: float, has_pre_norm=False,
                 pre_norm_weight=None):
        super().__init__()
        out_features, in_features = base_weight.shape
        self.base_weight = base_weight
        self.lora_A = mx.random.normal(shape=(rank, in_features)) * (1.0 / math.sqrt(in_features))
        self.lora_B = mx.zeros((out_features, rank))
        self.scale = alpha / rank
        self.has_pre_norm = has_pre_norm
        if has_pre_norm and pre_norm_weight is not None:
            self.pre_norm_weight = pre_norm_weight
        self._ternary_mode = False

    def __call__(self, x):
        if self.has_pre_norm:
            # FIX: stop_gradient on pre_norm_weight
            pnw = mx.stop_gradient(self.pre_norm_weight)
            norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)
            x = x * norm * pnw

        # FIX: stop_gradient on base_weight -- ensures it is NEVER updated
        w = mx.stop_gradient(self.base_weight)
        if self._ternary_mode:
            alpha_q = mx.mean(mx.abs(w))
            w_scaled = w / (alpha_q + 1e-7)
            w_q = mx.clip(mx.round(w_scaled), -1, 1) * alpha_q
            w = w_q  # no STE needed since base is frozen anyway

        base_out = x @ w.T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scale
        return base_out + lora_out


# ============================================================================
# Utilities
# ============================================================================

def count_params(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.parameters()))


def count_trainable(model):
    return sum(v.size for _, v in nn.utils.tree_flatten(model.trainable_parameters()))


def get_batch(tokens, batch_size, block_size, rng):
    max_start = len(tokens) - block_size - 1
    starts = [rng.randint(0, max_start) for _ in range(batch_size)]
    inputs = np.stack([tokens[s:s + block_size] for s in starts])
    targets = np.stack([tokens[s + 1:s + block_size + 1] for s in starts])
    return mx.array(inputs), mx.array(targets)


def compute_ppl(model, val_tokens, n_batches=50, batch_size=16, block_size=256):
    rng = random.Random(0)
    total_loss = 0.0
    for _ in range(n_batches):
        inputs, targets = get_batch(val_tokens, batch_size, block_size, rng)
        logits = model(inputs)
        B, T, V = logits.shape
        loss = nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )
        mx.eval(loss)
        total_loss += loss.item()
        del logits, loss
    return math.exp(total_loss / n_batches)


def set_ternary_mode(model, enabled):
    for layer in model.layers:
        for proj in [layer.attn.wq, layer.attn.wk, layer.attn.wv, layer.attn.wo]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled
        for proj in [layer.mlp.fc1, layer.mlp.fc2]:
            if hasattr(proj, '_ternary_mode'):
                proj._ternary_mode = enabled


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.0):
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-BLOCK_SIZE:]

    for _ in range(max_tokens):
        input_ids = mx.array([tokens[-BLOCK_SIZE:]])
        logits = model(input_ids)
        mx.eval(logits)
        next_logits = logits[0, -1, :]

        if temperature == 0.0:
            next_token = mx.argmax(next_logits).item()
        else:
            probs = mx.softmax(next_logits / temperature, axis=-1)
            mx.eval(probs)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)).item()

        tokens.append(next_token)
        del logits, next_logits

    return tokenizer.decode(tokens)


def compute_grad_norm(grads):
    """Compute L2 norm of all gradient tensors."""
    flat = nn.utils.tree_flatten(grads)
    total = 0.0
    for _, v in flat:
        total += mx.sum(v * v).item()
    return math.sqrt(total)


def load_base_model(ckpt_path):
    """Load ternary warm-start base model from checkpoint."""
    model = GPTModel(VOCAB_SIZE, BLOCK_SIZE, D_MODEL, N_HEADS, N_LAYERS, WarmStartLinear)
    weights = dict(mx.load(str(ckpt_path)))
    model.load_weights(list(weights.items()))
    set_ternary_mode(model, True)
    mx.eval(model.parameters())
    return model


def attach_lora(model):
    """Attach LoRA to all attention projections with PROPER FREEZING.

    Fix: We freeze the entire model, then attach LoRA modules, then freeze
    again to catch base_weight/pre_norm_weight that were added as new params.
    Then we unfreeze ONLY lora_A and lora_B.

    Additionally, LoRALinear uses mx.stop_gradient on base_weight in the
    forward pass as a belt-and-suspenders guarantee.
    """
    # Step 1: Freeze entire base model
    model.freeze()

    # Step 2: Replace attention projections with LoRA-wrapped versions
    for layer_idx, layer in enumerate(model.layers):
        for proj_name in ["wq", "wk", "wv", "wo"]:
            proj = getattr(layer.attn, proj_name)
            base_w = proj.weight
            has_norm = hasattr(proj, 'pre_quant_norm')
            norm_w = proj.pre_quant_norm.weight if has_norm else None
            lora = LoRALinear(base_w, LORA_RANK, LORA_ALPHA,
                              has_pre_norm=has_norm, pre_norm_weight=norm_w)
            lora._ternary_mode = True
            setattr(layer.attn, proj_name, lora)

    # Step 3: FIX -- freeze everything again (catches newly added base_weight, pre_norm_weight)
    model.freeze()

    # Step 4: Unfreeze ONLY lora_A and lora_B
    for layer in model.layers:
        for proj_name in ["wq", "wk", "wv", "wo"]:
            proj = getattr(layer.attn, proj_name)
            if isinstance(proj, LoRALinear):
                proj.unfreeze(keys=["lora_A", "lora_B"])

    mx.eval(model.parameters())


# ============================================================================
# Phase: Train one adapter
# ============================================================================

def phase_train_adapter(domain, domain_data, val_tokens, base_ckpt_path):
    """Train a single LoRA adapter with proper freeze handling."""
    print(f"\n--- Training adapter: {domain} ---")

    model = load_base_model(base_ckpt_path)

    # Compute base PPL before adapter
    base_ppl = compute_ppl(model, domain_data, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    base_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    print(f"  Base PPL on {domain}: {base_ppl:.2f}, on val: {base_val_ppl:.2f}")

    # Attach LoRA with proper freezing
    attach_lora(model)

    # VERIFY trainable param count (should be ~1M, NOT 34.6M)
    trainable = count_trainable(model)
    total = count_params(model)
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    expected_trainable = 4 * 2 * LORA_RANK * D_MODEL * N_LAYERS  # 4 projs * (A + B) * r * d * layers
    assert abs(trainable - expected_trainable) < 100, \
        f"FREEZE BUG: trainable={trainable:,} but expected ~{expected_trainable:,}"
    print(f"  VERIFIED: trainable params match expected ({expected_trainable:,})")

    # Train with gradient norm monitoring (FIX #3)
    optimizer = opt.AdamW(learning_rate=ADAPTER_LR, weight_decay=0.0)

    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        B, T, V = logits.shape
        return nn.losses.cross_entropy(
            logits.reshape(B * T, V), targets.reshape(B * T), reduction="mean"
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    rng = random.Random(123)

    gc.disable()
    t0 = time.time()
    grad_norms = []
    losses_log = []

    for step in range(1, ADAPTER_TRAIN_STEPS + 1):
        inputs, targets = get_batch(domain_data, ADAPTER_BATCH_SIZE, BLOCK_SIZE, rng)
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        loss_val = loss.item()
        losses_log.append(loss_val)

        # FIX #3: Gradient norm logging every 10 steps
        if step % 10 == 0:
            gn = compute_grad_norm(grads)
            grad_norms.append({"step": step, "grad_norm": gn, "loss": loss_val})
            if step % 100 == 0 or step == ADAPTER_TRAIN_STEPS:
                elapsed = time.time() - t0
                print(f"    step {step}/{ADAPTER_TRAIN_STEPS} | loss {loss_val:.4f} | "
                      f"grad_norm {gn:.4f} | {step / elapsed:.1f} steps/s")
            # Early warning: gradient explosion
            if gn > 100.0:
                print(f"  WARNING: gradient norm {gn:.2f} > 100 at step {step}, possible divergence")

    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    adapted_ppl = compute_ppl(model, domain_data, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    adapted_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    improvement = (base_ppl - adapted_ppl) / base_ppl * 100
    val_degradation = (adapted_val_ppl - base_val_ppl) / base_val_ppl * 100
    print(f"  Adapted PPL on {domain}: {adapted_ppl:.2f} (was {base_ppl:.2f}, {improvement:+.1f}%)")
    print(f"  Adapted PPL on val: {adapted_val_ppl:.2f} (was {base_val_ppl:.2f}, {val_degradation:+.1f}%)")

    # Save adapter weights
    ADAPTER_DIR.mkdir(exist_ok=True)
    adapter_weights = {}
    for name, param in nn.utils.tree_flatten(model.trainable_parameters()):
        adapter_weights[name] = param
    adapter_path = ADAPTER_DIR / f"{domain}_lora.npz"
    mx.savez(str(adapter_path), **adapter_weights)
    print(f"  Saved adapter: {adapter_path}")

    # Generate text
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    domain_prompts = {
        "science": "The process of cellular respiration",
        "history": "The fall of the Roman Empire",
        "technology": "Artificial neural networks are",
    }
    prompt = domain_prompts.get(domain, "The")
    text = generate_text(model, enc, prompt, max_tokens=80, temperature=0.0)
    print(f"  [ADAPTED-{domain}] '{prompt}' -> {text[:200]}...")

    result = {
        "domain": domain,
        "base_domain_ppl": base_ppl,
        "base_val_ppl": base_val_ppl,
        "adapted_domain_ppl": adapted_ppl,
        "adapted_val_ppl": adapted_val_ppl,
        "improvement_pct": round(improvement, 2),
        "val_degradation_pct": round(val_degradation, 2),
        "trainable_params": trainable,
        "total_params": total,
        "train_time_s": round(train_time, 1),
        "text_sample": {"prompt": prompt, "generated": text},
        "adapter_lr": ADAPTER_LR,
        "final_loss": losses_log[-1],
        "loss_trajectory": [losses_log[i] for i in range(99, ADAPTER_TRAIN_STEPS, 100)],
        "grad_norm_samples": grad_norms[::10],  # every 100 steps
    }
    cleanup(model, optimizer)
    return result


# ============================================================================
# Phase: Composition test
# ============================================================================

def phase_composition(val_tokens, domain_tokens, base_ckpt_path):
    """Test composing all 3 domain adapters via 1/N averaging."""
    print(f"\n--- Composition Test ---")

    model = load_base_model(base_ckpt_path)

    # Base PPL
    base_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    base_domain_ppls = {}
    for domain in DOMAIN_KEYWORDS:
        base_domain_ppls[domain] = compute_ppl(
            model, domain_tokens[domain], n_batches=30, batch_size=ADAPTER_BATCH_SIZE
        )

    # Attach LoRA with proper freezing
    attach_lora(model)

    # Load all adapter weights and average them
    domains = list(DOMAIN_KEYWORDS.keys())
    all_adapter_weights = {}
    for domain in domains:
        adapter_path = ADAPTER_DIR / f"{domain}_lora.npz"
        if not adapter_path.exists():
            print(f"  ERROR: adapter {adapter_path} not found")
            cleanup(model)
            return {"error": f"adapter_{domain}_missing"}
        all_adapter_weights[domain] = dict(mx.load(str(adapter_path)))

    # Set averaged LoRA weights
    for layer_idx, layer in enumerate(model.layers):
        for proj_name in ["wq", "wk", "wv", "wo"]:
            proj = getattr(layer.attn, proj_name)
            a_key = f"layers.{layer_idx}.attn.{proj_name}.lora_A"
            b_key = f"layers.{layer_idx}.attn.{proj_name}.lora_B"

            avg_A = None
            avg_B = None
            n_found = 0
            for domain in domains:
                dw = all_adapter_weights[domain]
                if a_key in dw and b_key in dw:
                    if avg_A is None:
                        avg_A = dw[a_key]
                        avg_B = dw[b_key]
                    else:
                        avg_A = avg_A + dw[a_key]
                        avg_B = avg_B + dw[b_key]
                    n_found += 1

            if n_found > 0 and avg_A is not None:
                proj.lora_A = avg_A / n_found
                proj.lora_B = avg_B / n_found

    mx.eval(model.parameters())

    # Evaluate composed model
    composed_val_ppl = compute_ppl(model, val_tokens, n_batches=30, batch_size=ADAPTER_BATCH_SIZE)
    composed_domain_ppls = {}
    for domain in DOMAIN_KEYWORDS:
        composed_domain_ppls[domain] = compute_ppl(
            model, domain_tokens[domain], n_batches=30, batch_size=ADAPTER_BATCH_SIZE
        )

    composition_ratio = composed_val_ppl / base_val_ppl

    print(f"\n  Composition Results:")
    print(f"  Base val PPL: {base_val_ppl:.2f}")
    print(f"  Composed val PPL: {composed_val_ppl:.2f}")
    print(f"  Composition ratio: {composition_ratio:.3f}")
    for domain in DOMAIN_KEYWORDS:
        base_d = base_domain_ppls[domain]
        comp_d = composed_domain_ppls[domain]
        print(f"  {domain}: base={base_d:.2f}, composed={comp_d:.2f}, ratio={comp_d/base_d:.3f}")

    # Generate text
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    comp_prompts = [
        "The process of photosynthesis",
        "In the year 1776",
        "Machine learning algorithms",
    ]
    samples = []
    for prompt in comp_prompts:
        text = generate_text(model, enc, prompt, max_tokens=80, temperature=0.0)
        samples.append({"prompt": prompt, "generated": text})
        print(f"\n  [COMPOSED] '{prompt}' -> {text[:200]}...")

    result = {
        "base_val_ppl": base_val_ppl,
        "composed_val_ppl": composed_val_ppl,
        "composition_ratio": composition_ratio,
        "base_domain_ppls": base_domain_ppls,
        "composed_domain_ppls": composed_domain_ppls,
        "text_samples": samples,
    }
    cleanup(model)
    for d in all_adapter_weights.values():
        del d
    gc.collect()
    mx.clear_cache()
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    t0_total = time.time()
    log_memory("start")

    print("=" * 60)
    print("ADAPTER RERUN: Freeze bug fixed, LR=1e-4, grad norm logging")
    print("=" * 60)

    # Verify checkpoint exists
    ternary_ckpt = CHECKPOINT_DIR / "ternary_warmstart.npz"
    if not ternary_ckpt.exists():
        print(f"ERROR: Ternary checkpoint not found at {ternary_ckpt}")
        return

    # Load cached data
    val_tokens = np.fromfile(str(DATA_CACHE / "val_tokens.bin"), dtype=np.int32)
    print(f"Val tokens: {len(val_tokens):,}")

    # Load domain data
    domain_tokens = {}
    for domain in DOMAIN_KEYWORDS:
        path = DATA_CACHE / "domains" / f"{domain}_tokens.bin"
        domain_tokens[domain] = np.fromfile(str(path), dtype=np.int32)
        print(f"  {domain}: {len(domain_tokens[domain]):,} tokens")
    log_memory("after-data")

    # Load prior results for base metrics
    prior_results_path = EXPERIMENT_DIR / "results.json"
    with open(prior_results_path) as f:
        prior_results = json.load(f)

    # Train adapters
    adapter_results = {}
    for domain in DOMAIN_KEYWORDS:
        result = phase_train_adapter(domain, domain_tokens[domain], val_tokens, ternary_ckpt)
        adapter_results[domain] = result
        log_memory(f"after-adapter-{domain}")

    # Composition test
    composition_result = phase_composition(val_tokens, domain_tokens, ternary_ckpt)
    log_memory("after-composition")

    # ========================================================================
    # Kill criteria assessment
    # ========================================================================
    print("\n" + "=" * 60)
    print("KILL CRITERIA ASSESSMENT (FIXED)")
    print("=" * 60)

    # K1: Already PASS from prior run (base model produces coherent text)
    k1_pass = prior_results["kill_criteria"]["K1_coherent_text"]
    warm_ppl = prior_results["warm_start"]["ppl"]
    fp32_ppl = prior_results["fp32_baseline"]["ppl"]
    ppl_ratio = warm_ppl / fp32_ppl
    print(f"\n[K1] Coherent text at d=1024: {'PASS' if k1_pass else 'FAIL'}")
    print(f"  PPL ratio: {ppl_ratio:.3f}x (from prior run)")

    # K2: REFORMULATED -- val PPL improvement between steps 4000 and 8000
    val_traj = prior_results["warm_start"]["val_trajectory"]
    ppl_4000 = None
    ppl_8000 = None
    for v in val_traj:
        if v["step"] == 4000:
            ppl_4000 = v["ppl"]
        if v["step"] == 8000:
            ppl_8000 = v["ppl"]
    if ppl_4000 and ppl_8000:
        k2_improvement = (ppl_4000 - ppl_8000) / ppl_4000 * 100
        k2_pass = k2_improvement > 5.0  # at least 5% improvement
        print(f"\n[K2] Learning progress (reformulated)")
        print(f"  PPL at step 4000: {ppl_4000:.2f}")
        print(f"  PPL at step 8000: {ppl_8000:.2f}")
        print(f"  Improvement: {k2_improvement:.1f}% (threshold: >5%)")
        print(f"  Result: {'PASS' if k2_pass else 'FAIL'}")
    else:
        k2_pass = False
        k2_improvement = 0
        print(f"\n[K2] Could not compute (missing val trajectory data)")

    # K3: Composition ratio
    comp_ratio = composition_result.get("composition_ratio", float('inf'))
    k3_pass = comp_ratio < 2.0
    print(f"\n[K3] Adapter composition (ratio < 2.0)")
    print(f"  Composition ratio: {comp_ratio:.3f}")
    print(f"  Result: {'PASS' if k3_pass else 'FAIL'}")

    # Adapter quality summary
    print(f"\n--- Adapter Quality ---")
    s2_pass = True
    for domain, a in adapter_results.items():
        imp = a["improvement_pct"]
        deg = a["val_degradation_pct"]
        print(f"  {domain}: domain PPL {a['base_domain_ppl']:.1f} -> {a['adapted_domain_ppl']:.1f} "
              f"({imp:+.1f}%), val degradation {deg:+.1f}%")
        if imp <= 0:
            s2_pass = False

    # Summary
    total_time = time.time() - t0_total
    print(f"\n{'=' * 60}")
    print(f"SUMMARY")
    print(f"{'=' * 60}")
    print(f"  K1 (coherent text):    {'PASS' if k1_pass else 'FAIL'} (from prior run)")
    print(f"  K2 (learning progress): {'PASS' if k2_pass else 'FAIL'} ({k2_improvement:.1f}% improvement)")
    print(f"  K3 (composition):      {'PASS' if k3_pass else 'FAIL'} (ratio={comp_ratio:.3f})")
    print(f"  S2 (adapter improves): {'PASS' if s2_pass else 'FAIL'}")
    print(f"  Time: {total_time:.0f}s ({total_time/60:.1f} min)")

    overall = "SUPPORTED" if all([k1_pass, k2_pass, k3_pass]) else "PARTIAL"
    print(f"  Overall: {overall}")

    # ========================================================================
    # Save results
    # ========================================================================
    results = {
        "experiment": "warmstart_scale_validation_FIXED",
        "fix_description": (
            "Rerun of adapter+composition phases with freeze bug fixed. "
            "Prior run had 34.6M trainable params (base_weight not frozen); "
            "fixed run has ~1M trainable params (LoRA A+B only). "
            "LR reduced from 1e-3 to 1e-4. Gradient norm logging added."
        ),
        "prior_base_results": {
            "fp32_ppl": fp32_ppl,
            "warm_start_ppl": warm_ppl,
            "ppl_ratio": ppl_ratio,
        },
        "architecture_note": (
            "FP32 baseline uses nn.Linear (no extra RMSNorm). "
            "Warm-start uses WarmStartLinear with extra RMSNorm in every projection. "
            "This makes the PPL ratio comparison not perfectly apples-to-apples. "
            "The warm-start has ~73K extra norm parameters. "
            "The actual ternary penalty may be slightly higher than reported 1.037x."
        ),
        "adapters": adapter_results,
        "composition": composition_result,
        "kill_criteria": {
            "K1_coherent_text": k1_pass,
            "K2_learning_progress": k2_pass,
            "K2_improvement_pct": k2_improvement if ppl_4000 else None,
            "K3_composition_ratio": k3_pass,
            "K3_ratio_value": comp_ratio,
        },
        "success_criteria": {
            "S1_grammatical_text": k1_pass,
            "S2_adapter_improvement": s2_pass,
        },
        "total_adapter_time_s": round(total_time, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
