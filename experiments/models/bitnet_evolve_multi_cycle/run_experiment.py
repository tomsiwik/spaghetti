#!/usr/bin/env python3
"""
Multi-Cycle Evolve Convergence Experiment

Tests whether retrain-from-scratch improves monotonically over 3+ cycles
on 2 domains (medical, code) with quality-gated selection.

Each cycle: fresh LoRA init -> train 1000 steps -> evaluate PPL + KR-Test + cosine.
Quality gate keeps best adapter per domain across cycles.

Kill criteria:
  K1: PPL plateaus or regresses (PPL_i^(3) >= PPL_i^(1) for ANY domain)
  K2: KR-Test regresses (KR_i^(3) < KR_i^(1) - 0.02 for ANY domain)
  K3: Composition safety violated (max |cos| > 0.05 for any cross-domain pair)

Runtime: ~30-40 min on Apple Silicon (MLX)
"""

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety (CODING_GUIDELINES mandatory)
device = mx.device_info()
mx.set_memory_limit(device["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 1000
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 25
MAX_CONTEXT_TOKENS = 192
N_CONTRASTIVE_PAIRS = 50
N_CYCLES = 3
SEEDS = [42, 137, 2024]  # Different seed per cycle

EXPERIMENT_DIR = Path(__file__).parent
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse data from prior experiments if available
PRIOR_DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

DOMAINS = {
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
    "code": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "max_samples_train": 500,
        "max_samples_val": 50,
    },
}


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    """Print current memory usage."""
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    """Release MLX memory between phases."""
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Ternary weight unpacking (from bitnet_2b_real_composition)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    w0 = (packed_weights & 3).astype(mx.bfloat16) - 1
    w1 = ((packed_weights >> 2) & 3).astype(mx.bfloat16) - 1
    w2 = ((packed_weights >> 4) & 3).astype(mx.bfloat16) - 1
    w3 = ((packed_weights >> 6) & 3).astype(mx.bfloat16) - 1
    unpacked = mx.concatenate([w0, w1, w2, w3], axis=0)[:out_features]
    scale = weight_scale.astype(mx.bfloat16)
    if invert_scale:
        unpacked = unpacked / scale
    else:
        unpacked = unpacked * scale
    return unpacked


def replace_bitlinear_with_linear(model):
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if isinstance(module, BitLinear):
                unpacked_w = unpack_ternary(
                    module.weight, module.out_features,
                    module.weight_scale, module.invert_weight_scales,
                )
                has_bias = module.bias is not None
                linear = nn.Linear(module.in_features, module.out_features, bias=has_bias)
                linear.weight = unpacked_w
                if has_bias:
                    linear.bias = module.bias
                updates.append((key, linear))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())
    log(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# LoRA (custom implementation matching retrain_evolve)
# ===========================================================================
class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=16, scale=20.0):
        super().__init__()
        self.linear = base_linear
        self.r = r
        self.scale = scale
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]
        s = 1.0 / math.sqrt(in_features)
        self.lora_a = mx.random.uniform(low=-s, high=s, shape=(in_features, r))
        self.lora_b = mx.zeros((r, out_features))

    def __call__(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out


def apply_lora(model, rank=16, scale=20.0):
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora = LoRALinear(module, r=rank, scale=scale)
                updates.append((key, lora))
                count += 1
        if updates:
            layer.update_modules(tree_unflatten(updates))
    log(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, val in tree_flatten(model.parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(val)
    mx.eval(params)
    return params


def reinit_lora(model, seed):
    """Re-initialize all LoRA params with a specific seed."""
    mx.random.seed(seed)
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                s = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-s, high=s, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def zero_lora(model):
    """Set all LoRA params to zero (for base evaluation)."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params):
    model.update(tree_unflatten(list(adapter_params.items())))
    mx.eval(model.parameters())


def save_adapter(model, path):
    path.mkdir(parents=True, exist_ok=True)
    params = get_lora_params(model)
    mx.savez(str(path / "adapter.npz"), **params)


def load_adapter(path):
    return dict(mx.load(str(path / "adapter.npz")))


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name, domain_config):
    """Download/prepare domain data, returning (train_data, val_data) as lists of dicts."""
    data_dir = DATA_DIR / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        log(f"  {domain_name}: data already exists")
        train_data = [json.loads(l) for l in open(train_path)]
        val_data = [json.loads(l) for l in open(valid_path)]
        return train_data, val_data

    # Check if prior experiment has data we can reuse
    prior_map = {"medical": "medical", "code": "python"}
    prior_name = prior_map.get(domain_name)
    prior_train = PRIOR_DATA_DIR / prior_name / "train.jsonl" if prior_name else None
    prior_valid = PRIOR_DATA_DIR / prior_name / "valid.jsonl" if prior_name else None

    if prior_train and prior_train.exists() and prior_valid.exists():
        log(f"  {domain_name}: reusing data from bitnet_2b_real_composition/{prior_name}")
        data_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(prior_train, train_path)
        shutil.copy(prior_valid, valid_path)
        train_data = [json.loads(l) for l in open(train_path)]
        val_data = [json.loads(l) for l in open(valid_path)]
        return train_data, val_data

    # Download from HuggingFace
    from datasets import load_dataset as hf_load
    data_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Downloading {domain_config['hf_dataset']}...")

    ds = hf_load(domain_config["hf_dataset"])
    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]

    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response"]:
            if alt in split_data.column_names:
                text_key = alt
                break

    max_train = domain_config["max_samples_train"]
    max_val = domain_config["max_samples_val"]

    texts = []
    for row in split_data:
        t = row[text_key]
        if isinstance(t, str) and len(t.strip()) > 20:
            texts.append(t.strip())
        if len(texts) >= max_train + max_val:
            break

    train_texts = texts[:max_train]
    val_texts = texts[max_train:max_train + max_val]

    train_data = [{"text": t} for t in train_texts]
    val_data = [{"text": t} for t in val_texts]

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    with open(valid_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    log(f"  {domain_name}: {len(train_data)} train, {len(val_data)} val")
    return train_data, val_data


# ===========================================================================
# Training
# ===========================================================================
def train_adapter(model, tokenizer, train_data, n_iters, seed):
    """Train LoRA adapter from fresh init with given seed."""
    reinit_lora(model, seed)

    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"])

    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    # Tokenize data
    train_tokens = []
    for item in train_data:
        toks = tokenizer.encode(item["text"])
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    def loss_fn(model, tokens):
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Shuffle training order with seed
    rng = random.Random(seed)
    indices = list(range(len(train_tokens)))

    losses = []
    t0 = time.time()

    gc.disable()
    for step in range(n_iters):
        if step % len(indices) == 0:
            rng.shuffle(indices)
        idx = indices[step % len(indices)]
        tokens = train_tokens[idx]

        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)

        losses.append(loss.item())
        if (step + 1) % 200 == 0:
            avg = sum(losses[-200:]) / len(losses[-200:])
            log(f"    Step {step+1}/{n_iters}: loss={avg:.4f}")
    gc.enable()
    gc.collect()

    train_time = time.time() - t0
    final_loss = sum(losses[-50:]) / len(losses[-50:])
    first_loss = sum(losses[:50]) / 50
    log(f"    Done in {train_time:.1f}s. Loss: {first_loss:.4f} -> {final_loss:.4f}")

    return {
        "train_loss_first50": round(first_loss, 4),
        "train_loss_final50": round(final_loss, 4),
        "train_time_s": round(train_time, 1),
        "seed": seed,
    }


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, val_data, max_batches=VAL_BATCHES):
    total_loss = 0.0
    total_tokens = 0

    for item in val_data[:max_batches]:
        tokens = tokenizer.encode(item["text"])
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += y.size

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# KR-Test (cross-item contrastive)
# ===========================================================================
def generate_contrastive_pairs(val_data, domain, n_pairs=N_CONTRASTIVE_PAIRS):
    """Generate cross-item contrastive pairs from validation data.

    For text-only data (no instruction/response split), we use prefix/continuation
    splitting: the first half of text A is the context, the second half of text A
    is 'correct', and the second half of text B is 'wrong'.
    """
    random.seed(42)
    valid_items = [item for item in val_data if len(item["text"]) >= 60]
    if len(valid_items) < 2:
        return []

    pairs = []
    n = len(valid_items)
    for idx_a in range(min(n, n_pairs)):
        item_a = valid_items[idx_a]
        idx_b = (idx_a + max(1, n // 3)) % n
        item_b = valid_items[idx_b]

        text_a = item_a["text"]
        text_b = item_b["text"]

        # Split at ~40% for context, rest for continuation
        split_a = len(text_a) * 2 // 5
        split_b = len(text_b) * 2 // 5

        context = text_a[:split_a]
        correct = text_a[split_a:]
        wrong = text_b[split_b:]

        if correct == wrong:
            continue

        pairs.append({
            "context": context,
            "correct": correct,
            "wrong": wrong,
            "domain": domain,
        })

    return pairs[:n_pairs]


def compute_log_probs(model, tokenizer, text, max_tokens=MAX_CONTEXT_TOKENS):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    if len(tokens) < 2:
        return []

    input_ids = mx.array([tokens])
    logits = model(input_ids)
    mx.eval(logits)

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    token_log_probs = []
    for i in range(len(tokens) - 1):
        lp = log_probs[0, i, tokens[i + 1]].item()
        token_log_probs.append(lp)
    return token_log_probs


def kr_test_score(model, tokenizer, pairs, label=""):
    """Compute KR-Test: fraction where correct gets higher log-prob than wrong."""
    n_correct = 0
    n_total = 0
    margins = []

    for pair in pairs:
        context = pair["context"]
        correct_text = context + pair["correct"]
        wrong_text = context + pair["wrong"]

        ctx_tokens = tokenizer.encode(context, add_special_tokens=False)
        ctx_len = max(0, len(ctx_tokens) - 1)

        correct_lps = compute_log_probs(model, tokenizer, correct_text)
        wrong_lps = compute_log_probs(model, tokenizer, wrong_text)

        correct_cont = correct_lps[ctx_len:]
        wrong_cont = wrong_lps[ctx_len:]

        min_len = min(len(correct_cont), len(wrong_cont))
        if min_len == 0:
            continue

        correct_sum = sum(correct_cont[:min_len])
        wrong_sum = sum(wrong_cont[:min_len])

        n_total += 1
        margins.append(correct_sum - wrong_sum)
        if correct_sum > wrong_sum:
            n_correct += 1

    score = n_correct / n_total if n_total > 0 else 0.0
    mean_margin = sum(margins) / len(margins) if margins else 0.0

    if label:
        log(f"    {label}: KR={score:.3f} ({n_correct}/{n_total}), margin={mean_margin:.2f}")

    return {"score": round(score, 4), "n_correct": n_correct, "n_total": n_total,
            "mean_margin": round(mean_margin, 4)}


# ===========================================================================
# Cross-adapter cosine similarity
# ===========================================================================
def adapter_cosine(params_a, params_b):
    """Compute |cos| between two flattened adapter parameter vectors."""
    va = mx.concatenate([v.reshape(-1).astype(mx.float32) for v in params_a.values()])
    vb = mx.concatenate([v.reshape(-1).astype(mx.float32) for v in params_b.values()])
    dot = mx.sum(va * vb)
    na = mx.sqrt(mx.sum(va * va) + 1e-8)
    nb = mx.sqrt(mx.sum(vb * vb) + 1e-8)
    cos = mx.abs(dot / (na * nb))
    mx.eval(cos)
    return round(cos.item(), 6)


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    t_global = time.time()
    log("=" * 70)
    log("Multi-Cycle Evolve Convergence Experiment")
    log("=" * 70)

    results = {
        "experiment": "bitnet_evolve_multi_cycle",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "n_cycles": N_CYCLES,
        "train_iters_per_cycle": TRAIN_ITERS,
        "domains": list(DOMAINS.keys()),
        "seeds": SEEDS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ------------------------------------------------------------------
    # Phase 0: Load model
    # ------------------------------------------------------------------
    log("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    log(f"  Loaded in {time.time()-t0:.1f}s")

    model = replace_bitlinear_with_linear(model)
    model = apply_lora(model, rank=LORA_RANK, scale=LORA_SCALE)

    # ------------------------------------------------------------------
    # Phase 1: Prepare data
    # ------------------------------------------------------------------
    log("\n[Phase 1] Preparing domain data...")
    domain_data = {}
    for name, config in DOMAINS.items():
        train_data, val_data = prepare_domain_data(name, config)
        domain_data[name] = {"train": train_data, "val": val_data}

    # ------------------------------------------------------------------
    # Phase 2: Generate contrastive pairs for KR-Test
    # ------------------------------------------------------------------
    log("\n[Phase 2] Generating contrastive pairs for KR-Test...")
    contrastive_pairs = {}
    for name in DOMAINS:
        pairs = generate_contrastive_pairs(domain_data[name]["val"], name)
        contrastive_pairs[name] = pairs
        log(f"  {name}: {len(pairs)} contrastive pairs")

    # ------------------------------------------------------------------
    # Phase 3: Base model evaluation
    # ------------------------------------------------------------------
    log("\n[Phase 3] Base model evaluation (zero LoRA)...")
    zero_lora(model)
    base_metrics = {}
    for name in DOMAINS:
        ppl = compute_ppl(model, tokenizer, domain_data[name]["val"])
        kr = kr_test_score(model, tokenizer, contrastive_pairs[name], f"base/{name}")
        base_metrics[name] = {"ppl": round(ppl, 4), "kr": kr["score"]}
        log(f"  {name}: base PPL={ppl:.2f}, KR={kr['score']:.3f}")
    results["base_metrics"] = base_metrics

    # ------------------------------------------------------------------
    # Phase 4: Multi-cycle training loop
    # ------------------------------------------------------------------
    log("\n[Phase 4] Multi-cycle training...")

    # Store results per domain per cycle
    cycle_results = {name: [] for name in DOMAINS}
    best_adapter_paths = {name: None for name in DOMAINS}  # path to best adapter on disk
    best_ppls = {name: float("inf") for name in DOMAINS}

    for cycle in range(N_CYCLES):
        seed = SEEDS[cycle]
        log(f"\n{'='*50}")
        log(f"CYCLE {cycle+1}/{N_CYCLES} (seed={seed})")
        log(f"{'='*50}")

        for name in DOMAINS:
            log(f"\n  --- {name} domain, cycle {cycle+1} ---")

            # Train from fresh init
            train_info = train_adapter(
                model, tokenizer, domain_data[name]["train"],
                n_iters=TRAIN_ITERS, seed=seed + hash(name) % 1000
            )

            # Save adapter to disk (never accumulate in memory)
            adapter_path = ADAPTERS_DIR / name / f"cycle_{cycle+1}"
            save_adapter(model, adapter_path)

            # Evaluate PPL
            ppl = compute_ppl(model, tokenizer, domain_data[name]["val"])
            log(f"    PPL = {ppl:.4f}")

            # Evaluate KR-Test
            kr = kr_test_score(
                model, tokenizer, contrastive_pairs[name],
                f"{name}/cycle{cycle+1}"
            )

            # Compute cross-domain cosine with OTHER domain's best adapter (load from disk)
            cosines = {}
            current_params = load_adapter(adapter_path)
            for other_name in DOMAINS:
                if other_name == name:
                    continue
                if best_adapter_paths[other_name] is not None:
                    other_params = load_adapter(best_adapter_paths[other_name])
                    cos = adapter_cosine(current_params, other_params)
                    cosines[other_name] = cos
                    log(f"    |cos({name}, {other_name})| = {cos:.6f}")
                    del other_params
            del current_params
            gc.collect()
            mx.clear_cache()

            # Quality gate
            ppl_improved = ppl < best_ppls[name]
            kr_ok = kr["score"] >= base_metrics[name]["kr"]
            cos_ok = all(c < 0.05 for c in cosines.values()) if cosines else True
            gate_pass = ppl_improved and kr_ok and cos_ok

            cycle_info = {
                "cycle": cycle + 1,
                "seed": seed,
                "ppl": round(ppl, 4),
                "kr_score": kr["score"],
                "kr_margin": kr["mean_margin"],
                "cosines": cosines,
                "train_loss_final": train_info["train_loss_final50"],
                "train_time_s": train_info["train_time_s"],
                "gate_pass": gate_pass,
                "ppl_improved": ppl_improved,
                "kr_ok": kr_ok,
                "cos_ok": cos_ok,
            }

            if gate_pass:
                best_ppls[name] = ppl
                best_adapter_paths[name] = adapter_path
                cycle_info["accepted"] = True
                log(f"    GATE: PASS (PPL improved, KR ok, cos ok) -> accepted")
            elif cycle == 0:
                # First cycle always accepted as baseline
                best_ppls[name] = ppl
                best_adapter_paths[name] = adapter_path
                cycle_info["accepted"] = True
                log(f"    GATE: first cycle -> accepted as baseline")
            else:
                cycle_info["accepted"] = False
                log(f"    GATE: REJECTED (ppl_improved={ppl_improved}, kr_ok={kr_ok}, cos_ok={cos_ok})")

            # Cleanup between domains
            log_memory(f"post-{name}-cycle{cycle+1}")

            cycle_results[name].append(cycle_info)

    results["cycle_results"] = cycle_results

    # ------------------------------------------------------------------
    # Phase 5: Composition safety check (all cross-domain pairs, all cycles)
    # ------------------------------------------------------------------
    log("\n[Phase 5] Final composition safety check...")
    max_cos_any = 0.0
    cos_details = []

    for cycle in range(N_CYCLES):
        for name_a in DOMAINS:
            for name_b in DOMAINS:
                if name_a >= name_b:
                    continue
                path_a = ADAPTERS_DIR / name_a / f"cycle_{cycle+1}"
                path_b = ADAPTERS_DIR / name_b / f"cycle_{cycle+1}"
                if not (path_a / "adapter.npz").exists() or not (path_b / "adapter.npz").exists():
                    continue
                params_a = load_adapter(path_a)
                params_b = load_adapter(path_b)
                cos = adapter_cosine(params_a, params_b)
                del params_a, params_b
                gc.collect()
                mx.clear_cache()
                max_cos_any = max(max_cos_any, cos)
                cos_details.append({
                    "pair": f"{name_a}-{name_b}",
                    "cycle": cycle + 1,
                    "abs_cos": cos,
                })
                log(f"  cycle {cycle+1}: |cos({name_a}, {name_b})| = {cos:.6f}")

    results["composition_safety"] = {
        "max_abs_cos": max_cos_any,
        "details": cos_details,
    }

    # ------------------------------------------------------------------
    # Phase 6: Kill criteria evaluation
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("KILL CRITERIA EVALUATION")
    log("=" * 70)

    # K1: PPL must improve (cycle 3 < cycle 1) for ALL domains
    k1_pass = True
    for name in DOMAINS:
        ppl_c1 = cycle_results[name][0]["ppl"]
        ppl_c3 = cycle_results[name][-1]["ppl"]
        improved = ppl_c3 < ppl_c1
        if not improved:
            k1_pass = False
        log(f"  K1 {name}: PPL cycle1={ppl_c1:.4f}, cycle3={ppl_c3:.4f}, "
            f"improved={improved}")

    # Also check best-of-3 (gated) PPL
    for name in DOMAINS:
        ppls = [c["ppl"] for c in cycle_results[name]]
        best_ppl = min(ppls)
        log(f"  K1 {name} (gated best-of-3): best PPL = {best_ppl:.4f} "
            f"(cycle {ppls.index(best_ppl)+1})")

    # K2: KR-Test must not regress (cycle 3 >= cycle 1 - 0.02)
    k2_pass = True
    for name in DOMAINS:
        kr_c1 = cycle_results[name][0]["kr_score"]
        kr_c3 = cycle_results[name][-1]["kr_score"]
        ok = kr_c3 >= kr_c1 - 0.02
        if not ok:
            k2_pass = False
        log(f"  K2 {name}: KR cycle1={kr_c1:.4f}, cycle3={kr_c3:.4f}, "
            f"non-regressed={ok}")

    # K3: Composition safety (max |cos| < 0.05)
    k3_pass = max_cos_any < 0.05
    log(f"  K3: max |cos| = {max_cos_any:.6f}, safe={k3_pass}")

    results["kill_criteria"] = {
        "k1_ppl_improves": k1_pass,
        "k2_kr_non_regression": k2_pass,
        "k3_composition_safe": k3_pass,
    }

    # ------------------------------------------------------------------
    # Phase 7: Summary
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    for name in DOMAINS:
        log(f"\n  {name} domain:")
        for c in cycle_results[name]:
            status = "ACCEPTED" if c["accepted"] else "REJECTED"
            log(f"    Cycle {c['cycle']}: PPL={c['ppl']:.4f}, KR={c['kr_score']:.4f}, "
                f"loss={c['train_loss_final']:.4f}, [{status}]")
        log(f"    Base PPL: {base_metrics[name]['ppl']:.4f}")
        log(f"    Best gated PPL: {best_ppls[name]:.4f}")

    all_pass = k1_pass and k2_pass and k3_pass
    verdict = "SUPPORTED" if all_pass else "KILLED"

    # Determine kill reason if killed
    kill_reasons = []
    if not k1_pass:
        kill_reasons.append("K1: PPL did not improve over cycles")
    if not k2_pass:
        kill_reasons.append("K2: KR-Test regressed")
    if not k3_pass:
        kill_reasons.append("K3: Composition safety violated")

    results["verdict"] = verdict
    results["kill_reasons"] = kill_reasons
    results["total_time_s"] = round(time.time() - t_global, 1)

    log(f"\n  K1 (PPL improves): {'PASS' if k1_pass else 'FAIL'}")
    log(f"  K2 (KR non-regression): {'PASS' if k2_pass else 'FAIL'}")
    log(f"  K3 (composition safe): {'PASS' if k3_pass else 'FAIL'}")
    log(f"\n  VERDICT: {verdict}")
    if kill_reasons:
        for r in kill_reasons:
            log(f"    -> {r}")
    log(f"\n  Total time: {results['total_time_s']:.0f}s")

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
