#!/usr/bin/env python3
"""
Clone-Compete Powered Tournament (N=200, 3-arm with cold-start control)

Tests whether clone-compete evolution produces statistically significant
improvement at N=200, and whether the warm-start advantage is real (not
just "more data helps").

3-arm tournament:
  Arm A: original legal adapter (baseline)
  Arm B: clone-v2 (warm-started from original, trained on legalbench)
  Arm C: cold-start control (fresh LoRA trained from scratch on SAME legalbench data)

Kill criteria:
  K1: clone win rate <55% at N=200 with p>0.05 (improvement is noise)
  K2: tournament p-value >0.05 at N=200 (underpowered even at larger N)
  K3: cold-start control matches clone performance (warm-start advantage is illusory)

Protocol:
  Phase 0: Load BitNet-2B-4T, unpack, apply LoRA
  Phase 1: Load existing adapters (original 5 + clone_v2 from prior experiment)
  Phase 2: Train cold-start control from scratch on same legalbench data
  Phase 3: Download 200+ held-out legal texts for tournament
  Phase 4: 3-arm tournament with proper statistics
  Phase 5: Composition quality check (replace original with clone/cold-start)
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
LEARNING_RATE = 1e-4
VAL_BATCHES = 50

# Cold-start training (match clone training budget exactly)
COLD_START_TRAIN_STEPS = 400  # original (200) + clone_v1 (200) = 400 total warm-start steps
COLD_START_DATA_SAMPLES = 500  # same data as clone

# Tournament
TARGET_TOURNAMENT_SAMPLES = 250  # target 250, need 200+ decisive
TOURNAMENT_MIN_DECISIVE = 200   # minimum decisive samples for valid test

# Paths
EXPERIMENT_DIR = Path(__file__).parent
PRIOR_EXPERIMENT_DIR = Path(__file__).parent.parent / "bitnet_clone_compete"
SOURCE_ADAPTERS_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "adapters"
SOURCE_DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"
DATA_DIR = EXPERIMENT_DIR / "data"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

DOMAINS = ["python", "math", "medical", "legal", "creative"]
TARGET_DOMAIN = "legal"


# ===========================================================================
# Model utilities (reused from bitnet_clone_compete)
# ===========================================================================
def unpack_ternary(packed_weights, out_features, weight_scale, invert_scale):
    """Unpack uint8-packed ternary weights to bfloat16 dense matrix."""
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
    """Replace all BitLinear layers with standard nn.Linear."""
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
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


def apply_lora_to_model(model, rank=16, scale=1.0):
    """Apply LoRA wrappers to all linear layers in transformer blocks."""
    target_keys = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    count = 0
    for layer in model.model.layers:
        lora_updates = []
        for key, module in layer.named_modules():
            if key in target_keys and isinstance(module, nn.Linear):
                lora_layer = LoRALinear.from_base(module, r=rank, scale=scale, dropout=0.0)
                lora_updates.append((key, lora_layer))
                count += 1
        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))
    print(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def get_lora_params(model):
    """Extract LoRA parameters as a flat dict."""
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def zero_lora_params(model):
    """Reset all LoRA params to zero (lora_b=0 means no adapter effect)."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                r = module.lora_a.shape[1]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    """Apply adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def load_adapter(path: Path) -> dict:
    """Load adapter weights from disk."""
    return dict(mx.load(str(path / "adapter.npz")))


def save_adapter(params: dict, path: Path):
    """Save adapter weights to disk."""
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)
    print(f"  Saved adapter: {len(params)} tensors to {path}")


# ===========================================================================
# Data
# ===========================================================================
def load_val_texts(domain: str):
    """Load validation texts for a domain from source experiment."""
    val_path = SOURCE_DATA_DIR / domain / "valid.jsonl"
    if not val_path.exists():
        raise FileNotFoundError(f"No validation data: {val_path}")
    texts = []
    with open(val_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def prepare_tournament_data():
    """Download 250+ held-out legal texts for powered tournament.

    Uses multiple sources, ensuring NO overlap with:
    - Original legal training data (first 500 from law-stack-exchange)
    - Clone training data (legalbench contract_nli tasks)
    - Prior tournament data (law-stack-exchange samples 600-638)

    Strategy:
    - law-stack-exchange samples 500+ (skip training data)
    - lex_glue ecthr_a test set (European Court of Human Rights cases)
    """
    data_dir = DATA_DIR / "tournament"
    tourn_path = data_dir / "tournament.jsonl"

    if tourn_path.exists():
        n_existing = sum(1 for _ in open(tourn_path))
        if n_existing >= TARGET_TOURNAMENT_SAMPLES:
            print(f"  Tournament data already exists ({n_existing} samples)")
            return data_dir
        print(f"  Tournament data too small ({n_existing}), regenerating...")

    data_dir.mkdir(parents=True, exist_ok=True)
    print("  Preparing tournament data (target: 250+ samples)...")

    from datasets import load_dataset as hf_load

    texts = []

    # Source 1: law-stack-exchange, skip first 500 (used for original training)
    try:
        ds = hf_load("jonathanli/law-stack-exchange")
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        count = 0
        for row in split:
            count += 1
            if count <= 500:
                continue
            text = row.get("body", "")
            if isinstance(text, str) and len(text.strip()) > 100:
                texts.append(text.strip()[:2000])
        print(f"    law-stack-exchange (skip 500): {len(texts)} samples")
    except Exception as e:
        print(f"    law-stack-exchange failed: {e}")

    # Source 2: lex_glue ecthr_a test set (European Court of Human Rights)
    # These are legal case texts, completely disjoint from training data
    try:
        ds = hf_load("coastalcph/lex_glue", "ecthr_a", split="test")
        added = 0
        for row in ds:
            text_parts = row.get("text", [])
            if isinstance(text_parts, list):
                text = " ".join(text_parts)
            else:
                text = str(text_parts)
            if len(text.strip()) > 200:
                # Take first 2000 chars to keep PPL computation tractable
                texts.append(text.strip()[:2000])
                added += 1
            if len(texts) >= 500:
                break
        print(f"    lex_glue ecthr_a: {added} samples")
    except Exception as e:
        print(f"    lex_glue failed: {e}")

    # Source 3: lex_glue ecthr_a validation set if still need more
    if len(texts) < TARGET_TOURNAMENT_SAMPLES:
        try:
            ds = hf_load("coastalcph/lex_glue", "ecthr_a", split="validation")
            added = 0
            for row in ds:
                text_parts = row.get("text", [])
                if isinstance(text_parts, list):
                    text = " ".join(text_parts)
                else:
                    text = str(text_parts)
                if len(text.strip()) > 200:
                    texts.append(text.strip()[:2000])
                    added += 1
                if len(texts) >= 500:
                    break
            print(f"    lex_glue ecthr_a val: {added} samples")
        except Exception as e:
            print(f"    lex_glue val failed: {e}")

    # Deduplicate and shuffle with fixed seed
    import random
    random.seed(77)  # different from all prior seeds (42, 99)
    texts = list(set(texts))
    random.shuffle(texts)

    # Take target amount
    texts = texts[:max(TARGET_TOURNAMENT_SAMPLES, len(texts))]

    with open(tourn_path, "w") as f:
        for t in texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  Tournament: {len(texts)} held-out samples saved")
    return data_dir


def load_clone_training_data():
    """Load the SAME legalbench data used by clone training (for cold-start control)."""
    train_path = PRIOR_EXPERIMENT_DIR / "data" / "legalbench" / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Clone training data not found: {train_path}")
    texts = []
    with open(train_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    print(f"  Loaded {len(texts)} clone training texts (for cold-start replication)")
    return texts


# ===========================================================================
# PPL computation
# ===========================================================================
def compute_ppl(model, tokenizer, texts, max_batches=50):
    """Compute average perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0
    for text in texts[:max_batches]:
        tokens = tokenizer.encode(text)
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
    return math.exp(min(total_loss / total_tokens, 100))


def compute_per_sample_ppl(model, tokenizer, texts):
    """Compute per-sample perplexity for tournament."""
    ppls = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            ppls.append(None)  # mark as unusable
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]
        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]
        logits = model(x)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        ppl = math.exp(min(loss.item(), 100))
        ppls.append(ppl)
    return ppls


# ===========================================================================
# Statistical tests
# ===========================================================================
def compute_tournament_stats(ppls_a, ppls_b, label_a="A", label_b="B"):
    """Compute full tournament statistics between two adapters.

    Returns dict with: win rates, binomial test, Wilcoxon signed-rank,
    Cohen's d, 95% CIs for win rate and effect size.
    """
    from scipy import stats
    import numpy as np

    # Pair up valid samples
    diffs = []  # positive = B is better (lower PPL)
    a_wins = 0
    b_wins = 0
    ties = 0

    for pa, pb in zip(ppls_a, ppls_b):
        if pa is None or pb is None:
            continue
        diff = pa - pb  # positive means B better
        diffs.append(diff)
        if abs(diff) < 1e-6:
            ties += 1
        elif pb < pa:
            b_wins += 1
        else:
            a_wins += 1

    n_total = len(diffs)
    n_decisive = a_wins + b_wins
    diffs_arr = np.array(diffs)

    # Binomial test (is B's win rate significantly > 0.5?)
    if n_decisive > 0:
        win_rate_b = b_wins / n_decisive
        binom_result = stats.binomtest(b_wins, n_decisive, 0.5, alternative='two-sided')
        binom_p = binom_result.pvalue
        # 95% CI for win rate (Wilson score interval via binomtest)
        try:
            ci = binom_result.proportion_ci(confidence_level=0.95)
            win_rate_ci = (ci.low if hasattr(ci, 'low') else ci[0],
                          ci.high if hasattr(ci, 'high') else ci[1])
        except Exception:
            # Fallback: normal approximation CI
            se = math.sqrt(win_rate_b * (1 - win_rate_b) / n_decisive)
            win_rate_ci = (win_rate_b - 1.96 * se, win_rate_b + 1.96 * se)
    else:
        win_rate_b = 0.5
        binom_p = 1.0
        win_rate_ci = (0.0, 1.0)

    # Wilcoxon signed-rank test (non-parametric paired test on PPL differences)
    if n_decisive >= 10:
        # Remove ties for Wilcoxon
        nonzero_diffs = diffs_arr[np.abs(diffs_arr) > 1e-6]
        if len(nonzero_diffs) >= 10:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nonzero_diffs, alternative='two-sided')
        else:
            wilcoxon_stat, wilcoxon_p = float('nan'), 1.0
    else:
        wilcoxon_stat, wilcoxon_p = float('nan'), 1.0

    # Paired t-test
    if n_total >= 10:
        t_stat, t_p = stats.ttest_1samp(diffs_arr, 0.0)
    else:
        t_stat, t_p = float('nan'), 1.0

    # Cohen's d (effect size: mean diff / pooled SD)
    if n_total > 1 and np.std(diffs_arr) > 0:
        cohens_d = np.mean(diffs_arr) / np.std(diffs_arr, ddof=1)
    else:
        cohens_d = 0.0

    # 95% CI for mean PPL difference
    if n_total >= 2:
        se = np.std(diffs_arr, ddof=1) / np.sqrt(n_total)
        t_crit = stats.t.ppf(0.975, df=n_total - 1)
        mean_diff = np.mean(diffs_arr)
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
    else:
        mean_diff = 0.0
        ci_low = ci_high = 0.0

    return {
        "label_a": label_a,
        "label_b": label_b,
        "n_total": n_total,
        "n_decisive": n_decisive,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "win_rate_b": round(float(win_rate_b), 4),
        "win_rate_ci_95": [round(float(win_rate_ci[0]), 4), round(float(win_rate_ci[1]), 4)],
        "binom_p": round(float(binom_p), 6),
        "binom_significant": bool(binom_p < 0.05),
        "wilcoxon_stat": round(float(wilcoxon_stat), 2) if not math.isnan(wilcoxon_stat) else None,
        "wilcoxon_p": round(float(wilcoxon_p), 6) if not math.isnan(wilcoxon_p) else None,
        "t_stat": round(float(t_stat), 4) if not math.isnan(t_stat) else None,
        "t_p": round(float(t_p), 6) if not math.isnan(t_p) else None,
        "cohens_d": round(float(cohens_d), 4),
        "mean_ppl_diff": round(float(mean_diff), 4),
        "ppl_diff_ci_95": [round(float(ci_low), 4), round(float(ci_high), 4)],
        "aggregate_ppl_a": round(float(np.mean([p for p in ppls_a if p is not None])), 4),
        "aggregate_ppl_b": round(float(np.mean([p for p in ppls_b if p is not None])), 4),
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_clone_compete_powered",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "target_domain": TARGET_DOMAIN,
        "cold_start_train_steps": COLD_START_TRAIN_STEPS,
        "tournament_target_n": TARGET_TOURNAMENT_SAMPLES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t0 = time.time()
    print("=" * 70)
    print("Clone-Compete POWERED Tournament (N=200, 3-arm)")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Load model
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t_phase = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t_phase:.1f}s")

    print("  Unpacking ternary weights...")
    t_phase = time.time()
    model = replace_bitlinear_with_linear(model)
    print(f"  Unpacked in {time.time() - t_phase:.1f}s")

    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA params: {trainable:,}")

    # ==================================================================
    # Phase 1: Load existing adapters
    # ==================================================================
    print("\n[Phase 1] Loading existing adapters...")

    # Load all 5 domain adapters from original experiment
    adapters = {}
    for domain in DOMAINS:
        adapter_path = SOURCE_ADAPTERS_DIR / domain
        if not adapter_path.exists():
            print(f"  ERROR: {adapter_path} does not exist!")
            results["error"] = f"Missing adapter: {domain}"
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)
            return
        adapters[domain] = load_adapter(adapter_path)
        print(f"  {domain}: loaded ({len(adapters[domain])} tensors)")

    # Load clone_v2 from prior experiment
    clone_v2_path = PRIOR_EXPERIMENT_DIR / "adapters" / "legal_clone_v2"
    if not clone_v2_path.exists():
        print(f"  ERROR: clone_v2 not found at {clone_v2_path}")
        results["error"] = "Missing clone_v2 adapter"
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        return
    clone_v2_adapter = load_adapter(clone_v2_path)
    print(f"  clone_v2: loaded ({len(clone_v2_adapter)} tensors)")

    # ==================================================================
    # Phase 2: Train cold-start control
    # ==================================================================
    print("\n[Phase 2] Training cold-start control adapter...")
    print("  Key: fresh LoRA from scratch on SAME data as clone, NOT warm-started")

    # Load same training data clone was trained on
    clone_train_texts = load_clone_training_data()

    # Tokenize
    train_tokens = []
    for text in clone_train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))
    print(f"  Tokenized {len(train_tokens)} training sequences")

    # Initialize fresh LoRA (zero out = no adapter signal)
    zero_lora_params(model)
    mx.eval(model.parameters())

    # Verify cold start: should match base PPL
    legal_val_texts = load_val_texts(TARGET_DOMAIN)
    cold_start_pre_ppl = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Cold-start pre-training PPL: {cold_start_pre_ppl:.2f} (should match base)")

    # Train cold-start for SAME total steps as clone's cumulative training
    # Clone v2 = original (200 steps) + clone_v1 (200 steps from original) + clone_v2 (200 more)
    # But the fair comparison is: clone inherits 200 steps of warm-start + gets 200 more = 400 total
    # Cold-start gets 400 steps from scratch on the same data
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_train = time.time()
    losses = []
    for step in range(COLD_START_TRAIN_STEPS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())
        if (step + 1) % 100 == 0 or step == 0:
            avg = sum(losses[-50:]) / min(len(losses), 50)
            print(f"    Step {step+1}/{COLD_START_TRAIN_STEPS}: loss={loss.item():.4f} (avg50={avg:.4f})")

    cold_train_time = time.time() - t_train
    cold_start_adapter = get_lora_params(model)
    save_adapter(cold_start_adapter, ADAPTERS_DIR / "legal_cold_start")

    # Evaluate cold-start
    cold_start_ppl = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Cold-start post-training PPL: {cold_start_ppl:.2f}")
    print(f"  Training time: {cold_train_time:.1f}s")

    results["cold_start"] = {
        "train_steps": COLD_START_TRAIN_STEPS,
        "train_time_s": round(cold_train_time, 1),
        "pre_train_ppl": round(cold_start_pre_ppl, 4),
        "post_train_ppl": round(cold_start_ppl, 4),
        "first_50_loss": round(sum(losses[:50]) / 50, 4),
        "last_50_loss": round(sum(losses[-50:]) / 50, 4),
    }

    # ==================================================================
    # Phase 2b: Evaluate all three arms on legal val data
    # ==================================================================
    print("\n[Phase 2b] Evaluating all arms on legal validation data...")

    # Base PPL (no adapter)
    zero_lora_params(model)
    mx.eval(model.parameters())
    base_legal_ppl = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Base (no adapter): PPL={base_legal_ppl:.2f}")

    # Original legal adapter
    zero_lora_params(model)
    apply_adapter_weights(model, adapters[TARGET_DOMAIN])
    mx.eval(model.parameters())
    original_legal_ppl = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Original adapter:  PPL={original_legal_ppl:.2f} ({(base_legal_ppl - original_legal_ppl) / base_legal_ppl * 100:+.1f}% vs base)")

    # Clone v2
    zero_lora_params(model)
    apply_adapter_weights(model, clone_v2_adapter)
    mx.eval(model.parameters())
    clone_v2_ppl = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Clone v2:          PPL={clone_v2_ppl:.2f} ({(original_legal_ppl - clone_v2_ppl) / original_legal_ppl * 100:+.1f}% vs original)")

    # Cold-start (already evaluated above but confirm with fresh apply)
    zero_lora_params(model)
    apply_adapter_weights(model, cold_start_adapter)
    mx.eval(model.parameters())
    cold_start_ppl_verify = compute_ppl(model, tokenizer, legal_val_texts)
    print(f"  Cold-start:        PPL={cold_start_ppl_verify:.2f} ({(original_legal_ppl - cold_start_ppl_verify) / original_legal_ppl * 100:+.1f}% vs original)")

    results["arm_ppls"] = {
        "base": round(base_legal_ppl, 4),
        "original": round(original_legal_ppl, 4),
        "clone_v2": round(clone_v2_ppl, 4),
        "cold_start": round(cold_start_ppl_verify, 4),
    }

    # ==================================================================
    # Phase 3: Prepare tournament data
    # ==================================================================
    print("\n[Phase 3] Preparing tournament data...")
    tournament_data_dir = prepare_tournament_data()
    tournament_texts = []
    with open(tournament_data_dir / "tournament.jsonl") as f:
        for line in f:
            tournament_texts.append(json.loads(line)["text"])
    print(f"  Tournament samples loaded: {len(tournament_texts)}")

    if len(tournament_texts) < TOURNAMENT_MIN_DECISIVE:
        print(f"  WARNING: Only {len(tournament_texts)} samples, target was {TARGET_TOURNAMENT_SAMPLES}")
        print(f"  Proceeding with available samples...")

    results["tournament_n_available"] = len(tournament_texts)

    # ==================================================================
    # Phase 4: 3-arm tournament
    # ==================================================================
    print("\n[Phase 4] Running 3-arm tournament...")

    # Compute per-sample PPL for all three arms
    print("  Computing per-sample PPL for original...")
    t_phase = time.time()
    zero_lora_params(model)
    apply_adapter_weights(model, adapters[TARGET_DOMAIN])
    mx.eval(model.parameters())
    ppls_original = compute_per_sample_ppl(model, tokenizer, tournament_texts)
    print(f"    {sum(1 for p in ppls_original if p is not None)} valid samples ({time.time()-t_phase:.1f}s)")

    print("  Computing per-sample PPL for clone_v2...")
    t_phase = time.time()
    zero_lora_params(model)
    apply_adapter_weights(model, clone_v2_adapter)
    mx.eval(model.parameters())
    ppls_clone = compute_per_sample_ppl(model, tokenizer, tournament_texts)
    print(f"    {sum(1 for p in ppls_clone if p is not None)} valid samples ({time.time()-t_phase:.1f}s)")

    print("  Computing per-sample PPL for cold-start...")
    t_phase = time.time()
    zero_lora_params(model)
    apply_adapter_weights(model, cold_start_adapter)
    mx.eval(model.parameters())
    ppls_cold = compute_per_sample_ppl(model, tokenizer, tournament_texts)
    print(f"    {sum(1 for p in ppls_cold if p is not None)} valid samples ({time.time()-t_phase:.1f}s)")

    # Tournament 1: Clone v2 vs Original
    print("\n  --- Tournament: Clone v2 vs Original ---")
    t1_stats = compute_tournament_stats(ppls_original, ppls_clone, "original", "clone_v2")
    print(f"    Clone win rate: {t1_stats['win_rate_b']:.1%} ({t1_stats['b_wins']}/{t1_stats['n_decisive']})")
    print(f"    Win rate 95% CI: [{t1_stats['win_rate_ci_95'][0]:.1%}, {t1_stats['win_rate_ci_95'][1]:.1%}]")
    print(f"    Binomial p: {t1_stats['binom_p']:.4f} {'***' if t1_stats['binom_p'] < 0.001 else '**' if t1_stats['binom_p'] < 0.01 else '*' if t1_stats['binom_p'] < 0.05 else 'ns'}")
    print(f"    Wilcoxon p: {t1_stats['wilcoxon_p']}")
    print(f"    Paired t-test p: {t1_stats['t_p']}")
    print(f"    Cohen's d: {t1_stats['cohens_d']:.3f}")
    print(f"    Mean PPL diff: {t1_stats['mean_ppl_diff']:.4f} [{t1_stats['ppl_diff_ci_95'][0]:.4f}, {t1_stats['ppl_diff_ci_95'][1]:.4f}]")
    results["tournament_clone_vs_original"] = t1_stats

    # Tournament 2: Cold-start vs Original
    print("\n  --- Tournament: Cold-start vs Original ---")
    t2_stats = compute_tournament_stats(ppls_original, ppls_cold, "original", "cold_start")
    print(f"    Cold-start win rate: {t2_stats['win_rate_b']:.1%} ({t2_stats['b_wins']}/{t2_stats['n_decisive']})")
    print(f"    Win rate 95% CI: [{t2_stats['win_rate_ci_95'][0]:.1%}, {t2_stats['win_rate_ci_95'][1]:.1%}]")
    print(f"    Binomial p: {t2_stats['binom_p']:.4f} {'***' if t2_stats['binom_p'] < 0.001 else '**' if t2_stats['binom_p'] < 0.01 else '*' if t2_stats['binom_p'] < 0.05 else 'ns'}")
    print(f"    Wilcoxon p: {t2_stats['wilcoxon_p']}")
    print(f"    Cohen's d: {t2_stats['cohens_d']:.3f}")
    print(f"    Mean PPL diff: {t2_stats['mean_ppl_diff']:.4f} [{t2_stats['ppl_diff_ci_95'][0]:.4f}, {t2_stats['ppl_diff_ci_95'][1]:.4f}]")
    results["tournament_cold_vs_original"] = t2_stats

    # Tournament 3: Clone v2 vs Cold-start (THE CRITICAL COMPARISON)
    print("\n  --- Tournament: Clone v2 vs Cold-start (CRITICAL) ---")
    t3_stats = compute_tournament_stats(ppls_cold, ppls_clone, "cold_start", "clone_v2")
    print(f"    Clone v2 win rate vs cold-start: {t3_stats['win_rate_b']:.1%} ({t3_stats['b_wins']}/{t3_stats['n_decisive']})")
    print(f"    Win rate 95% CI: [{t3_stats['win_rate_ci_95'][0]:.1%}, {t3_stats['win_rate_ci_95'][1]:.1%}]")
    print(f"    Binomial p: {t3_stats['binom_p']:.4f} {'***' if t3_stats['binom_p'] < 0.001 else '**' if t3_stats['binom_p'] < 0.01 else '*' if t3_stats['binom_p'] < 0.05 else 'ns'}")
    print(f"    Wilcoxon p: {t3_stats['wilcoxon_p']}")
    print(f"    Cohen's d: {t3_stats['cohens_d']:.3f}")
    print(f"    Mean PPL diff: {t3_stats['mean_ppl_diff']:.4f} [{t3_stats['ppl_diff_ci_95'][0]:.4f}, {t3_stats['ppl_diff_ci_95'][1]:.4f}]")
    results["tournament_clone_vs_cold_start"] = t3_stats

    # ==================================================================
    # Phase 5: Composition quality check
    # ==================================================================
    print("\n[Phase 5] Composition quality check (1/N with 5 adapters)...")

    composition_results = {}
    for arm_name, arm_adapter in [
        ("original", adapters[TARGET_DOMAIN]),
        ("clone_v2", clone_v2_adapter),
        ("cold_start", cold_start_adapter),
    ]:
        # Build composed adapter set: replace legal with arm adapter
        arm_adapters = {d: adapters[d] for d in DOMAINS}
        arm_adapters[TARGET_DOMAIN] = arm_adapter

        # 1/N merge
        merged = {}
        for key in adapters[DOMAINS[0]].keys():
            stacked = mx.stack([arm_adapters[d][key] for d in DOMAINS])
            merged[key] = mx.sum(stacked, axis=0) / len(DOMAINS)

        zero_lora_params(model)
        apply_adapter_weights(model, merged)
        mx.eval(model.parameters())

        domain_ppls = {}
        for domain in DOMAINS:
            texts = load_val_texts(domain)
            ppl = compute_ppl(model, tokenizer, texts)
            domain_ppls[domain] = round(ppl, 4)

        composition_results[arm_name] = domain_ppls
        print(f"  {arm_name}: {domain_ppls}")

    results["composition_quality"] = composition_results

    # Compute regression deltas
    regression_clone = {}
    regression_cold = {}
    max_regression_clone = 0
    max_regression_cold = 0
    for domain in DOMAINS:
        orig = composition_results["original"][domain]
        clone = composition_results["clone_v2"][domain]
        cold = composition_results["cold_start"][domain]

        delta_clone = (clone - orig) / orig * 100
        delta_cold = (cold - orig) / orig * 100

        regression_clone[domain] = round(delta_clone, 3)
        regression_cold[domain] = round(delta_cold, 3)

        if domain != TARGET_DOMAIN:
            max_regression_clone = max(max_regression_clone, delta_clone)
            max_regression_cold = max(max_regression_cold, delta_cold)

    results["regression_clone_vs_original"] = regression_clone
    results["regression_cold_vs_original"] = regression_cold
    results["max_regression_clone_pct"] = round(max_regression_clone, 3)
    results["max_regression_cold_pct"] = round(max_regression_cold, 3)

    print(f"\n  Regression (clone vs original): {regression_clone}")
    print(f"  Regression (cold vs original): {regression_cold}")
    print(f"  Max non-target regression: clone={max_regression_clone:.3f}%, cold={max_regression_cold:.3f}%")

    # ==================================================================
    # Kill criteria assessment
    # ==================================================================
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: clone win rate >55% at N=200 with p<0.05
    k1_win_rate = t1_stats["win_rate_b"]
    k1_p = t1_stats["binom_p"]
    k1_pass = k1_win_rate >= 0.55 and k1_p < 0.05
    k1_kill = k1_win_rate < 0.55 and k1_p > 0.05
    results["k1_win_rate"] = k1_win_rate
    results["k1_p_value"] = k1_p
    results["k1_pass"] = k1_pass
    results["k1_kill"] = k1_kill
    print(f"  K1 (clone win rate >=55%, p<0.05): {'PASS' if k1_pass else 'KILL' if k1_kill else 'INCONCLUSIVE'}")
    print(f"      Win rate: {k1_win_rate:.1%}, p={k1_p:.4f}")

    # K2: overall tournament p-value <0.05
    # Use Wilcoxon (most appropriate nonparametric paired test)
    k2_p = t1_stats["wilcoxon_p"] if t1_stats["wilcoxon_p"] is not None else t1_stats["binom_p"]
    k2_pass = k2_p < 0.05
    results["k2_p_value"] = k2_p
    results["k2_pass"] = k2_pass
    print(f"  K2 (tournament p<0.05): {'PASS' if k2_pass else 'KILL'}")
    print(f"      Best p-value (Wilcoxon): {k2_p}")

    # K3: cold-start control does NOT match clone
    # If clone significantly beats cold-start, warm-start advantage is real
    k3_clone_vs_cold_p = t3_stats["binom_p"]
    k3_clone_vs_cold_win = t3_stats["win_rate_b"]  # clone win rate vs cold-start
    k3_cold_matches = k3_clone_vs_cold_p > 0.05 and abs(k3_clone_vs_cold_win - 0.5) < 0.10
    k3_pass = not k3_cold_matches  # PASS if cold-start does NOT match clone
    results["k3_clone_vs_cold_p"] = k3_clone_vs_cold_p
    results["k3_clone_vs_cold_win_rate"] = k3_clone_vs_cold_win
    results["k3_cold_matches_clone"] = k3_cold_matches
    results["k3_pass"] = k3_pass
    print(f"  K3 (cold-start != clone): {'PASS (warm-start advantage real)' if k3_pass else 'KILL (warm-start is illusory)'}")
    print(f"      Clone vs cold-start win rate: {k3_clone_vs_cold_win:.1%}, p={k3_clone_vs_cold_p:.4f}")

    # Also report: does cold-start also beat original?
    cold_also_beats_original = t2_stats["binom_p"] < 0.05 and t2_stats["win_rate_b"] > 0.55
    results["cold_start_also_beats_original"] = cold_also_beats_original
    print(f"\n  Additional: cold-start beats original? {'YES' if cold_also_beats_original else 'NO'} "
          f"(win rate: {t2_stats['win_rate_b']:.1%}, p={t2_stats['binom_p']:.4f})")

    # ==================================================================
    # Overall verdict
    # ==================================================================
    print("\n" + "=" * 70)
    print("OVERALL VERDICT")
    print("=" * 70)

    all_pass = k1_pass and k2_pass and k3_pass
    any_kill = k1_kill or (not k2_pass) or (not k3_pass)

    if all_pass:
        verdict = "PROVEN"
        verdict_detail = "Clone significantly beats original with real warm-start advantage"
    elif k1_pass and k2_pass and not k3_pass:
        verdict = "SUPPORTED (warm-start caveat)"
        verdict_detail = "Clone beats original, but cold-start control matches -- improvement is from more data, not inheritance"
    elif not k2_pass:
        verdict = "KILLED"
        verdict_detail = "Tournament not significant at N=200 -- improvement is noise"
    elif k1_kill:
        verdict = "KILLED"
        verdict_detail = "Clone win rate below threshold -- no per-sample advantage"
    else:
        verdict = "INCONCLUSIVE"
        verdict_detail = "Mixed results, see detailed stats"

    results["verdict"] = verdict
    results["verdict_detail"] = verdict_detail
    results["all_k_pass"] = all_pass

    print(f"  {verdict}: {verdict_detail}")
    print()
    print(f"  K1: {'PASS' if k1_pass else 'KILL' if k1_kill else 'INCONCLUSIVE'} (clone win rate {k1_win_rate:.1%}, p={k1_p:.4f})")
    print(f"  K2: {'PASS' if k2_pass else 'KILL'} (Wilcoxon p={k2_p})")
    print(f"  K3: {'PASS' if k3_pass else 'KILL'} (clone vs cold-start: {k3_clone_vs_cold_win:.1%}, p={k3_clone_vs_cold_p:.4f})")

    total_time = time.time() - t0
    results["total_time_s"] = round(total_time, 1)
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (bool,)):
            return bool(obj)
        elif hasattr(obj, 'item'):
            return obj.item()
        elif type(obj).__name__ == 'bool_':
            return bool(obj)
        return obj

    results = make_serializable(results)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
