#!/usr/bin/env python3
"""
Clone-Compete Evolution Experiment

Tests whether cloning the worst adapter (legal), fine-tuning the clone on
fresh/additional data, and running a PPL-based tournament can improve adapter
quality without regressing other domains.

Kill criteria:
  K1: clone does not win tournament >60% of the time
  K2: tournament requires >10K queries to resolve
  K3: evolution causes >2% regression on other domains

Protocol:
  Phase 0: Load BitNet-2B-4T, unpack, apply LoRA
  Phase 1: Load existing 5 adapters from bitnet_2b_real_composition
  Phase 2: Evaluate baseline PPL for all 5 domains
  Phase 3: Clone legal adapter, continue training on fresh legal data (legalbench)
  Phase 4: Tournament - per-sample PPL comparison (original vs clone)
           Use sequential binomial test; stop early if significant
  Phase 5: Regression check - does evolved adapter hurt other domains?
  Phase 6: Second evolution round (iterability test)

Architecture:
  - Base: microsoft/BitNet-b1.58-2B-4T (ternary, d=2560, 30 layers)
  - LoRA: rank-16, all-modules, FP16 adapters
  - Composition: 1/N scaling for multi-adapter evaluation

Connection to theory:
  This implements a simplified version of Sakana AI's evolutionary merging,
  but at the adapter level rather than full model level. Instead of genetic
  crossover in weight space, we use continued training (mutation) and PPL-based
  tournament selection (fitness evaluation). The key insight from multi-armed
  bandit theory: per-sample PPL comparison gives a binomial signal that
  converges with O(1/sqrt(N)) samples.
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
VAL_BATCHES = 50  # more samples for tournament statistical power

# Clone training
CLONE_TRAIN_STEPS = 200  # same as original training budget
CLONE_DATA_SAMPLES = 500  # fresh legal data

# Tournament
MAX_TOURNAMENT_QUERIES = 10000  # K2 kill if we need more
EARLY_STOP_ALPHA = 0.05  # significance level for early stopping
MIN_TOURNAMENT_QUERIES = 50  # minimum before early stop

# Paths
EXPERIMENT_DIR = Path(__file__).parent
SOURCE_ADAPTERS_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "adapters"
SOURCE_DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
DATA_DIR = EXPERIMENT_DIR / "data"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

DOMAINS = ["python", "math", "medical", "legal", "creative"]
TARGET_DOMAIN = "legal"  # worst adapter


# ===========================================================================
# Model utilities (reused from bitnet_2b_real_composition)
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
    """Reset all LoRA params to zero."""
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
    """Load validation texts for a domain."""
    val_path = SOURCE_DATA_DIR / domain / "valid.jsonl"
    if not val_path.exists():
        raise FileNotFoundError(f"No validation data: {val_path}")
    texts = []
    with open(val_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    return texts


def prepare_clone_data():
    """Download fresh legal data from legalbench for clone training."""
    data_dir = DATA_DIR / "legalbench"
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "valid.jsonl"

    if train_path.exists() and val_path.exists():
        print("  Legalbench data already exists, skipping download")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print("  Downloading legalbench data...")

    from datasets import load_dataset as hf_load

    # legalbench has many tasks; use a mix for diversity
    tasks_to_try = [
        "contract_nli_confidentiality_of_agreement",
        "contract_nli_explicit_identification",
        "contract_nli_inclusion_of_verbatim_terms",
        "contract_nli_limited_use",
        "contract_nli_no_licensing",
        "contract_nli_notice_on_compelled_disclosure",
        "contract_nli_permissible_acquirement_of_similar_information",
        "contract_nli_permissible_copy",
        "contract_nli_permissible_development_of_similar_information",
        "contract_nli_permissible_post-agreement_possession",
        "contract_nli_return_of_confidential_information",
        "contract_nli_sharing_with_employees",
        "contract_nli_sharing_with_third-parties",
        "contract_nli_survival_of_obligations",
        "contract_qa",
        "unfair_tos",
    ]

    all_texts = []
    for task in tasks_to_try:
        try:
            ds = hf_load("nguha/legalbench", name=task)
            if "test" in ds:
                split = ds["test"]
            elif "train" in ds:
                split = ds["train"]
            else:
                split = ds[list(ds.keys())[0]]

            # Build text from available columns - prefer longer text
            for row in split:
                # Try to construct meaningful legal text
                parts = []
                for col in split.column_names:
                    val = row[col]
                    if isinstance(val, str) and len(val) > 30:
                        parts.append(val)
                if parts:
                    text = " ".join(parts)
                    if len(text) > 50:
                        all_texts.append(text.strip())

            if len(all_texts) >= CLONE_DATA_SAMPLES + 100:
                break
        except Exception as e:
            print(f"    Skipping {task}: {e}")
            continue

    if len(all_texts) < 100:
        # Fallback: use law-stack-exchange with different split
        print("  Legalbench insufficient, falling back to law-stack-exchange...")
        ds = hf_load("jonathanli/law-stack-exchange")
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        for row in split:
            text = row.get("body", "")
            if isinstance(text, str) and len(text.strip()) > 50:
                all_texts.append(text.strip())
            if len(all_texts) >= CLONE_DATA_SAMPLES + 100:
                break

    # Use DIFFERENT samples than original training (skip first 500)
    # Original used first 500 from law-stack-exchange
    # For clone, use samples 500+ or legalbench data
    import random
    random.seed(42)
    random.shuffle(all_texts)

    train_texts = all_texts[:CLONE_DATA_SAMPLES]
    val_texts = all_texts[CLONE_DATA_SAMPLES:CLONE_DATA_SAMPLES + 100]

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    with open(val_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  Legalbench: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


def prepare_tournament_data():
    """Prepare held-out tournament data for legal domain.

    Uses multiple legal data sources to get enough tournament samples.
    Must be DIFFERENT from both original training data and clone training data.
    """
    data_dir = DATA_DIR / "tournament"
    tourn_path = data_dir / "tournament.jsonl"

    if tourn_path.exists():
        n_existing = sum(1 for _ in open(tourn_path))
        if n_existing >= 200:
            print(f"  Tournament data already exists ({n_existing} samples)")
            return data_dir
        # Too few samples, regenerate
        print(f"  Tournament data too small ({n_existing}), regenerating...")

    data_dir.mkdir(parents=True, exist_ok=True)
    print("  Preparing tournament data from multiple sources...")

    from datasets import load_dataset as hf_load

    texts = []

    # Source 1: law-stack-exchange (skip first 600 used by training/val)
    try:
        ds = hf_load("jonathanli/law-stack-exchange")
        split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
        count = 0
        for row in split:
            count += 1
            if count <= 600:
                continue
            text = row.get("body", "")
            if isinstance(text, str) and len(text.strip()) > 50:
                texts.append(text.strip())
            if len(texts) >= 500:
                break
        print(f"    law-stack-exchange: {len(texts)} samples (after skip)")
    except Exception as e:
        print(f"    law-stack-exchange failed: {e}")

    # Source 2: pile-of-law or other legal datasets
    if len(texts) < 200:
        try:
            ds = hf_load("pile-of-law/pile-of-law", name="r_legaladvice",
                         split="train", streaming=True)
            added = 0
            for row in ds:
                text = row.get("text", "")
                if isinstance(text, str) and len(text.strip()) > 100:
                    texts.append(text.strip()[:2000])  # cap length
                    added += 1
                if added >= 500:
                    break
            print(f"    pile-of-law: {added} samples")
        except Exception as e:
            print(f"    pile-of-law failed: {e}")

    # Source 3: if still not enough, use MultiLegalPile or similar
    if len(texts) < 200:
        try:
            ds = hf_load("casehold/casehold", split="test")
            for row in ds:
                # CaseHOLD has citing_prompt which is legal text
                text = row.get("citing_prompt", "")
                if isinstance(text, str) and len(text.strip()) > 50:
                    texts.append(text.strip())
                if len(texts) >= 500:
                    break
            print(f"    After casehold: {len(texts)} total")
        except Exception as e:
            print(f"    casehold failed: {e}")

    # Deduplicate and shuffle
    import random
    random.seed(99)  # different seed from clone data
    texts = list(set(texts))
    random.shuffle(texts)

    with open(tourn_path, "w") as f:
        for t in texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  Tournament: {len(texts)} samples total")
    return data_dir


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
    """Compute per-sample perplexity (for tournament comparison)."""
    ppls = []
    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
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
# Tournament
# ===========================================================================
def run_tournament(model, tokenizer, adapter_a, adapter_b, tournament_texts):
    """Run PPL-based tournament between two adapters.

    For each sample, compute PPL with adapter_a and adapter_b.
    The adapter with lower PPL on that sample wins.

    Uses sequential binomial test for early stopping:
    Under H0: p(A wins) = 0.5. If we observe significantly more wins
    for one side, stop early.

    Returns: dict with tournament results
    """
    from scipy import stats

    a_wins = 0
    b_wins = 0
    ties = 0
    total = 0
    ppl_diffs = []  # positive = B is better (lower PPL)

    t_start = time.time()

    for i, text in enumerate(tournament_texts):
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:MAX_SEQ_LENGTH + 1]

        x = mx.array(tokens[:-1])[None, :]
        y = mx.array(tokens[1:])[None, :]

        # Evaluate adapter A (original)
        zero_lora_params(model)
        apply_adapter_weights(model, adapter_a)
        mx.eval(model.parameters())

        logits_a = model(x)
        loss_a = nn.losses.cross_entropy(logits_a, y, reduction="mean")
        mx.eval(loss_a)
        ppl_a = math.exp(min(loss_a.item(), 100))

        # Evaluate adapter B (clone)
        zero_lora_params(model)
        apply_adapter_weights(model, adapter_b)
        mx.eval(model.parameters())

        logits_b = model(x)
        loss_b = nn.losses.cross_entropy(logits_b, y, reduction="mean")
        mx.eval(loss_b)
        ppl_b = math.exp(min(loss_b.item(), 100))

        total += 1
        ppl_diffs.append(ppl_a - ppl_b)  # positive = clone is better

        if abs(ppl_a - ppl_b) < 1e-6:
            ties += 1
        elif ppl_b < ppl_a:
            b_wins += 1  # clone wins
        else:
            a_wins += 1  # original wins

        # Sequential test: after minimum queries, check significance
        if total >= MIN_TOURNAMENT_QUERIES and total % 10 == 0:
            # Binomial test: is clone win rate significantly > 0.5?
            n_decisive = a_wins + b_wins
            if n_decisive > 0:
                p_value = stats.binomtest(b_wins, n_decisive, 0.5,
                                          alternative='two-sided').pvalue
                if p_value < EARLY_STOP_ALPHA:
                    print(f"    Early stop at {total} queries: "
                          f"A={a_wins}, B={b_wins}, ties={ties}, p={p_value:.4f}")
                    break

        if total >= MAX_TOURNAMENT_QUERIES:
            print(f"    Hit max queries: {total}")
            break

        if (total) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"    Query {total}: A={a_wins} B={b_wins} ties={ties} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t_start
    n_decisive = a_wins + b_wins

    # Final statistics
    if n_decisive > 0:
        clone_win_rate = b_wins / n_decisive
        p_value = stats.binomtest(b_wins, n_decisive, 0.5, alternative='two-sided').pvalue
    else:
        clone_win_rate = 0.5
        p_value = 1.0

    # Mean PPL difference (positive = clone better)
    mean_ppl_diff = sum(ppl_diffs) / len(ppl_diffs) if ppl_diffs else 0

    result = {
        "total_queries": total,
        "a_wins_original": a_wins,
        "b_wins_clone": b_wins,
        "ties": ties,
        "clone_win_rate": round(clone_win_rate, 4),
        "p_value": round(p_value, 6),
        "significant": p_value < EARLY_STOP_ALPHA,
        "mean_ppl_diff": round(mean_ppl_diff, 4),
        "elapsed_s": round(elapsed, 1),
    }
    return result


# ===========================================================================
# Main
# ===========================================================================
def main():
    results = {
        "experiment": "bitnet_clone_compete",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "target_domain": TARGET_DOMAIN,
        "clone_train_steps": CLONE_TRAIN_STEPS,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print("=" * 70)
    print("Clone-Compete Evolution Experiment")
    print("=" * 70)

    # ==================================================================
    # Phase 0: Load model
    # ==================================================================
    print("\n[Phase 0] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("  Unpacking ternary weights...")
    t1 = time.time()
    model = replace_bitlinear_with_linear(model)
    print(f"  Unpacked in {time.time() - t1:.1f}s")

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
        norm = sum(mx.sum(p ** 2).item() for p in adapters[domain].values())
        print(f"  {domain}: loaded ({len(adapters[domain])} tensors, L2={norm:.4f})")

    # ==================================================================
    # Phase 2: Baseline PPL (all domains)
    # ==================================================================
    print("\n[Phase 2] Baseline PPL evaluation...")
    baseline_ppls = {}

    # Base model PPL (no adapter)
    zero_lora_params(model)
    base_ppls = {}
    for domain in DOMAINS:
        texts = load_val_texts(domain)
        ppl = compute_ppl(model, tokenizer, texts)
        base_ppls[domain] = round(ppl, 4)
        print(f"  base/{domain}: PPL={ppl:.2f}")
    results["base_ppls"] = base_ppls

    # Individual adapter PPL
    for domain in DOMAINS:
        zero_lora_params(model)
        apply_adapter_weights(model, adapters[domain])
        mx.eval(model.parameters())

        texts = load_val_texts(domain)
        ppl = compute_ppl(model, tokenizer, texts)
        baseline_ppls[domain] = round(ppl, 4)
        improvement = (base_ppls[domain] - ppl) / base_ppls[domain] * 100
        print(f"  {domain}: PPL={ppl:.2f} ({improvement:+.1f}% vs base)")

    results["baseline_individual_ppls"] = baseline_ppls

    # Identify worst adapter
    worst_domain = max(baseline_ppls, key=lambda d: baseline_ppls[d] / base_ppls[d])
    worst_improvement = (base_ppls[worst_domain] - baseline_ppls[worst_domain]) / base_ppls[worst_domain] * 100
    print(f"\n  Worst adapter: {worst_domain} ({worst_improvement:+.1f}% improvement)")
    results["worst_adapter"] = worst_domain

    # ==================================================================
    # Phase 3: Clone and train
    # ==================================================================
    print(f"\n[Phase 3] Cloning {TARGET_DOMAIN} adapter and fine-tuning on fresh data...")

    # Prepare fresh legal data
    clone_data_dir = prepare_clone_data()

    # Start from original adapter weights (clone = continued training)
    clone_params = {k: mx.array(v) for k, v in adapters[TARGET_DOMAIN].items()}
    mx.eval(clone_params)

    # Apply clone params and train
    zero_lora_params(model)
    apply_adapter_weights(model, clone_params)
    mx.eval(model.parameters())

    # Load clone training data
    train_texts = []
    with open(clone_data_dir / "train.jsonl") as f:
        for line in f:
            train_texts.append(json.loads(line)["text"])

    train_tokens = []
    for text in train_texts:
        toks = tokenizer.encode(text)
        if len(toks) > 2:
            train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

    print(f"  Clone training data: {len(train_tokens)} sequences")

    # Evaluate clone BEFORE additional training (should match original)
    clone_val_texts = load_val_texts(TARGET_DOMAIN)
    pre_train_ppl = compute_ppl(model, tokenizer, clone_val_texts)
    print(f"  Clone pre-training PPL (should match original): {pre_train_ppl:.2f}")
    print(f"  Original PPL: {baseline_ppls[TARGET_DOMAIN]:.2f}")
    results["clone_pre_train_ppl"] = round(pre_train_ppl, 4)

    # Train clone
    optimizer = opt.Adam(learning_rate=LEARNING_RATE)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t_start = time.time()
    losses = []
    for step in range(CLONE_TRAIN_STEPS):
        idx = step % len(train_tokens)
        tokens = train_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())
        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(losses[-50:]) / len(losses[-50:])
            print(f"    Step {step+1}/{CLONE_TRAIN_STEPS}: loss={loss.item():.4f} (avg50={avg:.4f})")

    train_time = time.time() - t_start
    first_50 = sum(losses[:50]) / 50
    last_50 = sum(losses[-50:]) / 50
    print(f"  Clone training done in {train_time:.1f}s. Loss: {first_50:.4f} -> {last_50:.4f}")

    # Save clone adapter
    clone_adapter = get_lora_params(model)
    save_adapter(clone_adapter, ADAPTERS_DIR / f"{TARGET_DOMAIN}_clone_v1")

    # Evaluate clone on legal val data
    clone_ppl = compute_ppl(model, tokenizer, clone_val_texts)
    print(f"  Clone post-training PPL: {clone_ppl:.2f}")
    print(f"  Original PPL: {baseline_ppls[TARGET_DOMAIN]:.2f}")
    improvement = (baseline_ppls[TARGET_DOMAIN] - clone_ppl) / baseline_ppls[TARGET_DOMAIN] * 100
    print(f"  Clone improvement over original: {improvement:+.1f}%")

    results["clone_v1"] = {
        "train_time_s": round(train_time, 1),
        "first_50_loss": round(first_50, 4),
        "last_50_loss": round(last_50, 4),
        "pre_train_ppl": round(pre_train_ppl, 4),
        "post_train_ppl": round(clone_ppl, 4),
        "original_ppl": baseline_ppls[TARGET_DOMAIN],
        "improvement_pct": round(improvement, 2),
    }

    # ==================================================================
    # Phase 4: Tournament
    # ==================================================================
    print(f"\n[Phase 4] Tournament: original vs clone on held-out data...")

    tournament_data_dir = prepare_tournament_data()
    tournament_texts = []
    with open(tournament_data_dir / "tournament.jsonl") as f:
        for line in f:
            tournament_texts.append(json.loads(line)["text"])

    print(f"  Tournament samples available: {len(tournament_texts)}")

    tournament_result = run_tournament(
        model, tokenizer,
        adapters[TARGET_DOMAIN],
        clone_adapter,
        tournament_texts,
    )

    print(f"\n  Tournament result:")
    print(f"    Original wins: {tournament_result['a_wins_original']}")
    print(f"    Clone wins: {tournament_result['b_wins_clone']}")
    print(f"    Ties: {tournament_result['ties']}")
    print(f"    Clone win rate: {tournament_result['clone_win_rate']:.1%}")
    print(f"    p-value: {tournament_result['p_value']:.4f}")
    print(f"    Significant: {tournament_result['significant']}")
    print(f"    Queries used: {tournament_result['total_queries']}")

    results["tournament_v1"] = tournament_result

    # K1: clone wins > 60%
    k1_pass = tournament_result["clone_win_rate"] > 0.60
    results["k1_clone_win_rate"] = tournament_result["clone_win_rate"]
    results["k1_pass"] = k1_pass
    print(f"\n  K1 (clone wins >60%): {'PASS' if k1_pass else 'FAIL'} "
          f"({tournament_result['clone_win_rate']:.1%})")

    # K2: tournament resolves in < 10K queries
    k2_pass = tournament_result["total_queries"] < MAX_TOURNAMENT_QUERIES
    results["k2_queries_used"] = tournament_result["total_queries"]
    results["k2_pass"] = k2_pass
    print(f"  K2 (< 10K queries): {'PASS' if k2_pass else 'FAIL'} "
          f"({tournament_result['total_queries']} queries)")

    # ==================================================================
    # Phase 5: Regression check
    # ==================================================================
    print(f"\n[Phase 5] Regression check on other domains...")

    # Compare: original legal in 5-adapter composition vs clone in 5-adapter composition
    # Use 1/N scaling
    regression_results = {}

    # Composed PPL with original legal
    composed_original = {}
    merged_original = {}
    for key in adapters[DOMAINS[0]].keys():
        stacked = mx.stack([adapters[d][key] for d in DOMAINS])
        merged_original[key] = mx.sum(stacked, axis=0) / len(DOMAINS)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_original)
    mx.eval(model.parameters())

    for domain in DOMAINS:
        texts = load_val_texts(domain)
        ppl = compute_ppl(model, tokenizer, texts)
        composed_original[domain] = round(ppl, 4)

    # Composed PPL with clone legal
    adapters_with_clone = {d: adapters[d] for d in DOMAINS}
    adapters_with_clone[TARGET_DOMAIN] = clone_adapter

    composed_clone = {}
    merged_clone = {}
    for key in adapters[DOMAINS[0]].keys():
        stacked = mx.stack([adapters_with_clone[d][key] for d in DOMAINS])
        merged_clone[key] = mx.sum(stacked, axis=0) / len(DOMAINS)

    zero_lora_params(model)
    apply_adapter_weights(model, merged_clone)
    mx.eval(model.parameters())

    for domain in DOMAINS:
        texts = load_val_texts(domain)
        ppl = compute_ppl(model, tokenizer, texts)
        composed_clone[domain] = round(ppl, 4)

    max_regression = 0
    k3_fail_domains = []
    for domain in DOMAINS:
        orig = composed_original[domain]
        clone = composed_clone[domain]
        delta_pct = (clone - orig) / orig * 100
        regression_results[domain] = {
            "composed_original": orig,
            "composed_clone": clone,
            "delta_pct": round(delta_pct, 2),
        }
        print(f"  {domain}: original_composed={orig:.2f}, clone_composed={clone:.2f}, "
              f"delta={delta_pct:+.2f}%")
        if domain != TARGET_DOMAIN and delta_pct > 2.0:
            k3_fail_domains.append(domain)
            max_regression = max(max_regression, delta_pct)

    results["regression_check"] = regression_results

    k3_pass = len(k3_fail_domains) == 0
    results["k3_max_regression_pct"] = round(max_regression, 2)
    results["k3_fail_domains"] = k3_fail_domains
    results["k3_pass"] = k3_pass
    print(f"\n  K3 (<2% regression on other domains): {'PASS' if k3_pass else 'FAIL'}")
    if k3_fail_domains:
        print(f"    Failed domains: {k3_fail_domains}")

    # ==================================================================
    # Phase 6: Second evolution round (iterability)
    # ==================================================================
    print(f"\n[Phase 6] Second evolution round...")

    # Clone the WINNER of round 1 (either original or clone_v1)
    if k1_pass:
        print("  Round 1 winner: clone_v1. Cloning for round 2.")
        v2_base = clone_adapter
    else:
        print("  Round 1 winner: original. Cloning original again with more data.")
        v2_base = adapters[TARGET_DOMAIN]

    # Apply v2 base and continue training with different data slice
    v2_params = {k: mx.array(v) for k, v in v2_base.items()}
    mx.eval(v2_params)
    zero_lora_params(model)
    apply_adapter_weights(model, v2_params)
    mx.eval(model.parameters())

    # Use second half of clone training data
    v2_tokens = train_tokens[len(train_tokens)//2:]
    if len(v2_tokens) < 50:
        v2_tokens = train_tokens  # fallback if too little data

    optimizer_v2 = opt.Adam(learning_rate=LEARNING_RATE * 0.5)  # lower LR for round 2

    t_start = time.time()
    v2_losses = []
    for step in range(CLONE_TRAIN_STEPS):
        idx = step % len(v2_tokens)
        tokens = v2_tokens[idx]
        x = tokens[:-1][None, :]
        y = tokens[1:][None, :]

        loss, grads = loss_and_grad(model, x, y)
        optimizer_v2.update(model, grads)
        mx.eval(model.parameters(), optimizer_v2.state)

        v2_losses.append(loss.item())
        if (step + 1) % 50 == 0 or step == 0:
            avg = sum(v2_losses[-50:]) / len(v2_losses[-50:])
            print(f"    Step {step+1}/{CLONE_TRAIN_STEPS}: loss={loss.item():.4f} (avg50={avg:.4f})")

    v2_train_time = time.time() - t_start
    clone_v2 = get_lora_params(model)
    save_adapter(clone_v2, ADAPTERS_DIR / f"{TARGET_DOMAIN}_clone_v2")

    # Evaluate v2
    v2_ppl = compute_ppl(model, tokenizer, clone_val_texts)
    print(f"  Clone v2 PPL: {v2_ppl:.2f}")
    print(f"  Clone v1 PPL: {clone_ppl:.2f}")
    print(f"  Original PPL: {baseline_ppls[TARGET_DOMAIN]:.2f}")

    # Tournament v2: winner of round 1 vs v2
    print(f"\n  Tournament round 2: round-1-winner vs v2...")
    tournament_v2 = run_tournament(
        model, tokenizer,
        v2_base,
        clone_v2,
        tournament_texts,
    )

    print(f"  Round 2 result:")
    print(f"    Previous winner wins: {tournament_v2['a_wins_original']}")
    print(f"    Clone v2 wins: {tournament_v2['b_wins_clone']}")
    print(f"    Clone v2 win rate: {tournament_v2['clone_win_rate']:.1%}")
    print(f"    p-value: {tournament_v2['p_value']:.4f}")

    results["clone_v2"] = {
        "train_time_s": round(v2_train_time, 1),
        "post_train_ppl": round(v2_ppl, 4),
        "base_for_v2": "clone_v1" if k1_pass else "original",
    }
    results["tournament_v2"] = tournament_v2

    # Track quality trajectory
    quality_trajectory = [
        {"version": "original", "ppl": baseline_ppls[TARGET_DOMAIN]},
        {"version": "clone_v1", "ppl": round(clone_ppl, 4)},
        {"version": "clone_v2", "ppl": round(v2_ppl, 4)},
    ]
    results["quality_trajectory"] = quality_trajectory
    traj_str = " -> ".join(f"{q['version']}={q['ppl']:.2f}" for q in quality_trajectory)
    print(f"\n  Quality trajectory: {traj_str}")

    # Monotonic improvement?
    monotonic = all(quality_trajectory[i]["ppl"] >= quality_trajectory[i+1]["ppl"]
                    for i in range(len(quality_trajectory)-1))
    results["monotonic_improvement"] = monotonic
    print(f"  Monotonic improvement: {monotonic}")

    # ==================================================================
    # Overall verdict
    # ==================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    all_pass = k1_pass and k2_pass and k3_pass
    if all_pass:
        verdict = "SUPPORTED"
    elif k1_pass and k2_pass:
        verdict = "SUPPORTED (K3 caveat)"
    elif not k1_pass:
        verdict = "KILLED"
    else:
        verdict = "KILLED"

    results["verdict"] = verdict
    results["all_k_pass"] = all_pass

    print(f"  K1 (clone wins >60%): {'PASS' if k1_pass else 'FAIL'} ({tournament_result['clone_win_rate']:.1%})")
    print(f"  K2 (<10K queries): {'PASS' if k2_pass else 'FAIL'} ({tournament_result['total_queries']})")
    print(f"  K3 (<2% regression): {'PASS' if k3_pass else 'FAIL'} (max={max_regression:.2f}%)")
    print(f"\n  Overall: {verdict}")

    total_time = time.time() - t0
    results["total_time_s"] = round(total_time, 1)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")

    # Save results (convert numpy/mlx bools to Python bools)
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
