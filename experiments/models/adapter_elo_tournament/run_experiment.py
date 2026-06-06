#!/usr/bin/env python3
"""
Adapter ELO Tournament Experiment

Tests whether an ELO rating system over pairwise composition comparisons
can rank adapter quality. This is a selection mechanism for the Evolve track:
train multiple variants, compare via composition PPL, keep the top-rated.

Kill criteria:
  K1: ELO ranking doesn't correlate with individual adapter quality (Kendall tau < 0.5)
  K2: Tournament overhead > 30 min for 10 adapters

Success criteria:
  S1: ELO ranking correlates well with individual adapter quality (Kendall tau >= 0.7)

Architecture:
  - Base: microsoft/BitNet-b1.58-2B-4T (ternary weights, d=2560, 30 layers)
  - LoRA: rank-16 on all attention + MLP projections
  - Variants: 4 per domain (different seeds + learning rates)
  - Tournament: full round-robin pairwise comparison via composition PPL
  - ELO: K=32, initial rating 1500, logistic model (Bradley-Terry)
"""

import gc
import json
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear

# ===========================================================================
# Configuration
# ===========================================================================
MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
TRAIN_ITERS = 100  # Shorter than reference (200) — we train 4x more adapters
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 15  # Enough for stable PPL estimate, fast enough for tournament

EXPERIMENT_DIR = Path(__file__).parent
ADAPTERS_DIR = EXPERIMENT_DIR / "adapters"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# 3 domains (fewer than reference — we need 4 variants per domain)
DOMAINS = {
    "medical": {
        "hf_dataset": "medalpaca/medical_meadow_medical_flashcards",
        "text_key": "output",
        "max_samples_train": 400,
        "max_samples_val": 50,
    },
    "math": {
        "hf_dataset": "gsm8k",
        "hf_subset": "main",
        "text_key": "answer",
        "max_samples_train": 400,
        "max_samples_val": 50,
    },
    "code": {
        "hf_dataset": "iamtarun/python_code_instructions_18k_alpaca",
        "text_key": "output",
        "max_samples_train": 400,
        "max_samples_val": 50,
    },
}

# Variant configs: different seeds and learning rates to create quality spread
VARIANT_CONFIGS = [
    {"seed": 42,  "lr": 1e-4,  "label": "baseline"},
    {"seed": 123, "lr": 5e-5,  "label": "low_lr"},
    {"seed": 7,   "lr": 2e-4,  "label": "high_lr"},
    {"seed": 99,  "lr": 1e-4,  "label": "alt_seed"},
]

# ELO parameters
ELO_INITIAL = 1500.0
ELO_K = 32.0
ELO_SCALE = 400.0  # Standard: 400/ln(10) ~ 173.7 for logistic model


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


# ===========================================================================
# Ternary unpacking (from reference)
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
    print(f"  Replaced {count} BitLinear -> nn.Linear")
    return model


# ===========================================================================
# LoRA utilities (from reference)
# ===========================================================================
def apply_lora_to_model(model, rank=16, scale=1.0):
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
    print(f"  Applied LoRA (r={rank}) to {count} layers")
    return model


def get_lora_params(model):
    params = {}
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_a" in name or "lora_b" in name:
            params[name] = mx.array(p)
    mx.eval(params)
    return params


def save_adapter(params, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    mx.savez(str(path / "adapter.npz"), **params)


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                in_dims = module.lora_a.shape[0]
                scale = 1.0 / math.sqrt(in_dims)
                module.lora_a = mx.random.uniform(
                    low=-scale, high=scale, shape=module.lora_a.shape
                )
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def compose_adapters(adapter_list: list, scale_per_adapter: float = None):
    N = len(adapter_list)
    if scale_per_adapter is None:
        scale_per_adapter = 1.0 / N
    merged = {}
    for key in adapter_list[0].keys():
        stacked = mx.stack([a[key] for a in adapter_list])
        merged[key] = mx.sum(stacked, axis=0) * scale_per_adapter
    return merged


# ===========================================================================
# Data preparation
# ===========================================================================
def prepare_domain_data(domain_name: str, domain_config: dict) -> Path:
    from datasets import load_dataset as hf_load

    data_dir = EXPERIMENT_DIR / "data" / domain_name
    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"

    if train_path.exists() and valid_path.exists():
        print(f"  Data for {domain_name} already exists")
        return data_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {domain_config['hf_dataset']}...")

    kwargs = {}
    if "hf_subset" in domain_config:
        kwargs["name"] = domain_config["hf_subset"]
    ds = hf_load(domain_config["hf_dataset"], **kwargs)

    split_data = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    text_key = domain_config["text_key"]
    if text_key not in split_data.column_names:
        for alt in ["text", "content", "output", "answer", "response", "question"]:
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

    with open(train_path, "w") as f:
        for t in train_texts:
            json.dump({"text": t}, f)
            f.write("\n")
    with open(valid_path, "w") as f:
        for t in val_texts:
            json.dump({"text": t}, f)
            f.write("\n")

    print(f"  {domain_name}: {len(train_texts)} train, {len(val_texts)} val")
    return data_dir


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = VAL_BATCHES):
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        return float("inf")

    texts = []
    with open(valid_path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])

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
        del logits, loss, x, y

    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


# ===========================================================================
# ELO Rating System
# ===========================================================================
class ELOSystem:
    """Standard ELO rating system for pairwise adapter comparison."""

    def __init__(self, k=ELO_K, scale=ELO_SCALE, initial=ELO_INITIAL):
        self.k = k
        self.scale = scale
        self.initial = initial
        self.ratings = {}
        self.match_history = []

    def add_player(self, name: str):
        self.ratings[name] = self.initial

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / self.scale))

    def record_match(self, winner: str, loser: str):
        """Record a match result and update ratings."""
        e_w = self.expected_score(self.ratings[winner], self.ratings[loser])
        e_l = 1.0 - e_w

        self.ratings[winner] += self.k * (1.0 - e_w)
        self.ratings[loser] += self.k * (0.0 - e_l)

        self.match_history.append({
            "winner": winner,
            "loser": loser,
            "winner_rating": self.ratings[winner],
            "loser_rating": self.ratings[loser],
        })

    def get_ranking(self) -> list:
        """Return players sorted by rating (descending)."""
        return sorted(self.ratings.items(), key=lambda x: -x[1])


def kendall_tau(ranking_a: list, ranking_b: list) -> float:
    """Compute Kendall tau correlation between two rankings.

    ranking_a and ranking_b are lists of items in ranked order (best first).
    Returns tau in [-1, 1]. 1 = perfect agreement, -1 = reversed.
    """
    # Both rankings must contain the same items
    assert set(ranking_a) == set(ranking_b), "Rankings must contain same items"
    n = len(ranking_a)
    if n < 2:
        return 1.0

    # Create position maps
    pos_a = {item: i for i, item in enumerate(ranking_a)}
    pos_b = {item: i for i, item in enumerate(ranking_b)}

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            item_i = ranking_a[i]
            item_j = ranking_a[j]
            # In ranking_a, item_i is before item_j (i < j)
            # Check if same order in ranking_b
            if pos_b[item_i] < pos_b[item_j]:
                concordant += 1
            else:
                discordant += 1

    total_pairs = n * (n - 1) / 2
    return (concordant - discordant) / total_pairs


# ===========================================================================
# Phase functions
# ===========================================================================
def phase_prepare_data():
    """Download and prepare all domain data."""
    print("\n[Phase 1] Preparing domain data...")
    data_dirs = {}
    for domain_name, config in DOMAINS.items():
        data_dirs[domain_name] = prepare_domain_data(domain_name, config)
    return data_dirs


def phase_train_variants(data_dirs: dict):
    """Train all adapter variants across all domains.

    Returns dict of {domain: {variant_label: adapter_path}}.
    Saves adapters to disk between trainings to manage memory.
    """
    print("\n[Phase 2] Training adapter variants...")
    print(f"  {len(DOMAINS)} domains x {len(VARIANT_CONFIGS)} variants = "
          f"{len(DOMAINS) * len(VARIANT_CONFIGS)} adapters to train")

    # Load model once
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Freeze base, unfreeze LoRA
    model.freeze()
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze(keys=["lora_a", "lora_b"], strict=False)

    trainable = sum(p.size for _, p in tree_flatten(model.trainable_parameters()))
    print(f"  Trainable LoRA parameters: {trainable:,}")

    variant_paths = {}
    train_metrics = {}

    for domain_name, data_dir in data_dirs.items():
        variant_paths[domain_name] = {}
        train_metrics[domain_name] = {}

        # Load training data once per domain
        train_texts = []
        with open(data_dir / "train.jsonl") as f:
            for line in f:
                train_texts.append(json.loads(line)["text"])

        for vc in VARIANT_CONFIGS:
            variant_label = f"{domain_name}_{vc['label']}"
            adapter_path = ADAPTERS_DIR / domain_name / vc["label"]

            if (adapter_path / "adapter.npz").exists():
                print(f"\n  --- {variant_label}: already trained, skipping ---")
                variant_paths[domain_name][vc["label"]] = adapter_path
                train_metrics[domain_name][vc["label"]] = {"skipped": True}
                continue

            print(f"\n  --- Training {variant_label} (seed={vc['seed']}, lr={vc['lr']}) ---")

            # Set seed for reproducibility
            mx.random.seed(vc["seed"])

            # Reset LoRA params
            zero_lora_params(model)

            # Tokenize with this seed's shuffling
            import random
            rng = random.Random(vc["seed"])
            shuffled = list(train_texts)
            rng.shuffle(shuffled)

            train_tokens = []
            for text in shuffled:
                toks = tokenizer.encode(text)
                if len(toks) > 2:
                    train_tokens.append(mx.array(toks[:MAX_SEQ_LENGTH + 1]))

            optimizer = opt.Adam(learning_rate=vc["lr"])

            def loss_fn(model, x, y):
                logits = model(x)
                return nn.losses.cross_entropy(logits, y, reduction="mean")

            loss_and_grad = nn.value_and_grad(model, loss_fn)

            t_start = time.time()
            losses = []
            gc.disable()
            for step in range(TRAIN_ITERS):
                idx = step % len(train_tokens)
                tokens = train_tokens[idx]
                x = tokens[:-1][None, :]
                y = tokens[1:][None, :]

                loss, grads = loss_and_grad(model, x, y)
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state, loss)

                loss_val = loss.item()
                losses.append(loss_val)

                if (step + 1) % 50 == 0:
                    avg = sum(losses[-50:]) / len(losses[-50:])
                    print(f"    Step {step+1}/{TRAIN_ITERS}: loss={loss_val:.4f} (avg50={avg:.4f})")

            gc.enable()
            gc.collect()
            train_time = time.time() - t_start

            first_25 = sum(losses[:25]) / 25
            last_25 = sum(losses[-25:]) / 25

            # Save adapter to disk
            params = get_lora_params(model)
            save_adapter(params, adapter_path)
            del params

            variant_paths[domain_name][vc["label"]] = adapter_path
            train_metrics[domain_name][vc["label"]] = {
                "train_time_s": round(train_time, 1),
                "first_25_loss": round(first_25, 4),
                "last_25_loss": round(last_25, 4),
                "converged": last_25 < first_25 * 0.95,
                "seed": vc["seed"],
                "lr": vc["lr"],
            }
            print(f"    Done in {train_time:.1f}s. Loss: {first_25:.4f} -> {last_25:.4f}")

            # Clean up optimizer state
            del optimizer, loss_and_grad, train_tokens
            gc.collect()
            mx.clear_cache()

    # Compute base PPL and individual adapted PPL for ground truth
    print("\n  Computing base and individual PPLs...")
    base_ppls = {}
    individual_ppls = {}

    # Remove LoRA, compute base PPL
    zero_lora_params(model)

    for domain_name, data_dir in data_dirs.items():
        ppl = compute_ppl(model, tokenizer, data_dir)
        base_ppls[domain_name] = ppl
        print(f"  {domain_name} base PPL: {ppl:.2f}")

    # Compute individual adapter PPL (standalone, not composed)
    for domain_name in DOMAINS:
        individual_ppls[domain_name] = {}
        for vc in VARIANT_CONFIGS:
            adapter_path = variant_paths[domain_name][vc["label"]]
            adapter = load_adapter(adapter_path)
            apply_adapter_weights(model, adapter, scale=1.0)
            mx.eval(model.parameters())

            ppl = compute_ppl(model, tokenizer, data_dirs[domain_name])
            individual_ppls[domain_name][vc["label"]] = ppl
            print(f"  {domain_name}/{vc['label']}: adapted PPL = {ppl:.2f}")

            # Reset for next
            zero_lora_params(model)
            del adapter
            gc.collect()

    cleanup(model, tokenizer)
    return variant_paths, train_metrics, base_ppls, individual_ppls


def phase_tournament(data_dirs: dict, variant_paths: dict, base_ppls: dict):
    """Run ELO tournament: pairwise composition comparison within each domain.

    For each domain, compare every pair of variants by swapping them into
    a composition with the best adapter from each OTHER domain (fixed context).
    """
    print("\n[Phase 3] Running ELO tournament...")
    t_tournament_start = time.time()

    # Load model fresh for evaluation
    model, tokenizer = load(MODEL_ID)
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)
    model.freeze()

    domain_names = list(DOMAINS.keys())
    elo_systems = {}
    match_details = {}

    for domain_name in domain_names:
        print(f"\n  --- Tournament for {domain_name} ---")
        elo = ELOSystem()
        variant_labels = [vc["label"] for vc in VARIANT_CONFIGS]
        for vl in variant_labels:
            elo.add_player(vl)

        match_details[domain_name] = []

        # Load context adapters: use the first variant ("baseline") from OTHER domains
        # This provides a fixed composition context for fair comparison
        context_adapters = []
        context_domain_names = [d for d in domain_names if d != domain_name]
        for cd in context_domain_names:
            adapter = load_adapter(variant_paths[cd]["baseline"])
            context_adapters.append(adapter)

        # Full round-robin: compare every pair of variants
        # Run 2 rounds for more stable ratings
        for round_num in range(2):
            pairs = list(combinations(variant_labels, 2))
            for va, vb in pairs:
                # Load variant adapters
                adapter_a = load_adapter(variant_paths[domain_name][va])
                adapter_b = load_adapter(variant_paths[domain_name][vb])

                # Compose with context + variant A
                all_a = context_adapters + [adapter_a]
                composed_a = compose_adapters(all_a)
                apply_adapter_weights(model, composed_a)
                mx.eval(model.parameters())
                ppl_a = compute_ppl(model, tokenizer, data_dirs[domain_name])

                # Reset and compose with context + variant B
                zero_lora_params(model)
                all_b = context_adapters + [adapter_b]
                composed_b = compose_adapters(all_b)
                apply_adapter_weights(model, composed_b)
                mx.eval(model.parameters())
                ppl_b = compute_ppl(model, tokenizer, data_dirs[domain_name])

                # Reset for next pair
                zero_lora_params(model)

                # Record match: lower PPL wins
                if ppl_a < ppl_b:
                    elo.record_match(winner=va, loser=vb)
                    winner = va
                else:
                    elo.record_match(winner=vb, loser=va)
                    winner = vb

                match_details[domain_name].append({
                    "round": round_num,
                    "a": va, "b": vb,
                    "ppl_a": round(ppl_a, 2), "ppl_b": round(ppl_b, 2),
                    "winner": winner,
                })
                print(f"    R{round_num+1} {va} vs {vb}: PPL {ppl_a:.2f} vs {ppl_b:.2f} -> {winner}")

                del adapter_a, adapter_b, composed_a, composed_b, all_a, all_b

        elo_systems[domain_name] = elo

        # Clean up context adapters
        del context_adapters
        gc.collect()
        mx.clear_cache()

    tournament_time = time.time() - t_tournament_start
    print(f"\n  Tournament completed in {tournament_time:.1f}s")

    cleanup(model, tokenizer)
    return elo_systems, match_details, tournament_time


def phase_analyze(elo_systems, individual_ppls, base_ppls, match_details, tournament_time):
    """Compute correlations and assess kill criteria."""
    print("\n[Phase 4] Analyzing results...")

    domain_results = {}
    all_taus = []

    for domain_name in DOMAINS:
        elo = elo_systems[domain_name]
        elo_ranking = elo.get_ranking()  # list of (label, rating), best first
        elo_order = [label for label, _ in elo_ranking]

        # Ground truth: sort by individual PPL (lower is better)
        ppls = individual_ppls[domain_name]
        quality_order = sorted(ppls.keys(), key=lambda k: ppls[k])

        # Compute Kendall tau
        tau = kendall_tau(elo_order, quality_order)
        all_taus.append(tau)

        # Quality ratios (base_ppl / adapted_ppl)
        quality_ratios = {
            k: round(base_ppls[domain_name] / v, 4) for k, v in ppls.items()
        }

        domain_results[domain_name] = {
            "elo_ranking": [{"label": l, "rating": round(r, 1)} for l, r in elo_ranking],
            "quality_ranking": quality_order,
            "individual_ppls": {k: round(v, 2) for k, v in ppls.items()},
            "quality_ratios": quality_ratios,
            "kendall_tau": round(tau, 4),
            "matches": match_details[domain_name],
        }

        print(f"\n  {domain_name}:")
        print(f"    ELO ranking:     {elo_order}")
        print(f"    Quality ranking: {quality_order}")
        for label, rating in elo_ranking:
            print(f"      {label}: ELO={rating:.0f}, PPL={ppls[label]:.2f}, "
                  f"quality={quality_ratios[label]:.3f}")
        print(f"    Kendall tau: {tau:.4f}")

    mean_tau = sum(all_taus) / len(all_taus)
    min_tau = min(all_taus)

    # K1: tau < 0.5 -> KILL
    k1_pass = min_tau >= 0.5
    # K2: tournament overhead > 30 min
    k2_pass = tournament_time < 30 * 60
    # S1: tau >= 0.7
    s1_pass = mean_tau >= 0.7

    print(f"\n  === Kill Criteria ===")
    print(f"  K1: min Kendall tau = {min_tau:.4f} (threshold >= 0.5) -> {'PASS' if k1_pass else 'FAIL'}")
    print(f"  K2: tournament time = {tournament_time:.0f}s (threshold < 1800s) -> {'PASS' if k2_pass else 'FAIL'}")
    print(f"\n  === Success Criteria ===")
    print(f"  S1: mean Kendall tau = {mean_tau:.4f} (threshold >= 0.7) -> {'PASS' if s1_pass else 'FAIL'}")

    return {
        "domain_results": domain_results,
        "mean_kendall_tau": round(mean_tau, 4),
        "min_kendall_tau": round(min_tau, 4),
        "all_taus": [round(t, 4) for t in all_taus],
        "tournament_time_s": round(tournament_time, 1),
        "k1_pass": k1_pass,
        "k2_pass": k2_pass,
        "s1_pass": s1_pass,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    log_memory("start")

    results = {
        "experiment": "adapter_elo_tournament",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "train_iters": TRAIN_ITERS,
        "domains": list(DOMAINS.keys()),
        "n_variants": len(VARIANT_CONFIGS),
        "variant_configs": VARIANT_CONFIGS,
        "elo_k": ELO_K,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Phase 1: Data
    data_dirs = phase_prepare_data()
    log_memory("after-data")

    # Phase 2: Train all variants
    variant_paths, train_metrics, base_ppls, individual_ppls = phase_train_variants(data_dirs)
    log_memory("after-training")
    results["train_metrics"] = train_metrics
    results["base_ppls"] = {k: round(v, 2) for k, v in base_ppls.items()}

    # Phase 3: Tournament
    elo_systems, match_details, tournament_time = phase_tournament(
        data_dirs, variant_paths, base_ppls
    )
    log_memory("after-tournament")

    # Phase 4: Analysis
    analysis = phase_analyze(elo_systems, individual_ppls, base_ppls,
                             match_details, tournament_time)
    results.update(analysis)

    results["total_time_s"] = round(time.time() - t0, 1)

    # Save results
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total time: {results['total_time_s']:.0f}s")


if __name__ == "__main__":
    main()
