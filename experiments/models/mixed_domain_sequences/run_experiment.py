#!/usr/bin/env python3
"""
Mixed-Domain Sequences: Per-token routing on sequences with genuine intra-sequence
domain heterogeneity.

Prior result (exp_molora_per_token_mlx): Per-token routing is equivalent to
per-sequence on homogeneous domains (-0.46%). This experiment tests the regime
where per-token routing SHOULD help: mixed-domain sequences where the first half
is code and the second half is math, etc.

Kill criteria:
  K1 (id=235): Per-token < 5% better than per-sequence on mixed inputs -> KILL
  K2 (id=236): Router can't distinguish domain boundaries within sequence -> KILL

Success criteria:
  S1 (id=19): Per-token routing >10% better than per-sequence on mixed sequences

Platform: Apple M5 Pro 48GB, MLX, $0.
"""

import gc
import json
import math
import os
import random
import time
from pathlib import Path
from collections import defaultdict
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

# Memory safety
device = mx.device_info()
total_mem = device["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Reuse existing adapters and data
ADAPTER_DIR = Path(__file__).parent.parent / "tiny_routing_heads" / "adapters"
DATA_DIR = Path(__file__).parent.parent / "bitnet_2b_real_composition" / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 20  # sequences per domain pair
SEED = 42

# Router config (same as MoLoRA experiment)
ROUTER_HIDDEN_DIM = 64
ROUTER_LR = 3e-4
ROUTER_TRAIN_STEPS = 800  # more steps since mixed data is harder
ROUTER_TEMPERATURE = 1.0
TOP_K = 2

DOMAINS = ["python", "math", "medical", "legal", "creative"]
N_DOMAINS = len(DOMAINS)


def log(msg):
    print(msg, flush=True)


def log_memory(label=""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    peak = mx.get_peak_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB peak={peak:.2f}GB")


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ===========================================================================
# Model loading utilities (reused from prior experiments)
# ===========================================================================
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.bitlinear_layers import BitLinear


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
    log(f"  Applied LoRA (r={rank}) to {count} linear layers")
    return model


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_to_model(model, adapter_params, scale=1.0):
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_adapter_in_model(model):
    updates = []
    for name, p in tree_flatten(model.trainable_parameters()):
        if "lora_b" in name:
            updates.append((name, mx.zeros_like(p)))
    if updates:
        model.update(tree_unflatten(updates))


def compose_adapters(adapter_list, weights=None):
    """Merge multiple adapter parameter dicts with given weights (default 1/N)."""
    N = len(adapter_list)
    if weights is None:
        weights = [1.0 / N] * N
    merged = {}
    for key in adapter_list[0].keys():
        merged[key] = sum(adapter_list[i][key] * weights[i] for i in range(N))
    return merged


def load_domain_texts(domain_name, split="valid"):
    fpath = DATA_DIR / domain_name / f"{split}.jsonl"
    if not fpath.exists():
        return []
    texts = []
    with open(fpath) as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def get_hidden_states(model, x):
    """Extract hidden states from last layer (full sequence, no pooling)."""
    h = model.model.embed_tokens(x)
    for layer in model.model.layers:
        h = layer(h)
    h = model.model.norm(h)
    return h


# ===========================================================================
# Per-Token Router (reused from MoLoRA experiment)
# ===========================================================================
class PerTokenRouter(nn.Module):
    """Per-token router with independent binary gates per adapter.

    Architecture: Linear(d, h) -> GELU -> Linear(h, N_experts)
    Input: hidden states (batch, seq_len, d)
    Output: gate logits (batch, seq_len, N_experts)
    """

    def __init__(self, input_dim, n_experts, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_experts)
        self.n_experts = n_experts

    def __call__(self, h):
        x = nn.gelu(self.fc1(h))
        return self.fc2(x)

    def gumbel_sigmoid_sample(self, logits, temperature=1.0):
        u = mx.random.uniform(shape=logits.shape)
        u = mx.clip(u, 1e-6, 1.0 - 1e-6)
        gumbel_noise = mx.log(u) - mx.log(1.0 - u)
        y = mx.sigmoid((logits + gumbel_noise) / temperature)
        return y


# ===========================================================================
# Phase 0: Create mixed-domain sequences
# ===========================================================================
def phase_create_mixed_data(tokenizer):
    """Create synthetic mixed-domain sequences by concatenating segments.

    For each domain pair (A, B), create sequences that are:
    - First ~50% tokens from domain A
    - Last ~50% tokens from domain B
    The boundary position is tracked for K2 evaluation.
    """
    log("\n" + "=" * 70)
    log("[Phase 0] Creating mixed-domain sequences")
    log("=" * 70)

    rng = random.Random(SEED)

    # Load all domain texts
    domain_texts = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        domain_texts[domain] = texts
        log(f"  {domain}: {len(texts)} texts available")

    # Generate all 10 domain pairs
    domain_pairs = list(combinations(DOMAINS, 2))
    log(f"  Creating {len(domain_pairs)} domain pairs x {VAL_BATCHES} sequences")

    mixed_sequences = []  # list of dicts

    for domain_a, domain_b in domain_pairs:
        texts_a = domain_texts[domain_a]
        texts_b = domain_texts[domain_b]

        for i in range(VAL_BATCHES):
            # Pick random texts from each domain
            text_a = texts_a[rng.randint(0, len(texts_a) - 1)]
            text_b = texts_b[rng.randint(0, len(texts_b) - 1)]

            # Tokenize
            tokens_a = tokenizer.encode(text_a)
            tokens_b = tokenizer.encode(text_b)

            # Target: ~128 tokens from each domain (total ~256)
            half = MAX_SEQ_LENGTH // 2
            seg_a = tokens_a[:half]
            seg_b = tokens_b[:half]

            if len(seg_a) < 10 or len(seg_b) < 10:
                continue

            # Concatenate: [domain_a tokens | domain_b tokens]
            combined = seg_a + seg_b
            boundary_pos = len(seg_a)  # token position where domain changes

            # Create per-token domain labels
            # 0 = domain_a, 1 = domain_b (for this pair)
            domain_labels = [DOMAINS.index(domain_a)] * len(seg_a) + \
                            [DOMAINS.index(domain_b)] * len(seg_b)

            mixed_sequences.append({
                "tokens": combined,
                "boundary_pos": boundary_pos,
                "domain_a": domain_a,
                "domain_b": domain_b,
                "domain_a_idx": DOMAINS.index(domain_a),
                "domain_b_idx": DOMAINS.index(domain_b),
                "per_token_labels": domain_labels,
                "n_tokens": len(combined),
            })

    log(f"  Created {len(mixed_sequences)} mixed sequences")
    log(f"  Avg length: {sum(s['n_tokens'] for s in mixed_sequences)/len(mixed_sequences):.0f} tokens")

    return mixed_sequences


# ===========================================================================
# Phase 1: Collect hidden states for router training on MIXED data
# ===========================================================================
def phase_collect_training_data(model_id, tokenizer_ref):
    """Extract hidden states with per-token domain labels for router training.

    Key difference from MoLoRA: training data includes MIXED sequences where
    different tokens have different domain labels.
    """
    log("\n" + "=" * 70)
    log("[Phase 1] Collecting hidden states for router training")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)

    d_model = model.model.layers[0].self_attn.q_proj.weight.shape[1]
    log(f"  d_model = {d_model}")

    rng = random.Random(SEED)

    # Phase 1a: Pure domain data (same as MoLoRA, for comparison)
    pure_train = []
    pure_val = []
    for domain_idx, domain in enumerate(DOMAINS):
        for split, container in [("train", pure_train), ("valid", pure_val)]:
            texts = load_domain_texts(domain, split=split)
            max_samples = 30 if split == "train" else 15
            for text in texts[:max_samples]:
                tokens = tokenizer.encode(text)
                if len(tokens) < 4:
                    continue
                tokens = tokens[:MAX_SEQ_LENGTH]
                x = mx.array(tokens)[None, :]
                h = get_hidden_states(model, x)
                mx.eval(h)
                # Per-token labels: all tokens get same domain label
                labels = [domain_idx] * h.shape[1]
                container.append((h, labels))
                del x
        log(f"    {domain}: {sum(1 for d in pure_train if d[1][0] == domain_idx)} train, "
            f"{sum(1 for d in pure_val if d[1][0] == domain_idx)} val")

    # Phase 1b: Mixed domain data for training
    log("  Creating mixed training data...")
    domain_texts_train = {}
    for domain in DOMAINS:
        domain_texts_train[domain] = load_domain_texts(domain, split="train")

    mixed_train = []
    domain_pairs = list(combinations(DOMAINS, 2))

    for domain_a, domain_b in domain_pairs:
        texts_a = domain_texts_train[domain_a]
        texts_b = domain_texts_train[domain_b]
        n_mixed = 8  # 8 mixed samples per pair = 80 total

        for _ in range(n_mixed):
            text_a = texts_a[rng.randint(0, len(texts_a) - 1)]
            text_b = texts_b[rng.randint(0, len(texts_b) - 1)]

            toks_a = tokenizer.encode(text_a)
            toks_b = tokenizer.encode(text_b)

            half = MAX_SEQ_LENGTH // 2
            seg_a = toks_a[:half]
            seg_b = toks_b[:half]

            if len(seg_a) < 10 or len(seg_b) < 10:
                continue

            combined = seg_a + seg_b
            x = mx.array(combined)[None, :]
            h = get_hidden_states(model, x)
            mx.eval(h)

            labels = [DOMAINS.index(domain_a)] * len(seg_a) + \
                     [DOMAINS.index(domain_b)] * len(seg_b)
            mixed_train.append((h, labels))
            del x

    log(f"  Mixed training samples: {len(mixed_train)}")

    # Combine pure + mixed for training
    all_train = pure_train + mixed_train

    log(f"  Total training samples: {len(all_train)} (pure: {len(pure_train)}, mixed: {len(mixed_train)})")
    log(f"  Total validation samples: {len(pure_val)}")
    log(f"  Data collection: {time.time() - t0:.1f}s")
    log_memory("after-data-collection")

    result = (all_train, pure_val, d_model, tokenizer)
    cleanup(model)
    return result


# ===========================================================================
# Phase 2: Train per-token router with per-token labels
# ===========================================================================
def phase_train_router(train_data, d_model):
    """Train per-token Gumbel-sigmoid router.

    Key difference from MoLoRA: training uses per-token labels (not per-sequence).
    For mixed sequences, different tokens have different target experts.
    """
    log("\n" + "=" * 70)
    log("[Phase 2] Training per-token router (with per-token labels)")
    log("=" * 70)

    router = PerTokenRouter(d_model, N_DOMAINS, ROUTER_HIDDEN_DIM)
    mx.eval(router.parameters())

    n_params = sum(p.size for _, p in tree_flatten(router.parameters()))
    log(f"  Router params: {n_params:,}")

    optimizer = opt.Adam(learning_rate=ROUTER_LR)
    rng = random.Random(SEED)

    def router_loss_fn(router, h, labels_array):
        """BCE loss with per-token labels.

        labels_array: (1, T) int array of domain indices per token.
        Target: one-hot(labels[t]) for each token position t.
        """
        logits = router(h)  # (1, T, N)
        gates = router.gumbel_sigmoid_sample(logits, temperature=ROUTER_TEMPERATURE)

        # Build per-token target from labels
        # labels_array is (1, T) -> one-hot is (1, T, N)
        target = mx.zeros_like(gates)
        for expert_idx in range(N_DOMAINS):
            mask = mx.equal(labels_array, expert_idx)  # (1, T)
            mask = mx.expand_dims(mask, axis=-1)  # (1, T, 1)
            # Create one-hot column
            col = mx.zeros((1, 1, N_DOMAINS))
            col_list = [mx.zeros((1, 1, 1)) for _ in range(N_DOMAINS)]
            col_list[expert_idx] = mx.ones((1, 1, 1))
            col = mx.concatenate(col_list, axis=-1)  # (1, 1, N)
            target = target + mask.astype(gates.dtype) * col

        loss = nn.losses.binary_cross_entropy(gates, target, reduction="mean")
        return loss

    loss_and_grad = nn.value_and_grad(router, router_loss_fn)

    gc.disable()
    losses = []
    t0 = time.time()

    for step in range(ROUTER_TRAIN_STEPS):
        idx = rng.randint(0, len(train_data) - 1)
        h, labels = train_data[idx]
        labels_array = mx.array(labels)[None, :]  # (1, T)

        loss, grads = loss_and_grad(router, h, labels_array)
        optimizer.update(router, grads)
        mx.eval(router.parameters(), optimizer.state, loss)
        losses.append(loss.item())

        if (step + 1) % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            log(f"    Step {step+1}/{ROUTER_TRAIN_STEPS}: loss={avg:.4f}")

    gc.enable()
    gc.collect()

    log(f"  Router training: {time.time() - t0:.1f}s")
    log(f"  Final loss (last 50): {sum(losses[-50:])/len(losses[-50:]):.4f}")

    # Save router
    router_path = EXPERIMENT_DIR / "router"
    router_path.mkdir(parents=True, exist_ok=True)
    router_params = dict(tree_flatten(router.parameters()))
    mx.savez(str(router_path / "router.npz"), **{k: v for k, v in router_params.items()})
    log(f"  Saved router to {router_path}")

    return router, n_params


# ===========================================================================
# Phase 3: Evaluate on mixed-domain sequences
# ===========================================================================
def phase_evaluate_mixed(model_id, router, mixed_sequences):
    """Compare per-token vs per-sequence routing on mixed-domain sequences.

    Three conditions:
    1. Uniform 1/N: all adapters equally weighted
    2. Per-sequence: router uses mean-pooled hidden states -> top-2 for whole sequence
    3. Per-token: router assigns top-2 per token position

    Also measures K2: whether routing patterns change at domain boundaries.
    """
    log("\n" + "=" * 70)
    log("[Phase 3] Evaluating on mixed-domain sequences")
    log("=" * 70)

    t0 = time.time()
    model, tokenizer = load(model_id)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # Load adapters
    adapters = {}
    for domain in DOMAINS:
        adapters[domain] = load_adapter(ADAPTER_DIR / domain)
        log(f"  Loaded adapter: {domain}")

    # Precompute uniform composition
    uniform_composed = compose_adapters([adapters[d] for d in DOMAINS])

    # Per-pair results
    pair_results = {}
    boundary_detection = {}  # K2: routing pattern changes at boundaries

    # Group sequences by pair
    pair_sequences = defaultdict(list)
    for seq in mixed_sequences:
        pair_key = f"{seq['domain_a']}+{seq['domain_b']}"
        pair_sequences[pair_key].append(seq)

    for pair_key, sequences in pair_sequences.items():
        log(f"\n  === Pair: {pair_key} ({len(sequences)} sequences) ===")

        uniform_losses = 0.0
        uniform_tokens = 0
        per_seq_losses = 0.0
        per_seq_tokens = 0
        per_token_losses = 0.0
        per_token_tokens = 0
        oracle_losses = 0.0
        oracle_tokens = 0

        # K2 tracking: for each sequence, record expert selection per position
        pair_boundary_scores = []

        for seq_data in sequences:
            tokens = seq_data["tokens"]
            boundary = seq_data["boundary_pos"]
            domain_a_idx = seq_data["domain_a_idx"]
            domain_b_idx = seq_data["domain_b_idx"]

            if len(tokens) < 3:
                continue

            x = mx.array(tokens[:-1])[None, :]
            y = mx.array(tokens[1:])[None, :]
            T = x.shape[1]

            # --- Condition 1: Uniform 1/N ---
            apply_adapter_to_model(model, uniform_composed)
            logits_u = model(x)
            loss_u = nn.losses.cross_entropy(logits_u, y, reduction="sum")
            mx.eval(loss_u)
            uniform_losses += loss_u.item()
            uniform_tokens += y.size
            zero_adapter_in_model(model)
            del logits_u, loss_u

            # --- Get hidden states for routing ---
            h = get_hidden_states(model, x)
            mx.eval(h)

            # --- Condition 2: Per-sequence routing ---
            h_pool = mx.mean(h, axis=1, keepdims=True)  # (1, 1, d)
            seq_logits = router(h_pool)
            seq_scores = mx.sigmoid(seq_logits[0, 0])
            mx.eval(seq_scores)
            score_list = seq_scores.tolist()

            sorted_experts = sorted(enumerate(score_list), key=lambda x: x[1], reverse=True)
            top2 = sorted_experts[:TOP_K]
            total_score = sum(s for _, s in top2)
            if total_score < 1e-8:
                total_score = 1.0

            composed_seq = {}
            for expert_idx, expert_score in top2:
                w = expert_score / total_score
                for key, val in adapters[DOMAINS[expert_idx]].items():
                    if key not in composed_seq:
                        composed_seq[key] = val * w
                    else:
                        composed_seq[key] = composed_seq[key] + val * w

            apply_adapter_to_model(model, composed_seq)
            logits_s = model(x)
            loss_s = nn.losses.cross_entropy(logits_s, y, reduction="sum")
            mx.eval(loss_s)
            per_seq_losses += loss_s.item()
            per_seq_tokens += y.size
            zero_adapter_in_model(model)
            del logits_s, loss_s, composed_seq, h_pool, seq_logits, seq_scores

            # --- Condition 3: Per-token routing ---
            gate_logits = router(h)
            gate_probs = mx.sigmoid(gate_logits)
            mx.eval(gate_probs)
            gate_np = gate_probs[0].tolist()  # T x N

            # Group tokens by expert set
            token_groups = {}
            token_weights_map = {}

            per_token_expert_selections = []  # for K2 boundary detection

            for t in range(T):
                scores = gate_np[t]
                sorted_exp = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                top_k = sorted_exp[:TOP_K]
                expert_set = frozenset(idx for idx, _ in top_k)
                total_s = sum(s for _, s in top_k)
                if total_s < 1e-8:
                    total_s = 1.0
                weights = {idx: s / total_s for idx, s in top_k}

                if expert_set not in token_groups:
                    token_groups[expert_set] = []
                    token_weights_map[expert_set] = weights
                token_groups[expert_set].append(t)

                # Track which expert is primary for K2
                primary_expert = top_k[0][0]
                per_token_expert_selections.append(primary_expert)

            # Compute per-token loss per group
            for expert_set, positions in token_groups.items():
                weights = token_weights_map[expert_set]
                composed_group = {}
                for expert_idx, w in weights.items():
                    for key, val in adapters[DOMAINS[expert_idx]].items():
                        if key not in composed_group:
                            composed_group[key] = val * w
                        else:
                            composed_group[key] = composed_group[key] + val * w

                apply_adapter_to_model(model, composed_group)
                logits_t = model(x)
                mx.eval(logits_t)

                for pos in positions:
                    if pos < T:
                        token_logits = logits_t[0, pos:pos+1]
                        token_target = y[0, pos:pos+1]
                        token_loss = nn.losses.cross_entropy(
                            token_logits, token_target, reduction="sum"
                        )
                        mx.eval(token_loss)
                        per_token_losses += token_loss.item()
                        per_token_tokens += 1
                        del token_logits, token_target, token_loss

                zero_adapter_in_model(model)
                del composed_group, logits_t

            # --- Condition 4: Oracle (knows the boundary) ---
            # Use domain_a adapter for first half, domain_b for second half
            oracle_composed_a = adapters[seq_data["domain_a"]]
            oracle_composed_b = adapters[seq_data["domain_b"]]

            # First half: domain_a adapter
            apply_adapter_to_model(model, oracle_composed_a)
            logits_o = model(x)
            mx.eval(logits_o)

            for pos in range(min(boundary - 1, T)):
                if pos < T:
                    token_logits = logits_o[0, pos:pos+1]
                    token_target = y[0, pos:pos+1]
                    token_loss = nn.losses.cross_entropy(
                        token_logits, token_target, reduction="sum"
                    )
                    mx.eval(token_loss)
                    oracle_losses += token_loss.item()
                    oracle_tokens += 1
                    del token_logits, token_target, token_loss
            zero_adapter_in_model(model)
            del logits_o

            # Second half: domain_b adapter
            apply_adapter_to_model(model, oracle_composed_b)
            logits_o = model(x)
            mx.eval(logits_o)

            for pos in range(max(boundary - 1, 0), T):
                if pos < T:
                    token_logits = logits_o[0, pos:pos+1]
                    token_target = y[0, pos:pos+1]
                    token_loss = nn.losses.cross_entropy(
                        token_logits, token_target, reduction="sum"
                    )
                    mx.eval(token_loss)
                    oracle_losses += token_loss.item()
                    oracle_tokens += 1
                    del token_logits, token_target, token_loss
            zero_adapter_in_model(model)
            del logits_o

            # --- K2: Boundary detection ---
            # Check if primary expert changes near the boundary
            if boundary > 5 and boundary < T - 5:
                # Expert selection in first half (positions before boundary)
                first_half = per_token_expert_selections[:boundary - 1]
                second_half = per_token_expert_selections[boundary - 1:]

                # Fraction of first-half tokens selecting domain_a
                correct_a = sum(1 for e in first_half if e == domain_a_idx)
                frac_a = correct_a / len(first_half) if first_half else 0

                # Fraction of second-half tokens selecting domain_b
                correct_b = sum(1 for e in second_half if e == domain_b_idx)
                frac_b = correct_b / len(second_half) if second_half else 0

                pair_boundary_scores.append({
                    "frac_correct_a": frac_a,
                    "frac_correct_b": frac_b,
                    "avg_correct": (frac_a + frac_b) / 2,
                    "boundary_pos": boundary,
                    "T": T,
                })

            del h, gate_logits, gate_probs, x, y

        # Compute PPLs for this pair
        pair_uniform_ppl = math.exp(min(uniform_losses / max(uniform_tokens, 1), 100))
        pair_per_seq_ppl = math.exp(min(per_seq_losses / max(per_seq_tokens, 1), 100))
        pair_per_token_ppl = math.exp(min(per_token_losses / max(per_token_tokens, 1), 100))
        pair_oracle_ppl = math.exp(min(oracle_losses / max(oracle_tokens, 1), 100))

        log(f"    Uniform 1/N:   {pair_uniform_ppl:.4f}")
        log(f"    Per-sequence:   {pair_per_seq_ppl:.4f}")
        log(f"    Per-token:      {pair_per_token_ppl:.4f}")
        log(f"    Oracle:         {pair_oracle_ppl:.4f}")

        pair_results[pair_key] = {
            "uniform_ppl": round(pair_uniform_ppl, 4),
            "per_seq_ppl": round(pair_per_seq_ppl, 4),
            "per_token_ppl": round(pair_per_token_ppl, 4),
            "oracle_ppl": round(pair_oracle_ppl, 4),
            "n_sequences": len(sequences),
        }

        if pair_boundary_scores:
            avg_correct = sum(s["avg_correct"] for s in pair_boundary_scores) / len(pair_boundary_scores)
            avg_frac_a = sum(s["frac_correct_a"] for s in pair_boundary_scores) / len(pair_boundary_scores)
            avg_frac_b = sum(s["frac_correct_b"] for s in pair_boundary_scores) / len(pair_boundary_scores)
            boundary_detection[pair_key] = {
                "avg_boundary_accuracy": round(avg_correct, 4),
                "avg_frac_correct_a": round(avg_frac_a, 4),
                "avg_frac_correct_b": round(avg_frac_b, 4),
                "n_samples": len(pair_boundary_scores),
            }
            log(f"    Boundary detection: {avg_correct:.2%} correct (A:{avg_frac_a:.2%}, B:{avg_frac_b:.2%})")

    log_memory("after-eval")

    result = {
        "pair_results": pair_results,
        "boundary_detection": boundary_detection,
    }
    cleanup(model, tokenizer)
    return result


# ===========================================================================
# Phase 4: Kill criteria assessment
# ===========================================================================
def phase_analysis(eval_results, router_n_params):
    """Assess kill criteria and write results."""
    log("\n" + "=" * 70)
    log("[Phase 4] Kill criteria assessment")
    log("=" * 70)

    pair_results = eval_results["pair_results"]
    boundary_detection = eval_results["boundary_detection"]

    # Aggregate across all pairs
    all_uniform = [r["uniform_ppl"] for r in pair_results.values()]
    all_per_seq = [r["per_seq_ppl"] for r in pair_results.values()]
    all_per_token = [r["per_token_ppl"] for r in pair_results.values()]
    all_oracle = [r["oracle_ppl"] for r in pair_results.values()]

    avg_uniform = sum(all_uniform) / len(all_uniform)
    avg_per_seq = sum(all_per_seq) / len(all_per_seq)
    avg_per_token = sum(all_per_token) / len(all_per_token)
    avg_oracle = sum(all_oracle) / len(all_oracle)

    log(f"\n  Overall PPL (averaged across {len(pair_results)} domain pairs):")
    log(f"    Uniform 1/N:      {avg_uniform:.4f}")
    log(f"    Per-sequence:     {avg_per_seq:.4f}")
    log(f"    Per-token:        {avg_per_token:.4f}")
    log(f"    Oracle:           {avg_oracle:.4f}")

    # Per-pair breakdown
    log(f"\n  Per-pair breakdown:")
    log(f"    {'Pair':<25} {'Uniform':>10} {'PerSeq':>10} {'PerToken':>10} {'Oracle':>10} {'PT vs PS':>10}")
    for pair_key, r in sorted(pair_results.items()):
        improvement = ((r["per_seq_ppl"] - r["per_token_ppl"]) / r["per_seq_ppl"]) * 100
        log(f"    {pair_key:<25} {r['uniform_ppl']:>10.4f} {r['per_seq_ppl']:>10.4f} "
            f"{r['per_token_ppl']:>10.4f} {r['oracle_ppl']:>10.4f} {improvement:>+9.1f}%")

    # K1: Per-token > 5% better than per-sequence on mixed inputs
    improvement_pct = ((avg_per_seq - avg_per_token) / avg_per_seq) * 100
    k1_pass = improvement_pct >= 5.0
    log(f"\n  K1: Per-token vs per-sequence improvement: {improvement_pct:+.2f}%")
    log(f"      Threshold: >= 5% improvement")
    log(f"      {'PASS' if k1_pass else 'FAIL (KILL)'}")

    # K2: Router can distinguish domain boundaries
    if boundary_detection:
        all_boundary_acc = [b["avg_boundary_accuracy"] for b in boundary_detection.values()]
        avg_boundary_acc = sum(all_boundary_acc) / len(all_boundary_acc)
        # K2 passes if boundary accuracy > 0.5 (better than random)
        # Random baseline for 5 experts = 0.2, so 0.4 is 2x random
        k2_pass = avg_boundary_acc > 0.4
        log(f"\n  K2: Avg boundary detection accuracy: {avg_boundary_acc:.2%}")
        log(f"      Threshold: > 40% (2x random=20%)")
        log(f"      {'PASS' if k2_pass else 'FAIL (KILL)'}")

        log(f"\n  Boundary detection per pair:")
        for pair_key, b in sorted(boundary_detection.items()):
            log(f"    {pair_key:<25} accuracy={b['avg_boundary_accuracy']:.2%} "
                f"(A:{b['avg_frac_correct_a']:.2%}, B:{b['avg_frac_correct_b']:.2%})")
    else:
        k2_pass = False
        avg_boundary_acc = 0.0
        log("\n  K2: No boundary detection data available -> FAIL")

    # S1: Per-token routing >10% better than per-sequence
    s1_pass = improvement_pct >= 10.0
    log(f"\n  S1: Per-token >10% better than per-sequence")
    log(f"      Actual improvement: {improvement_pct:+.2f}%")
    log(f"      {'PASS' if s1_pass else 'FAIL'}")

    # Oracle gap analysis
    oracle_gap = ((avg_per_token - avg_oracle) / avg_oracle) * 100
    log(f"\n  Oracle gap: per-token is {oracle_gap:+.1f}% above oracle")
    log(f"  (Room for improvement: oracle uses perfect domain knowledge)")

    # How much of the oracle advantage does per-token capture?
    if avg_uniform > avg_oracle:
        total_range = avg_uniform - avg_oracle
        per_token_capture = (avg_uniform - avg_per_token) / total_range * 100 if total_range > 0 else 0
        per_seq_capture = (avg_uniform - avg_per_seq) / total_range * 100 if total_range > 0 else 0
        log(f"  Per-sequence captures {per_seq_capture:.1f}% of uniform->oracle gap")
        log(f"  Per-token captures {per_token_capture:.1f}% of uniform->oracle gap")

    # Build results
    results = {
        "experiment": "mixed_domain_sequences",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "n_domain_pairs": len(pair_results),
        "sequences_per_pair": VAL_BATCHES,
        "lora_rank": LORA_RANK,
        "router_hidden_dim": ROUTER_HIDDEN_DIM,
        "router_train_steps": ROUTER_TRAIN_STEPS,
        "router_n_params": router_n_params,
        "top_k": TOP_K,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pair_results": pair_results,
        "boundary_detection": boundary_detection,
        "summary": {
            "avg_uniform_ppl": round(avg_uniform, 4),
            "avg_per_seq_ppl": round(avg_per_seq, 4),
            "avg_per_token_ppl": round(avg_per_token, 4),
            "avg_oracle_ppl": round(avg_oracle, 4),
            "per_token_vs_per_seq_pct": round(improvement_pct, 2),
            "avg_boundary_accuracy": round(avg_boundary_acc, 4),
        },
        "K1_improvement_pct": round(improvement_pct, 2),
        "K1_threshold": 5.0,
        "K1_pass": k1_pass,
        "K2_boundary_accuracy": round(avg_boundary_acc, 4),
        "K2_threshold": 0.4,
        "K2_pass": k2_pass,
        "S1_improvement_pct": round(improvement_pct, 2),
        "S1_threshold": 10.0,
        "S1_pass": s1_pass,
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\n  Results saved to {RESULTS_FILE}")

    return results


# ===========================================================================
# Main
# ===========================================================================
def main():
    t0 = time.time()
    log_memory("start")

    log("=" * 70)
    log("Mixed-Domain Sequences: Per-token vs Per-sequence Routing")
    log("=" * 70)

    # Phase 0: Create mixed sequences (needs tokenizer)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    mixed_sequences = phase_create_mixed_data(tokenizer)
    del tokenizer
    log_memory("after-mixed-data")

    # Phase 1: Collect hidden states for training
    train_data, val_data, d_model, tokenizer = phase_collect_training_data(MODEL_ID, None)
    log_memory("after-training-data")

    # Phase 2: Train router with per-token labels
    router, n_params = phase_train_router(train_data, d_model)
    # Free training data
    del train_data, val_data
    gc.collect()
    mx.clear_cache()
    log_memory("after-router-training")

    # Phase 3: Evaluate on mixed sequences
    eval_results = phase_evaluate_mixed(MODEL_ID, router, mixed_sequences)
    log_memory("after-evaluation")

    # Phase 4: Analysis
    results = phase_analysis(eval_results, n_params)

    total_time = time.time() - t0
    log(f"\n  Total runtime: {total_time:.1f}s ({total_time/60:.1f}min)")
    log(f"\n  VERDICT: K1={'PASS' if results['K1_pass'] else 'FAIL'}, "
        f"K2={'PASS' if results['K2_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
