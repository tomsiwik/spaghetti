#!/usr/bin/env python3
"""Room Model POC: pre-summed W_combined with automatic soft routing at N=5.

Verifies three theorems from MATH.md:
  Theorem 1: Pre-summed equivalence (linearity of matmul)
  Theorem 2: Soft routing from projection geometry
  Theorem 3: Bandwidth-speed prediction (~40-50 tok/s)

Kill criteria:
  K763: W_combined output MSE vs sequential v3 < 1e-6 (mathematical equivalence)
  K764: Soft routing domain accuracy >= 60% on 5-domain validation set
  K765: Composed PPL within 10% of v3 N=5 baseline

Platform: Apple M5 Pro 48GB, MLX
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

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Sources
NTP_SOURCE = EXPERIMENT_DIR.parent / "real_data_domain_experts"
SFT_SOURCE = EXPERIMENT_DIR.parent / "bitnet_sft_generation_v3" / "sft_adapters"
SKELETON_PATH = NTP_SOURCE / "adapters" / "grassmannian_skeleton.npz"
DATA_DIR = NTP_SOURCE / "data"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_SCALE = 20.0
MAX_SEQ = 256
SEED = 42
DOMAINS = ["medical", "code", "math", "legal", "finance"]
N_DOMAINS = len(DOMAINS)
N_VAL = 30  # validation samples per domain for PPL
N_ROUTE_TEST = 20  # samples per domain for routing accuracy

ADAPTER_TARGETS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)


def log(m):
    print(m, flush=True)


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


def load_data(domain, split="valid", n=None):
    """Load text data from jsonl file."""
    texts = []
    with open(DATA_DIR / domain / f"{split}.jsonl") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if n and len(texts) >= n:
                break
    return texts


# ============================================================================
# Room Model: Dense Additive Linear
# ============================================================================

class RoomLinear(nn.Module):
    """Additive dense delta: y = base(x) + x @ W_combined.

    W_combined = sum_i alpha * A_i @ B_i for all N domain adapters.
    Stored as a single bf16 matrix per module.
    """

    def __init__(self, base: nn.Module, W_combined: mx.array):
        super().__init__()
        self.base = base
        # W_combined: (in_features, out_features) in bf16
        self._w_combined = W_combined.astype(mx.bfloat16)
        self.freeze()

    def __call__(self, x):
        y = self.base(x)
        return y + (x @ self._w_combined).astype(y.dtype)


def _get_module_dims(frozen_A_data, adapter_Bs_data, li, key, n_domains):
    """Get (in_features, out_features) from A/B shapes for a module.

    BitLinear weights are packed (ternary), so m.weight.shape is NOT
    the logical (out, in). We derive dimensions from the A/B matrices.
    """
    for di in range(n_domains):
        ak = f"layer_{li}_{key}_domain_{di}"
        bk = f"model.layers.{li}.{key}.lora_b"
        if ak in frozen_A_data and bk in adapter_Bs_data[di]:
            A = frozen_A_data[ak]
            B = adapter_Bs_data[di][bk]
            in_f = A.shape[0] if isinstance(A, np.ndarray) else A.shape[0]
            out_f = B.shape[1] if isinstance(B, np.ndarray) else B.shape[1]
            return in_f, out_f
    return None, None


# ============================================================================
# Phase 1: Mathematical Equivalence (K763)
# ============================================================================

def phase_equivalence():
    """Verify Theorem 1: pre-summed output == sequential adapter sum.

    For each module, compute:
      sequential_output = sum_i alpha * (x @ A_i) @ B_i
      room_output = x @ W_combined  where W_combined = sum_i alpha * A_i @ B_i
    MSE should be < 1e-6 (float32 computation).
    """
    log("\n=== Phase 1: Mathematical Equivalence (K763) ===")

    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)

    # Load all adapter B-matrices and frozen A-matrices
    frozen_A = dict(np.load(str(SKELETON_PATH)))
    adapter_Bs = {}
    for di, domain in enumerate(DOMAINS):
        path = SFT_SOURCE / domain / "adapter.npz"
        adapter_Bs[di] = dict(mx.load(str(path)))

    # Create a test input (a medical text for interesting hidden states)
    test_text = load_data("medical", "valid", 1)[0]
    test_tokens = tokenizer.encode(test_text)[:MAX_SEQ]
    input_ids = mx.array(test_tokens)[None, :]  # (1, T)

    # Test equivalence using random inputs with correct dimensions per module.
    # The proof is purely algebraic (distributive property) -- the actual
    # hidden states don't matter, only dimensional correctness.
    log("  Testing algebraic equivalence with random inputs per module...")

    mx.random.seed(SEED)
    T = 16  # short sequence for testing

    mse_values = []
    max_mse = 0.0
    n_modules_tested = 0

    # Test 3 representative layers
    test_layers = [0, 14, 29]

    for li in test_layers:
        layer = model.model.layers[li]

        for key in ADAPTER_TARGETS:
            # Get the base module to determine dimensions
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            # Determine dimensions from A and B (NOT from packed BitLinear weight)
            # A: (in_features, rank), B: (rank, out_features)
            # Find a valid A/B pair to get dimensions
            sample_A = None
            sample_B = None
            for di_s in range(N_DOMAINS):
                ak_s = f"layer_{li}_{key}_domain_{di_s}"
                bk_s = f"model.layers.{li}.{key}.lora_b"
                if ak_s in frozen_A and bk_s in adapter_Bs[di_s]:
                    sample_A = frozen_A[ak_s]
                    sample_B = np.array(adapter_Bs[di_s][bk_s])
                    break
            if sample_A is None:
                continue

            in_features = sample_A.shape[0]
            out_features = sample_B.shape[1]

            # Random input with correct in_features dimension
            x = mx.random.normal((1, T, in_features)).astype(mx.float32)
            mx.eval(x)

            # Sequential: sum of individual adapter outputs (in float32)
            sequential_sum = mx.zeros((1, T, out_features), dtype=mx.float32)

            # Compute W_combined in float32
            w_combined = mx.zeros((in_features, out_features), dtype=mx.float32)

            for di in range(N_DOMAINS):
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"

                if ak not in frozen_A or bk not in adapter_Bs[di]:
                    continue

                A = mx.array(frozen_A[ak]).astype(mx.float32)  # (in, rank)
                B = adapter_Bs[di][bk].astype(mx.float32)      # (rank, out)

                # Sequential: alpha * (x @ A) @ B
                adapter_out = LORA_SCALE * (x @ A) @ B
                mx.eval(adapter_out)
                sequential_sum = sequential_sum + adapter_out

                # Accumulate W_combined: alpha * A @ B
                delta_W = LORA_SCALE * (A @ B)
                mx.eval(delta_W)
                w_combined = w_combined + delta_W

            mx.eval(sequential_sum, w_combined)

            # Room model: x @ W_combined
            room_out = x @ w_combined
            mx.eval(room_out)

            # MSE
            diff = sequential_sum - room_out
            mse = mx.mean(diff * diff).item()
            mse_values.append(mse)
            max_mse = max(max_mse, mse)
            n_modules_tested += 1

            if mse > 1e-8:
                log(f"    WARNING: layer {li} {key}: MSE = {mse:.2e}")

            del x, sequential_sum, w_combined, room_out
        gc.collect()

    mean_mse = float(np.mean(mse_values))
    log(f"  Per-module: tested {n_modules_tested} modules across layers {test_layers}")
    log(f"  Per-module mean MSE: {mean_mse:.2e}")
    log(f"  Per-module max MSE:  {max_mse:.2e}")

    cleanup(model, tokenizer)
    del frozen_A, adapter_Bs
    gc.collect()
    mx.clear_cache()

    # --- Full-model logit equivalence test ---
    # Inject room model, run forward pass, compare logits against
    # sequential adapter application (sum of all 5 adapters).
    log("  Full-model logit equivalence test...")

    from mlx_lm import load as mlx_load
    from pierre.pierre import attach_adapter, detach_adapters
    from mlx.utils import tree_unflatten

    # (a) Get room model logits
    model_room, tok_room = mlx_load(MODEL_ID)
    frozen_A_room = dict(np.load(str(SKELETON_PATH)))
    adapter_Bs_room = {}
    for di, domain in enumerate(DOMAINS):
        adapter_Bs_room[di] = dict(mx.load(str(SFT_SOURCE / domain / "adapter.npz")))

    for li in range(len(model_room.model.layers)):
        layer = model_room.model.layers[li]
        updates = []
        for key_name in ADAPTER_TARGETS:
            m = layer
            for part in key_name.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue
            in_f, out_f = _get_module_dims(
                frozen_A_room, adapter_Bs_room, li, key_name, N_DOMAINS)
            if in_f is None:
                continue
            wc = mx.zeros((in_f, out_f), dtype=mx.float32)
            for di in range(N_DOMAINS):
                ak = f"layer_{li}_{key_name}_domain_{di}"
                bk = f"model.layers.{li}.{key_name}.lora_b"
                if ak not in frozen_A_room or bk not in adapter_Bs_room[di]:
                    continue
                A = mx.array(frozen_A_room[ak]).astype(mx.float32)
                B = adapter_Bs_room[di][bk].astype(mx.float32)
                wc = wc + LORA_SCALE * (A @ B)
            mx.eval(wc)
            updates.append((key_name, RoomLinear(m, wc)))
        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model_room.parameters())

    test_text = load_data("medical", "valid", 1)[0]
    test_tokens = tok_room.encode(test_text)[:64]  # short for memory
    input_ids = mx.array(test_tokens)[None, :]
    room_logits = model_room(input_ids).astype(mx.float32)
    mx.eval(room_logits)
    room_logits_np = np.array(room_logits)

    cleanup(model_room, tok_room)
    del frozen_A_room, adapter_Bs_room
    gc.collect()
    mx.clear_cache()

    # (b) Get sequential all-adapter-sum logits
    # This means: run the base model with ALL adapters attached simultaneously.
    # But v3 only supports attaching one adapter at a time.
    # So we compute: for each adapter, get the logit delta, then sum them.
    # logits_total = base_logits + sum_i (logits_with_adapter_i - base_logits)
    #              = base_logits + sum_i delta_logits_i

    model_base, tok_base = mlx_load(MODEL_ID)
    frozen_A_seq = dict(np.load(str(SKELETON_PATH)))

    input_ids_seq = mx.array(test_tokens)[None, :]
    base_logits = model_base(input_ids_seq).astype(mx.float32)
    mx.eval(base_logits)
    base_logits_np = np.array(base_logits)

    delta_logits_sum = np.zeros_like(base_logits_np)

    for di, domain in enumerate(DOMAINS):
        adapter_B = dict(mx.load(str(SFT_SOURCE / domain / "adapter.npz")))
        attach_adapter(model_base, frozen_A_seq, adapter_B, di, LORA_SCALE)

        logits_with = model_base(input_ids_seq).astype(mx.float32)
        mx.eval(logits_with)
        logits_with_np = np.array(logits_with)

        delta_logits_sum += (logits_with_np - base_logits_np)

        detach_adapters(model_base)
        del adapter_B
        gc.collect()

    seq_logits_np = base_logits_np + delta_logits_sum
    cleanup(model_base, tok_base)
    del frozen_A_seq
    gc.collect()
    mx.clear_cache()

    # MSE between room model and sequential-sum logits
    logit_diff = room_logits_np - seq_logits_np
    logit_mse = float(np.mean(logit_diff ** 2))
    logit_max_abs = float(np.max(np.abs(logit_diff)))

    log(f"  Full-model logit MSE: {logit_mse:.2e}")
    log(f"  Full-model max |diff|: {logit_max_abs:.2e}")

    # K763 uses the logit-level MSE (stricter test)
    k763_pass = logit_mse < 1e-6
    log(f"  K763: {'PASS' if k763_pass else 'FAIL'} (logit MSE {logit_mse:.2e} vs threshold 1e-6)")

    return {
        "n_modules_tested": n_modules_tested,
        "layers_tested": test_layers,
        "per_module_mean_mse": mean_mse,
        "per_module_max_mse": max_mse,
        "logit_mse": logit_mse,
        "logit_max_abs_diff": logit_max_abs,
        "k763_pass": k763_pass,
    }


# ============================================================================
# Phase 2: Soft Routing Accuracy (K764)
# ============================================================================

def phase_soft_routing():
    """Verify Theorem 2: projection geometry produces domain-discriminative weights.

    For each validation text, compute soft routing weights:
      w_i(h) = ||h @ A_i^T||_2 / sum_j ||h @ A_j^T||_2

    Then check if argmax(w) matches the true domain.
    """
    log("\n=== Phase 2: Soft Routing Accuracy (K764) ===")

    from mlx_lm import load

    model, tokenizer = load(MODEL_ID)
    frozen_A = dict(np.load(str(SKELETON_PATH)))

    # Compute mean-pooled hidden states for each text
    # (same encoding as the ridge router uses)
    from pierre.pierre import encode

    correct = 0
    total = 0
    per_domain = {}
    all_weights = []

    for di, domain in enumerate(DOMAINS):
        texts = load_data(domain, "valid", N_ROUTE_TEST)
        domain_correct = 0

        for text in texts:
            tokens = tokenizer.encode(text)[:MAX_SEQ]
            if len(tokens) < 4:
                continue

            # Get mean-pooled hidden state
            h = encode(model, mx.array(tokens)[None, :])  # (1, d)
            h = h.squeeze(0)  # (d,)
            mx.eval(h)

            # Compute projection norms for each domain adapter
            # Use layer 0 A-matrices as representative (all layers have same structure)
            # Actually, for true soft routing, we should use ALL layers
            # But the "wall" interpretation uses the A-matrices directly on hidden states
            # Since hidden states at the final layer best represent domain semantics,
            # we project through a representative set of A-matrices

            # Aggregate projection norms across all modules at a representative layer
            # (layer 14, middle of the model -- hidden states are most domain-discriminative)
            proj_norms = []
            for dj in range(N_DOMAINS):
                # Sum squared projection norms across all modules
                norm_sq_sum = 0.0
                n_modules = 0
                for key in ADAPTER_TARGETS:
                    ak = f"layer_14_{key}_domain_{dj}"
                    if ak not in frozen_A:
                        continue
                    A = mx.array(frozen_A[ak]).astype(mx.float32)  # (in_features, rank)

                    # Project h onto A-subspace
                    # h is (d,), A is (in_features, rank)
                    # For attention/MLP inputs, in_features = d = 2560
                    # For MLP down_proj, in_features = 6912 (hidden dim)
                    if A.shape[0] != h.shape[0]:
                        continue  # skip dimension mismatch
                    proj = h @ A  # (rank,)
                    norm_sq = mx.sum(proj * proj).item()
                    norm_sq_sum += norm_sq
                    n_modules += 1

                if n_modules > 0:
                    proj_norms.append(math.sqrt(norm_sq_sum / n_modules))
                else:
                    proj_norms.append(0.0)

            # Normalize to get soft routing weights
            total_norm = sum(proj_norms)
            if total_norm > 0:
                weights = [n / total_norm for n in proj_norms]
            else:
                weights = [1.0 / N_DOMAINS] * N_DOMAINS

            predicted = int(np.argmax(weights))
            if predicted == di:
                correct += 1
                domain_correct += 1
            total += 1
            all_weights.append({
                "true_domain": domain,
                "predicted": DOMAINS[predicted],
                "weights": {DOMAINS[j]: round(w, 4) for j, w in enumerate(weights)},
                "correct": predicted == di,
            })

        domain_acc = domain_correct / len(texts) if texts else 0
        per_domain[domain] = {
            "accuracy": round(domain_acc, 3),
            "n_samples": len(texts),
        }
        log(f"  {domain}: {domain_acc:.1%} ({domain_correct}/{len(texts)})")

    overall_acc = correct / total if total > 0 else 0
    log(f"  Overall routing accuracy: {overall_acc:.1%} ({correct}/{total})")
    log(f"  K764: {'PASS' if overall_acc >= 0.60 else 'FAIL'} (threshold 60%)")

    # Also compute mean weight for correct domain vs others
    correct_domain_weight = []
    other_domain_weight = []
    for entry in all_weights:
        true = entry["true_domain"]
        w_true = entry["weights"][true]
        w_others = [v for k, v in entry["weights"].items() if k != true]
        correct_domain_weight.append(w_true)
        other_domain_weight.append(float(np.mean(w_others)))

    log(f"  Mean weight on correct domain: {np.mean(correct_domain_weight):.3f}")
    log(f"  Mean weight on other domains:  {np.mean(other_domain_weight):.3f}")

    cleanup(model, tokenizer)
    del frozen_A
    gc.collect()
    mx.clear_cache()

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_domain": per_domain,
        "n_total": total,
        "n_correct": correct,
        "mean_correct_domain_weight": round(float(np.mean(correct_domain_weight)), 4),
        "mean_other_domain_weight": round(float(np.mean(other_domain_weight)), 4),
        "k764_pass": overall_acc >= 0.60,
        "sample_weights": all_weights[:5],  # first 5 for inspection
    }


# ============================================================================
# Phase 3: PPL Measurement (K765)
# ============================================================================

def _compute_ppl(model, tokenizer, texts, max_seq=256):
    """Compute perplexity on a list of texts."""
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text)[:max_seq]
        if len(tokens) < 4:
            continue
        input_ids = mx.array(tokens)[None, :]
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        loss = nn.losses.cross_entropy(logits, targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.shape[1]
        del logits, loss

    if total_tokens == 0:
        return float('inf')
    return math.exp(total_loss / total_tokens)


def phase_ppl():
    """Verify K765: Room model PPL within 10% of v3 N=5 uniform baseline.

    v3 uniform composition = compose_adapters(all 5, uniform weights) then attach.
    Room model = inject W_combined (sum of all 5 deltas).

    These SHOULD be mathematically equivalent (Theorem 1), so PPL should match
    exactly (modulo bf16 quantization of W_combined).
    """
    log("\n=== Phase 3: PPL Measurement (K765) ===")

    from mlx_lm import load
    from pierre.pierre import RuntimeLoRA, compose_adapters, attach_adapter

    # Step 1: Base model PPL
    log("  Step 1: Base model PPL")
    model, tokenizer = load(MODEL_ID)
    frozen_A = dict(np.load(str(SKELETON_PATH)))

    base_ppls = {}
    for domain in DOMAINS:
        texts = load_data(domain, "valid", N_VAL)
        ppl = _compute_ppl(model, tokenizer, texts)
        base_ppls[domain] = round(ppl, 3)
        log(f"    Base {domain}: PPL = {ppl:.3f}")

    cleanup(model, tokenizer)

    # Step 2: v3-style uniform composition PPL (attach composed adapter)
    log("  Step 2: v3 uniform composition PPL")
    model, tokenizer = load(MODEL_ID)

    # Compose all 5 adapters with uniform weights
    adapter_Bs = []
    for di, domain in enumerate(DOMAINS):
        path = SFT_SOURCE / domain / "adapter.npz"
        adapter_Bs.append(dict(mx.load(str(path))))
    composed_B = compose_adapters(adapter_Bs)

    # Attach with domain 0's A-matrices (uniform composition uses same A)
    # Wait -- compose_adapters averages B-matrices. Then attach uses ONE domain's A.
    # This is NOT the same as summing all deltas.
    # v3 uniform = alpha * (x @ A_0) @ mean(B_i) with norm rescaling
    # Room model = sum_i alpha * (x @ A_i) @ B_i
    # These are DIFFERENT operations!
    #
    # The correct v3 comparison is: attach each adapter separately and sum.
    # But v3 doesn't do that. v3 either routes to one adapter (top-1)
    # or composes B-matrices (NRE average) and uses one A.
    #
    # The Room Model is closer to "all adapters active simultaneously."
    # The fairest comparison is: for each text, what PPL does the Room Model
    # achieve vs the best-routed single adapter?

    # Actually, let's compute BOTH:
    # (a) Room model: W_combined = sum of all deltas
    # (b) v3 single-adapter: routed top-1 adapter
    # (c) v3 composed: NRE average B, using domain 0's A

    # First: v3 composed (NRE average)
    v3_composed_ppls = {}
    n_attached = attach_adapter(model, frozen_A, composed_B, 0, LORA_SCALE)
    log(f"    v3 composed: attached {n_attached} modules")
    for domain in DOMAINS:
        texts = load_data(domain, "valid", N_VAL)
        ppl = _compute_ppl(model, tokenizer, texts)
        v3_composed_ppls[domain] = round(ppl, 3)
        log(f"    v3-composed {domain}: PPL = {ppl:.3f}")

    from pierre.pierre import detach_adapters
    detach_adapters(model)
    cleanup(model, tokenizer)
    del composed_B, adapter_Bs
    gc.collect()
    mx.clear_cache()

    # Step 3: Room model PPL (inject W_combined)
    log("  Step 3: Room model PPL")
    model, tokenizer = load(MODEL_ID)
    frozen_A_data = dict(np.load(str(SKELETON_PATH)))
    adapter_Bs_data = {}
    for di, domain in enumerate(DOMAINS):
        path = SFT_SOURCE / domain / "adapter.npz"
        adapter_Bs_data[di] = dict(mx.load(str(path)))

    # Compute and inject W_combined for each module
    from mlx.utils import tree_unflatten

    n_injected = 0
    w_combined_bytes = 0
    n_layers = len(model.model.layers)

    for li in range(n_layers):
        layer = model.model.layers[li]
        updates = []

        for key in ADAPTER_TARGETS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            in_features, out_features = _get_module_dims(
                frozen_A_data, adapter_Bs_data, li, key, N_DOMAINS)
            if in_features is None:
                continue

            # Compute W_combined in float32 for precision
            w_combined = mx.zeros((in_features, out_features), dtype=mx.float32)

            for di in range(N_DOMAINS):
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"
                if ak not in frozen_A_data or bk not in adapter_Bs_data[di]:
                    continue

                A = mx.array(frozen_A_data[ak]).astype(mx.float32)
                B = adapter_Bs_data[di][bk].astype(mx.float32)
                delta_W = LORA_SCALE * (A @ B)
                mx.eval(delta_W)
                w_combined = w_combined + delta_W

            mx.eval(w_combined)
            updates.append((key, RoomLinear(m, w_combined)))
            w_combined_bytes += in_features * out_features * 2  # bf16
            n_injected += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))

    mx.eval(model.parameters())
    log(f"    Injected {n_injected} RoomLinear modules")
    log(f"    W_combined total: {w_combined_bytes / 1e9:.2f} GB (bf16)")

    room_ppls = {}
    for domain in DOMAINS:
        texts = load_data(domain, "valid", N_VAL)
        ppl = _compute_ppl(model, tokenizer, texts)
        room_ppls[domain] = round(ppl, 3)
        log(f"    Room {domain}: PPL = {ppl:.3f}")

    cleanup(model, tokenizer)
    del frozen_A_data, adapter_Bs_data
    gc.collect()
    mx.clear_cache()

    # Step 4: v3 single-adapter (best-routed) PPL for each domain
    log("  Step 4: v3 single-adapter (oracle) PPL")
    v3_single_ppls = {}
    frozen_A_oracle = dict(np.load(str(SKELETON_PATH)))
    for di, domain in enumerate(DOMAINS):
        model, tokenizer = load(MODEL_ID)
        adapter_B = dict(mx.load(str(SFT_SOURCE / domain / "adapter.npz")))
        n = attach_adapter(model, frozen_A_oracle, adapter_B, di, LORA_SCALE)
        texts = load_data(domain, "valid", N_VAL)
        ppl = _compute_ppl(model, tokenizer, texts)
        v3_single_ppls[domain] = round(ppl, 3)
        log(f"    v3-single {domain}: PPL = {ppl:.3f}")
        detach_adapters(model)
        cleanup(model, tokenizer)
        del adapter_B
        gc.collect()
        mx.clear_cache()
    del frozen_A_oracle

    # Analysis
    mean_base = float(np.mean(list(base_ppls.values())))
    mean_room = float(np.mean(list(room_ppls.values())))
    mean_v3_composed = float(np.mean(list(v3_composed_ppls.values())))
    mean_v3_single = float(np.mean(list(v3_single_ppls.values())))

    # K765: room model PPL within 10% of v3 composed baseline
    # Since room model applies ALL adapters simultaneously, the fairest
    # comparison is v3-composed (also all adapters, different method)
    ppl_ratio = mean_room / mean_v3_composed if mean_v3_composed > 0 else float('inf')
    k765_pass = ppl_ratio <= 1.10

    log(f"\n  PPL Summary:")
    log(f"    Base:         {mean_base:.3f}")
    log(f"    v3-single:    {mean_v3_single:.3f} (oracle routed)")
    log(f"    v3-composed:  {mean_v3_composed:.3f} (NRE uniform)")
    log(f"    Room model:   {mean_room:.3f} (all deltas summed)")
    log(f"    Room/v3-composed ratio: {ppl_ratio:.4f}")
    log(f"    K765: {'PASS' if k765_pass else 'FAIL'} (threshold 1.10)")

    return {
        "base_ppls": base_ppls,
        "v3_single_ppls": v3_single_ppls,
        "v3_composed_ppls": v3_composed_ppls,
        "room_ppls": room_ppls,
        "mean_base": round(mean_base, 3),
        "mean_v3_single": round(mean_v3_single, 3),
        "mean_v3_composed": round(mean_v3_composed, 3),
        "mean_room": round(mean_room, 3),
        "ppl_ratio_vs_composed": round(ppl_ratio, 4),
        "n_injected_modules": n_injected,
        "w_combined_gb": round(w_combined_bytes / 1e9, 3),
        "k765_pass": k765_pass,
    }


# ============================================================================
# Phase 4: Speed Measurement
# ============================================================================

def phase_speed():
    """Measure tok/s for Room model vs base vs v3-single.

    Theorem 3 predicts ~40-50 tok/s for Room model (bandwidth-limited).
    This phase measures actual performance honestly.
    """
    log("\n=== Phase 4: Speed Measurement ===")

    from mlx_lm import load, generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler
    from pierre.pierre import attach_adapter, detach_adapters

    prompt = "### Instruction:\nExplain photosynthesis.\n\n### Response:\n"
    sampler = make_sampler(temp=0.0)
    gen_tokens = 128

    def bench(model, tok, label, warmup=3, trials=5):
        for _ in range(warmup):
            mlx_generate(model, tok, prompt=prompt, max_tokens=32,
                         sampler=sampler, verbose=False)
        mx.reset_peak_memory()
        times = []
        for _ in range(trials):
            t0 = time.time()
            out = mlx_generate(model, tok, prompt=prompt, max_tokens=gen_tokens,
                               sampler=sampler, verbose=False)
            dt = time.time() - t0
            n_out = len(tok.encode(out)) - len(tok.encode(prompt))
            times.append({"s": dt, "toks": n_out})
        tps = sum(t["toks"] for t in times) / sum(t["s"] for t in times)
        peak = mx.get_peak_memory() / 1e9
        log(f"  {label}: {tps:.1f} tok/s, peak={peak:.2f}GB")
        return round(tps, 1), round(peak, 2)

    # (a) Base model
    model, tok = load(MODEL_ID)
    base_tps, base_mem = bench(model, tok, "Base (native BitLinear)")
    cleanup(model, tok)

    # (b) v3 single adapter
    model, tok = load(MODEL_ID)
    frozen_A = dict(np.load(str(SKELETON_PATH)))
    adapter_B = dict(mx.load(str(SFT_SOURCE / "medical" / "adapter.npz")))
    attach_adapter(model, frozen_A, adapter_B, 0, LORA_SCALE)
    v3_tps, v3_mem = bench(model, tok, "v3 single adapter (factored)")
    detach_adapters(model)
    cleanup(model, tok)
    del adapter_B
    gc.collect()
    mx.clear_cache()

    # (c) Room model (all 5 adapters summed)
    model, tok = load(MODEL_ID)
    frozen_A_data = dict(np.load(str(SKELETON_PATH)))
    adapter_Bs_data = {}
    for di, domain in enumerate(DOMAINS):
        adapter_Bs_data[di] = dict(mx.load(str(SFT_SOURCE / domain / "adapter.npz")))

    from mlx.utils import tree_unflatten
    n_injected = 0
    for li in range(len(model.model.layers)):
        layer = model.model.layers[li]
        updates = []
        for key in ADAPTER_TARGETS:
            m = layer
            for part in key.split("."):
                m = getattr(m, part, None)
                if m is None:
                    break
            if m is None:
                continue

            in_features, out_features = _get_module_dims(
                frozen_A_data, adapter_Bs_data, li, key, N_DOMAINS)
            if in_features is None:
                continue
            w_combined = mx.zeros((in_features, out_features), dtype=mx.float32)

            for di in range(N_DOMAINS):
                ak = f"layer_{li}_{key}_domain_{di}"
                bk = f"model.layers.{li}.{key}.lora_b"
                if ak not in frozen_A_data or bk not in adapter_Bs_data[di]:
                    continue
                A = mx.array(frozen_A_data[ak]).astype(mx.float32)
                B = adapter_Bs_data[di][bk].astype(mx.float32)
                w_combined = w_combined + LORA_SCALE * (A @ B)

            mx.eval(w_combined)
            updates.append((key, RoomLinear(m, w_combined)))
            n_injected += 1

        if updates:
            layer.update_modules(tree_unflatten(updates))
    mx.eval(model.parameters())

    room_tps, room_mem = bench(model, tok, f"Room model ({n_injected} modules)")
    cleanup(model, tok)
    del frozen_A_data, adapter_Bs_data, frozen_A
    gc.collect()
    mx.clear_cache()

    log(f"\n  Speed Summary:")
    log(f"    Base:      {base_tps} tok/s, {base_mem} GB")
    log(f"    v3-single: {v3_tps} tok/s, {v3_mem} GB")
    log(f"    Room:      {room_tps} tok/s, {room_mem} GB")
    log(f"    Predicted: 40-50 tok/s (Theorem 3)")

    return {
        "base_tps": base_tps,
        "base_mem_gb": base_mem,
        "v3_single_tps": v3_tps,
        "v3_single_mem_gb": v3_mem,
        "room_tps": room_tps,
        "room_mem_gb": room_mem,
        "predicted_room_tps": "40-50",
        "n_injected": n_injected,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    log("Room Model POC: Pre-Summed W_combined with Automatic Soft Routing")
    log("=" * 70)
    mx.random.seed(SEED)
    log_memory("start")

    # Phase 1: Mathematical equivalence
    r1 = phase_equivalence()
    log_memory("after-equivalence")

    # Phase 2: Soft routing accuracy
    r2 = phase_soft_routing()
    log_memory("after-routing")

    # Phase 3: PPL measurement
    r3 = phase_ppl()
    log_memory("after-ppl")

    # Phase 4: Speed measurement
    r4 = phase_speed()
    log_memory("after-speed")

    # Kill criteria assessment
    k763 = r1["k763_pass"]
    k764 = r2["k764_pass"]
    k765 = r3["k765_pass"]

    results = {
        "experiment": "room_model_poc",
        "total_time_s": round(time.time() - t0, 1),
        "equivalence": r1,
        "soft_routing": r2,
        "ppl": r3,
        "speed": r4,
        "kill_criteria": {
            "K763": {
                "pass": k763,
                "value": r1["logit_mse"],
                "threshold": 1e-6,
                "description": "W_combined logit MSE vs sequential-sum < 1e-6",
            },
            "K764": {
                "pass": k764,
                "value": r2["overall_accuracy"],
                "threshold": 0.60,
                "description": "Soft routing domain accuracy >= 60%",
            },
            "K765": {
                "pass": k765,
                "value": r3["ppl_ratio_vs_composed"],
                "threshold": 1.10,
                "description": "Room PPL within 10% of v3 composed baseline",
            },
        },
        "all_pass": k763 and k764 and k765,
        "predictions_vs_actual": {
            "per_module_mse": {
                "predicted": "~0 (Theorem 1, float32)",
                "actual": r1["per_module_max_mse"],
            },
            "logit_mse": {
                "predicted": "< 1e-6 (Theorem 1, float32 accumulation)",
                "actual": r1["logit_mse"],
            },
            "routing_accuracy": {
                "predicted": ">= 60% (Theorem 2)",
                "actual": r2["overall_accuracy"],
            },
            "ppl_ratio": {
                "predicted": "<= 1.10 (Theorem 1 implies ~1.0)",
                "actual": r3["ppl_ratio_vs_composed"],
            },
            "speed_tok_s": {
                "predicted": "40-50 (Theorem 3)",
                "actual": r4["room_tps"],
            },
            "memory_gb": {
                "predicted": "~5.8 GB",
                "actual": r4["room_mem_gb"],
            },
        },
        "architecture": {
            "model": MODEL_ID,
            "n_domains": N_DOMAINS,
            "domains": DOMAINS,
            "lora_scale": LORA_SCALE,
            "lora_rank": 16,
            "n_layers": 30,
            "adapter_targets": ADAPTER_TARGETS,
            "n_modules": 210,
        },
    }

    log(f"\n{'='*70}")
    log("Kill Criteria:")
    for k, v in results["kill_criteria"].items():
        status = "PASS" if v["pass"] else "FAIL"
        log(f"  {k}: {status} -- {v['description']}: {v['value']}")

    log(f"\nPredictions vs Actual:")
    for k, v in results["predictions_vs_actual"].items():
        log(f"  {k}: predicted={v['predicted']}, actual={v['actual']}")

    verdict = "ALL PASS" if results["all_pass"] else "KILLED"
    log(f"\n{verdict} in {results['total_time_s']:.0f}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    log(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
