#!/usr/bin/env python3
"""24x24 Cross-Domain PPL Matrix: Are Grassmannian adapters domain-specific?

For each of 24 adapters applied to each of 24 domain validation sets, compute PPL.
Builds a 24x24 matrix to determine adapter specialization vs interchangeability.

Also computes base PPL (no adapter) for each domain as reference.

Kill criteria:
  K599: Diagonal dominance ratio < 1.05 (adapters interchangeable at PPL level)
  K600: Fewer than 12/24 domains show lower PPL with own adapter vs mean of others
  K601: Mean PPL improvement < 20% with correct loading (adapters don't work)

Platform: Apple M5 Pro 48GB, MLX.
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
from mlx.utils import tree_unflatten

# Memory safety (MANDATORY per CODING_GUIDELINES)
device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Source data from the 25-domain training experiment
REAL_DATA_DIR = EXPERIMENT_DIR.parent / "real_data_25_domain_adapters"
ADAPTERS_DIR = REAL_DATA_DIR / "adapters"
DATA_DIR = REAL_DATA_DIR / "data"
SKELETON_PATH = ADAPTERS_DIR / "grassmannian_skeleton_n24.npz"

MODEL_ID = "microsoft/BitNet-b1.58-2B-4T"
LORA_RANK = 16
LORA_SCALE = 20.0
MAX_SEQ_LENGTH = 256
VAL_BATCHES = 20  # per domain per adapter evaluation
SEED = 42

# Domain ordering MUST match training experiment (not alphabetical)
TRAINING_DOMAIN_ORDER = [
    "medical", "code", "math", "legal", "finance",
    "science", "history", "philosophy", "creative_writing", "cooking",
    "health_fitness", "psychology", "education", "engineering", "agriculture",
    "environmental", "politics", "economics", "sociology", "linguistics",
    "cybersecurity", "marketing", "sports", "music",
]

DOMAINS = [
    d for d in TRAINING_DOMAIN_ORDER
    if (ADAPTERS_DIR / d / "adapter.npz").exists() and (DATA_DIR / d).exists()
]
N_DOMAINS = len(DOMAINS)
SKELETON_IDX = {d: i for i, d in enumerate(DOMAINS)}

TARGET_KEYS = [
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
]


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
# Model utilities (reused from fix_grassmannian_loading_retest)
# ===========================================================================
from mlx_lm import load
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
    """Replace BitLinear with nn.Linear for differentiable LoRA."""
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


class TernaryLoRALinear(nn.Module):
    """LoRA with frozen Grassmannian A and STE-ternary B."""
    def __init__(self, base_linear: nn.Linear, rank: int = 16,
                 scale: float = 20.0, a_init: mx.array = None):
        super().__init__()
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        self.linear = base_linear

        if a_init is not None:
            self.lora_a = a_init
        else:
            s = 1.0 / math.sqrt(in_features)
            self.lora_a = mx.random.uniform(
                low=-s, high=s, shape=(in_features, rank)
            )

        self.lora_b = mx.zeros((rank, out_features))
        self.scale = scale
        self.rank = rank

        self.linear.freeze()
        self.freeze(keys=["lora_a"], strict=False)

    def __call__(self, x):
        base_out = self.linear(x)
        b = self.lora_b
        # STE ternary quantization of B (same as training)
        alpha = mx.mean(mx.abs(b))
        b_scaled = b / (alpha + 1e-7)
        b_q = mx.clip(mx.round(b_scaled), -1, 1) * alpha
        b_ste = b + mx.stop_gradient(b_q - b)
        lora_out = (x @ self.lora_a) @ b_ste * self.scale
        return base_out + lora_out


def apply_ternary_lora(model, rank, scale, a_matrices):
    """Apply TernaryLoRALinear to all target projections with Grassmannian A."""
    count = 0
    for li, layer in enumerate(model.model.layers):
        lora_updates = []
        for key in TARGET_KEYS:
            parts = key.split(".")
            module = layer
            for part in parts:
                module = getattr(module, part, None)
                if module is None:
                    break
            if module is None or not isinstance(module, nn.Linear):
                continue

            a_key = (li, key)
            a_mx = None
            if a_key in a_matrices:
                a_mx = mx.array(a_matrices[a_key]).astype(mx.bfloat16)

            lora = TernaryLoRALinear(module, rank=rank, scale=scale, a_init=a_mx)
            lora_updates.append((key, lora))
            count += 1

        if lora_updates:
            layer.update_modules(tree_unflatten(lora_updates))

    log(f"  Applied TernaryLoRA (r={rank}) to {count} layers")
    return model


def set_lora_a(model, skeleton, domain_idx, n_layers):
    """Set A matrices from skeleton for a given domain index.

    CRITICAL: domain_idx is the position in DOMAINS list (= training order position).
    The skeleton key format is layer_{li}_{key}_domain_{domain_idx}.
    """
    for li in range(n_layers):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_{domain_idx}"
            if skey in skeleton:
                a_mx = mx.array(skeleton[skey]).astype(mx.bfloat16)
                parts = key.split(".")
                module = model.model.layers[li]
                for part in parts:
                    module = getattr(module, part)
                if isinstance(module, TernaryLoRALinear):
                    module.lora_a = a_mx


def zero_b_params(model):
    """Zero all lora_b matrices."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, TernaryLoRALinear):
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def load_adapter(path: Path) -> dict:
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict):
    model.update(tree_unflatten(list(adapter_params.items())))


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


def compute_ppl(model, tokenizer, texts, max_batches=20):
    """Compute PPL on a set of texts."""
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
    return math.exp(min(total_loss / total_tokens, 100))


# ===========================================================================
# Phase 1: Compute base PPL (no adapter) for all domains
# ===========================================================================
def phase_base_ppl():
    """Compute base PPL for all 24 domains without any adapter."""
    log("\n" + "=" * 70)
    log("[Phase 1] Base PPL (no adapter) for all domains")
    log("=" * 70)

    t0 = time.time()

    # Load skeleton for initial A matrices (needed for model structure)
    skeleton = dict(np.load(str(SKELETON_PATH)))
    a_matrices = {}
    for li in range(30):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    # Zero B weights -> pure base model output
    zero_b_params(model)

    base_ppls = {}
    for domain in DOMAINS:
        texts = load_domain_texts(domain, split="valid")
        ppl = compute_ppl(model, tokenizer, texts, max_batches=VAL_BATCHES)
        base_ppls[domain] = round(ppl, 4)
        log(f"  {domain:20s}: base_ppl={ppl:.2f}")

    avg_base = sum(base_ppls.values()) / len(base_ppls)
    log(f"\n  Average base PPL: {avg_base:.4f}")
    log(f"  Phase 1 time: {time.time() - t0:.1f}s")
    log_memory("post-base")

    cleanup(model, tokenizer)
    del skeleton
    return base_ppls


# ===========================================================================
# Phase 2: Build 24x24 cross-domain PPL matrix
# ===========================================================================
def phase_cross_domain_matrix(base_ppls):
    """For each adapter, evaluate on all 24 domains. Build NxN matrix."""
    log("\n" + "=" * 70)
    log("[Phase 2] Building 24x24 cross-domain PPL matrix")
    log("=" * 70)

    t0 = time.time()

    # Load skeleton
    skeleton = dict(np.load(str(SKELETON_PATH)))

    # Build initial A matrices (domain 0)
    a_matrices = {}
    for li in range(30):
        for key in TARGET_KEYS:
            skey = f"layer_{li}_{key}_domain_0"
            if skey in skeleton:
                a_matrices[(li, key)] = skeleton[skey]

    model, tokenizer = load(MODEL_ID)
    log(f"  Model loaded in {time.time() - t0:.1f}s")
    model = replace_bitlinear_with_linear(model)
    model = apply_ternary_lora(model, LORA_RANK, LORA_SCALE, a_matrices)
    model.freeze()

    n_layers = len(model.model.layers)

    # Pre-load all validation texts (small in memory)
    all_val_texts = {}
    for domain in DOMAINS:
        all_val_texts[domain] = load_domain_texts(domain, split="valid")
    log(f"  Loaded validation texts for {len(all_val_texts)} domains")

    # Pre-load all adapter B weights
    all_adapter_params = {}
    for domain in DOMAINS:
        all_adapter_params[domain] = load_adapter(ADAPTERS_DIR / domain)
    log(f"  Loaded {len(all_adapter_params)} adapter B-weight sets")

    # Build the matrix: ppl_matrix[adapter_domain][eval_domain] = PPL
    ppl_matrix = {}

    for ai, adapter_domain in enumerate(DOMAINS):
        log(f"\n  --- Adapter {ai+1}/{N_DOMAINS}: {adapter_domain} ---")

        # Set correct A matrices for this adapter's domain
        set_lora_a(model, skeleton, ai, n_layers)
        # Load this adapter's trained B weights
        zero_b_params(model)
        apply_adapter_weights(model, all_adapter_params[adapter_domain])
        mx.eval(model.parameters())

        ppl_matrix[adapter_domain] = {}

        for ei, eval_domain in enumerate(DOMAINS):
            texts = all_val_texts[eval_domain]
            ppl = compute_ppl(model, tokenizer, texts, max_batches=VAL_BATCHES)
            ppl_matrix[adapter_domain][eval_domain] = round(ppl, 4)

        # Log diagonal entry and a few off-diagonals
        diag_ppl = ppl_matrix[adapter_domain][adapter_domain]
        off_diag = [ppl_matrix[adapter_domain][d] for d in DOMAINS if d != adapter_domain]
        avg_off = sum(off_diag) / len(off_diag) if off_diag else 0
        log(f"  diagonal={diag_ppl:.2f}  avg_off_diag={avg_off:.2f}  ratio={avg_off/diag_ppl:.3f}")

        if (ai + 1) % 6 == 0:
            log_memory(f"after-{ai+1}-adapters")

    elapsed = time.time() - t0
    log(f"\n  Phase 2 time: {elapsed:.1f}s")
    log_memory("post-matrix")

    cleanup(model, tokenizer)
    del skeleton, all_adapter_params
    return ppl_matrix


# ===========================================================================
# Phase 3: Analyze matrix and compute metrics
# ===========================================================================
def phase_analyze(ppl_matrix, base_ppls):
    """Compute diagonal dominance, specificity metrics, domain clusters."""
    log("\n" + "=" * 70)
    log("[Phase 3] Analyzing cross-domain PPL matrix")
    log("=" * 70)

    # Diagonal dominance ratio per domain
    ddr_per_domain = {}
    diagonal_wins = 0

    for domain in DOMAINS:
        diag = ppl_matrix[domain][domain]
        off_diag = [ppl_matrix[adapter][domain] for adapter in DOMAINS if adapter != domain]
        avg_off = sum(off_diag) / len(off_diag)
        ddr = avg_off / diag if diag > 0 else 0
        ddr_per_domain[domain] = round(ddr, 4)

        if diag < avg_off:
            diagonal_wins += 1

        log(f"  {domain:20s}: diag={diag:.2f} avg_off={avg_off:.2f} DDR={ddr:.3f} {'WIN' if diag < avg_off else 'LOSE'}")

    global_ddr = sum(ddr_per_domain.values()) / len(ddr_per_domain)
    log(f"\n  Global DDR: {global_ddr:.4f}")
    log(f"  Diagonal wins: {diagonal_wins}/{N_DOMAINS}")

    # PPL improvement with correct adapter vs base
    improvements = {}
    for domain in DOMAINS:
        base = base_ppls[domain]
        adapted = ppl_matrix[domain][domain]
        imp = (base - adapted) / base * 100 if base > 0 else 0
        improvements[domain] = round(imp, 2)

    avg_improvement = sum(improvements.values()) / len(improvements)
    log(f"  Avg improvement (own adapter vs base): {avg_improvement:.1f}%")

    # Per-adapter: how many domains does each adapter "help" (beat base)?
    adapter_generality = {}
    for adapter in DOMAINS:
        helped = 0
        for eval_d in DOMAINS:
            if ppl_matrix[adapter][eval_d] < base_ppls[eval_d]:
                helped += 1
        adapter_generality[adapter] = helped

    avg_generality = sum(adapter_generality.values()) / len(adapter_generality)
    log(f"  Avg adapter generality (domains helped): {avg_generality:.1f}/{N_DOMAINS}")

    # Find best adapter for each domain (not necessarily diagonal)
    best_adapter_per_domain = {}
    for eval_d in DOMAINS:
        best_ppl = float("inf")
        best_adapter = None
        for adapter in DOMAINS:
            if ppl_matrix[adapter][eval_d] < best_ppl:
                best_ppl = ppl_matrix[adapter][eval_d]
                best_adapter = adapter
        best_adapter_per_domain[eval_d] = {
            "best_adapter": best_adapter,
            "best_ppl": round(best_ppl, 4),
            "own_ppl": round(ppl_matrix[eval_d][eval_d], 4),
            "is_diagonal": best_adapter == eval_d,
        }
        if best_adapter != eval_d:
            log(f"  {eval_d}: best adapter is {best_adapter} (ppl={best_ppl:.2f}) not self (ppl={ppl_matrix[eval_d][eval_d]:.2f})")

    diagonal_is_best = sum(1 for v in best_adapter_per_domain.values() if v["is_diagonal"])
    log(f"  Domains where own adapter is best: {diagonal_is_best}/{N_DOMAINS}")

    # Compute row variance (adapter applied to all domains)
    # High variance = adapter strongly favors some domains
    row_variance = {}
    for adapter in DOMAINS:
        ppls = [ppl_matrix[adapter][d] for d in DOMAINS]
        mean_ppl = sum(ppls) / len(ppls)
        var = sum((p - mean_ppl)**2 for p in ppls) / len(ppls)
        row_variance[adapter] = round(var, 4)

    # Compute column variance (domain evaluated with all adapters)
    # High variance = domain is sensitive to adapter choice
    col_variance = {}
    for eval_d in DOMAINS:
        ppls = [ppl_matrix[adapter][eval_d] for adapter in DOMAINS]
        mean_ppl = sum(ppls) / len(ppls)
        var = sum((p - mean_ppl)**2 for p in ppls) / len(ppls)
        col_variance[eval_d] = round(var, 4)

    return {
        "ddr_per_domain": ddr_per_domain,
        "global_ddr": round(global_ddr, 4),
        "diagonal_wins": diagonal_wins,
        "diagonal_is_best": diagonal_is_best,
        "improvements_pct": improvements,
        "avg_improvement_pct": round(avg_improvement, 2),
        "adapter_generality": adapter_generality,
        "avg_generality": round(avg_generality, 2),
        "best_adapter_per_domain": best_adapter_per_domain,
        "row_variance": row_variance,
        "col_variance": col_variance,
    }


# ===========================================================================
# Main
# ===========================================================================
def main():
    t_start = time.time()
    log(f"24x24 Cross-Domain PPL Matrix")
    log(f"Domains: {N_DOMAINS} -- {', '.join(DOMAINS)}")
    log_memory("start")

    # Phase 1: Base PPL
    base_ppls = phase_base_ppl()

    # Phase 2: Build 24x24 matrix
    ppl_matrix = phase_cross_domain_matrix(base_ppls)

    # Phase 3: Analyze
    analysis = phase_analyze(ppl_matrix, base_ppls)

    # Kill criteria assessment
    k599_pass = analysis["global_ddr"] >= 1.05
    k600_pass = analysis["diagonal_wins"] >= 12
    k601_pass = analysis["avg_improvement_pct"] >= 20.0

    peak_gb = mx.get_peak_memory() / 1e9

    results = {
        "experiment": "cross_domain_ppl_matrix_n24",
        "model": MODEL_ID,
        "n_domains": N_DOMAINS,
        "domains": DOMAINS,
        "lora_rank": LORA_RANK,
        "lora_scale": LORA_SCALE,
        "val_batches": VAL_BATCHES,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "base_ppls": base_ppls,
        "ppl_matrix": ppl_matrix,
        **analysis,
        "kill_criteria": {
            "K599_diagonal_dominance_ge_1.05": "PASS" if k599_pass else "FAIL",
            "K599_value": analysis["global_ddr"],
            "K600_diagonal_wins_ge_12": "PASS" if k600_pass else "FAIL",
            "K600_value": analysis["diagonal_wins"],
            "K601_avg_improvement_ge_20pct": "PASS" if k601_pass else "FAIL",
            "K601_value": analysis["avg_improvement_pct"],
            "peak_memory_gb": round(peak_gb, 2),
        },
        "total_time_s": round(time.time() - t_start, 1),
    }

    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults saved to {RESULTS_FILE}")

    # Summary
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"  Global DDR:          {analysis['global_ddr']:.4f}")
    log(f"  Diagonal wins:       {analysis['diagonal_wins']}/{N_DOMAINS}")
    log(f"  Diagonal is best:    {analysis['diagonal_is_best']}/{N_DOMAINS}")
    log(f"  Avg improvement:     {analysis['avg_improvement_pct']:.1f}%")
    log(f"  Avg generality:      {analysis['avg_generality']:.1f}/{N_DOMAINS}")
    log(f"  Peak memory:         {peak_gb:.1f} GB")
    log(f"  Total time:          {results['total_time_s']:.0f}s")

    log(f"\n  K599 (DDR >= 1.05):           {'PASS' if k599_pass else 'FAIL'} ({analysis['global_ddr']:.4f})")
    log(f"  K600 (wins >= 12):            {'PASS' if k600_pass else 'FAIL'} ({analysis['diagonal_wins']}/24)")
    log(f"  K601 (improvement >= 20%):    {'PASS' if k601_pass else 'FAIL'} ({analysis['avg_improvement_pct']:.1f}%)")

    verdict = "SUPPORTED" if (k599_pass and k600_pass and k601_pass) else "KILLED" if (not k599_pass and not k600_pass) else "PARTIAL"
    log(f"\n  VERDICT: {verdict}")
    if k599_pass and k600_pass:
        log("  Adapters ARE domain-specific. Routing matters.")
    elif not k599_pass and not k600_pass:
        log("  Adapters are INTERCHANGEABLE. Routing is unnecessary.")
    else:
        log("  Mixed results. Adapters show partial specialization.")


if __name__ == "__main__":
    main()
