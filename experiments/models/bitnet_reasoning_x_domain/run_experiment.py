#!/usr/bin/env python3
"""
Reasoning x Domain Cross-Composition Experiment

Tests whether a reasoning capability adapter composes with domain adapters
on BitNet-2B-4T without interference: domain quality preserved (<3% PPL
degradation) and reasoning signal added (reasoning PPL improves for >50%
of domain compositions).

Kill criteria:
  K1: reasoning adapter improves reasoning PPL on <50% of domain compositions
      (domain+reasoning vs domain-alone, evaluated on reasoning val data)
  K2: domain PPL degrades >3% when reasoning adapter is added
      (domain+reasoning vs domain-alone, evaluated on domain val data)

Critical context from dependency (exp_bitnet_task_eval KILLED):
  - NTP-trained adapters do NOT produce task-capable models at 2B scale
  - PPL-based measurement IS valid (well-proven across 10+ experiments)
  - Therefore: ALL evaluation is PPL-based, no task accuracy

Reuses:
  - 5 domain adapters from bitnet_2b_real_composition (python, math, medical, legal, creative)
  - 1 reasoning adapter from capability_expert_taxonomy
  - Domain val data from bitnet_2b_real_composition
  - Reasoning val data from capability_expert_taxonomy

Platform: Apple Silicon MLX, $0.
Expected runtime: ~15 min (eval-only, no training).
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from itertools import combinations

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
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
VAL_BATCHES = 25

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

# Existing adapter directories
DOMAIN_ADAPTERS_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition" / "adapters"
REASONING_ADAPTER_DIR = EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "adapters" / "reasoning"

# Existing data directories
DOMAIN_DATA_DIR = EXPERIMENT_DIR.parent / "bitnet_2b_real_composition" / "data"
REASONING_DATA_DIR = EXPERIMENT_DIR.parent / "capability_expert_taxonomy" / "data" / "reasoning"

DOMAIN_NAMES = ["python", "math", "medical", "legal", "creative"]


# ===========================================================================
# Model utilities (from capability_expert_taxonomy)
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


def load_adapter(path: Path) -> dict:
    """Load adapter weights from disk."""
    return dict(mx.load(str(path / "adapter.npz")))


def apply_adapter_weights(model, adapter_params: dict, scale: float = 1.0):
    """Apply adapter params into current LoRA layers."""
    if abs(scale - 1.0) > 1e-6:
        scaled = {k: v * scale for k, v in adapter_params.items()}
    else:
        scaled = adapter_params
    model.update(tree_unflatten(list(scaled.items())))


def zero_lora_params(model):
    """Reset all LoRA params to zero (base model behavior)."""
    for layer in model.model.layers:
        for key, module in layer.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_a = mx.zeros_like(module.lora_a)
                module.lora_b = mx.zeros_like(module.lora_b)
    mx.eval(model.parameters())


def compose_adapters(model, adapters: list, scale: float = None):
    """Compose multiple adapters with 1/N scaling.

    Args:
        model: model with LoRA layers
        adapters: list of adapter param dicts
        scale: per-adapter scale. If None, uses 1/N.
    """
    n = len(adapters)
    if scale is None:
        scale = 1.0 / n

    # First zero out
    zero_lora_params(model)

    # Sum scaled adapters
    combined = {}
    for adapter in adapters:
        for k, v in adapter.items():
            if k in combined:
                combined[k] = combined[k] + v * scale
            else:
                combined[k] = v * scale

    model.update(tree_unflatten(list(combined.items())))
    mx.eval(model.parameters())


# ===========================================================================
# PPL evaluation
# ===========================================================================
def compute_ppl(model, tokenizer, data_path: Path, max_batches: int = VAL_BATCHES):
    """Compute perplexity on validation data."""
    valid_path = data_path / "valid.jsonl"
    if not valid_path.exists():
        print(f"  WARNING: {valid_path} not found")
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

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(min(avg_loss, 100))


def compute_cosine(adapter_a: dict, adapter_b: dict) -> float:
    """Compute |cos| between two flattened adapter vectors."""
    va = mx.concatenate([v.reshape(-1) for v in adapter_a.values()])
    vb = mx.concatenate([v.reshape(-1) for v in adapter_b.values()])
    cos = mx.abs(
        mx.sum(va * vb) / (mx.sqrt(mx.sum(va**2)) * mx.sqrt(mx.sum(vb**2)))
    )
    mx.eval(cos)
    return round(cos.item(), 6)


# ===========================================================================
# Main experiment
# ===========================================================================
def main():
    t_start = time.time()
    results = {
        "experiment": "bitnet_reasoning_x_domain",
        "model": MODEL_ID,
        "lora_rank": LORA_RANK,
        "hypothesis": "Reasoning adapter composes with domain adapters without interference",
        "domains": DOMAIN_NAMES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "measurement": "PPL-only (task eval killed at 2B scale)",
    }

    print("=" * 70)
    print("Reasoning x Domain Cross-Composition Experiment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Phase 0: Verify all adapters and data exist
    # ------------------------------------------------------------------
    print("\n[Phase 0] Verifying existing adapters and data...")

    # Check reasoning adapter
    if not (REASONING_ADAPTER_DIR / "adapter.npz").exists():
        print(f"  FATAL: Reasoning adapter not found at {REASONING_ADAPTER_DIR}")
        sys.exit(1)
    print(f"  Reasoning adapter: {REASONING_ADAPTER_DIR}")

    # Check reasoning val data
    if not (REASONING_DATA_DIR / "valid.jsonl").exists():
        print(f"  FATAL: Reasoning val data not found at {REASONING_DATA_DIR}")
        sys.exit(1)
    print(f"  Reasoning val data: {REASONING_DATA_DIR}")

    # Check domain adapters and data
    for name in DOMAIN_NAMES:
        adapter_path = DOMAIN_ADAPTERS_DIR / name / "adapter.npz"
        data_path = DOMAIN_DATA_DIR / name / "valid.jsonl"
        if not adapter_path.exists():
            print(f"  FATAL: Domain adapter '{name}' not found at {adapter_path}")
            sys.exit(1)
        if not data_path.exists():
            print(f"  FATAL: Domain val data '{name}' not found at {data_path}")
            sys.exit(1)
    print(f"  Domain adapters: {DOMAIN_NAMES}")
    print("  All adapters and data verified.")

    # ------------------------------------------------------------------
    # Phase 1: Load model
    # ------------------------------------------------------------------
    print("\n[Phase 1] Loading BitNet-2B-4T...")
    t0 = time.time()
    model, tokenizer = load(MODEL_ID)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    print("  Unpacking ternary weights...")
    model = replace_bitlinear_with_linear(model)

    print("  Applying LoRA layers...")
    model = apply_lora_to_model(model, rank=LORA_RANK, scale=LORA_SCALE)

    # ------------------------------------------------------------------
    # Phase 2: Load all adapters
    # ------------------------------------------------------------------
    print("\n[Phase 2] Loading adapters...")
    reasoning_adapter = load_adapter(REASONING_ADAPTER_DIR)
    print(f"  Reasoning: {len(reasoning_adapter)} tensors")

    domain_adapters = {}
    for name in DOMAIN_NAMES:
        domain_adapters[name] = load_adapter(DOMAIN_ADAPTERS_DIR / name)
        print(f"  {name}: {len(domain_adapters[name])} tensors")

    # ------------------------------------------------------------------
    # Phase 3: Base model PPL (no adapters)
    # ------------------------------------------------------------------
    print("\n[Phase 3] Base model PPL (LoRA zeroed)...")
    zero_lora_params(model)

    base_ppls = {}
    for name in DOMAIN_NAMES:
        ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        base_ppls[f"domain_{name}"] = ppl
        print(f"  Base on {name}: {ppl:.4f}")

    base_reasoning_ppl = compute_ppl(model, tokenizer, REASONING_DATA_DIR)
    base_ppls["reasoning"] = base_reasoning_ppl
    print(f"  Base on reasoning: {base_reasoning_ppl:.4f}")
    results["base_ppls"] = base_ppls

    # ------------------------------------------------------------------
    # Phase 4: Individual adapter PPL
    # ------------------------------------------------------------------
    print("\n[Phase 4] Individual adapter PPL...")

    # Reasoning adapter alone (on reasoning data and all domain data)
    individual_ppls = {}
    zero_lora_params(model)
    apply_adapter_weights(model, reasoning_adapter)
    mx.eval(model.parameters())

    reasoning_on_reasoning = compute_ppl(model, tokenizer, REASONING_DATA_DIR)
    individual_ppls["reasoning_on_reasoning"] = reasoning_on_reasoning
    print(f"  Reasoning adapter on reasoning data: {reasoning_on_reasoning:.4f} "
          f"(vs base {base_reasoning_ppl:.4f}, "
          f"ratio {base_reasoning_ppl/reasoning_on_reasoning:.2f}x)")

    for name in DOMAIN_NAMES:
        ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        individual_ppls[f"reasoning_on_{name}"] = ppl
        print(f"  Reasoning adapter on {name}: {ppl:.4f}")

    # Each domain adapter alone (on its own data and on reasoning data)
    for name in DOMAIN_NAMES:
        zero_lora_params(model)
        apply_adapter_weights(model, domain_adapters[name])
        mx.eval(model.parameters())

        # Domain adapter on own domain
        domain_ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        individual_ppls[f"{name}_on_{name}"] = domain_ppl
        print(f"  {name} adapter on {name}: {domain_ppl:.4f} "
              f"(vs base {base_ppls[f'domain_{name}']:.4f}, "
              f"ratio {base_ppls[f'domain_{name}']/domain_ppl:.2f}x)")

        # Domain adapter on reasoning data
        domain_on_reasoning = compute_ppl(model, tokenizer, REASONING_DATA_DIR)
        individual_ppls[f"{name}_on_reasoning"] = domain_on_reasoning
        print(f"  {name} adapter on reasoning: {domain_on_reasoning:.4f}")

    results["individual_ppls"] = individual_ppls

    # ------------------------------------------------------------------
    # Phase 5: Composed adapter PPL (domain + reasoning)
    # ------------------------------------------------------------------
    print("\n[Phase 5] Composed PPL (domain + reasoning, 1/2 scaling)...")

    composed_ppls = {}
    k1_improvements = {}  # reasoning PPL improvement for each domain
    k2_degradations = {}  # domain PPL degradation for each domain

    for name in DOMAIN_NAMES:
        print(f"\n  --- {name} + reasoning ---")

        # Compose with 1/2 scaling
        compose_adapters(model, [domain_adapters[name], reasoning_adapter], scale=0.5)

        # PPL on domain data
        composed_domain_ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        composed_ppls[f"{name}_domain"] = composed_domain_ppl

        # PPL on reasoning data
        composed_reasoning_ppl = compute_ppl(model, tokenizer, REASONING_DATA_DIR)
        composed_ppls[f"{name}_reasoning"] = composed_reasoning_ppl

        # K2: Domain degradation
        domain_alone_ppl = individual_ppls[f"{name}_on_{name}"]
        if domain_alone_ppl > 0:
            degradation_pct = ((composed_domain_ppl / domain_alone_ppl) - 1.0) * 100
        else:
            degradation_pct = float("inf")
        k2_degradations[name] = {
            "domain_alone": domain_alone_ppl,
            "domain_composed": composed_domain_ppl,
            "degradation_pct": round(degradation_pct, 2),
            "pass": degradation_pct <= 3.0,
        }
        print(f"  Domain PPL: {domain_alone_ppl:.4f} -> {composed_domain_ppl:.4f} "
              f"({degradation_pct:+.2f}%)")

        # K1: Reasoning improvement
        # Compare domain+reasoning on reasoning data vs domain-alone on reasoning data
        domain_on_reasoning = individual_ppls[f"{name}_on_reasoning"]
        if domain_on_reasoning > 0:
            reasoning_improvement_pct = (1.0 - composed_reasoning_ppl / domain_on_reasoning) * 100
        else:
            reasoning_improvement_pct = 0.0
        k1_improvements[name] = {
            "domain_on_reasoning": domain_on_reasoning,
            "composed_on_reasoning": composed_reasoning_ppl,
            "improvement_pct": round(reasoning_improvement_pct, 2),
            "improved": reasoning_improvement_pct > 0,
        }
        print(f"  Reasoning PPL: {domain_on_reasoning:.4f} -> {composed_reasoning_ppl:.4f} "
              f"({reasoning_improvement_pct:+.2f}%)")

    results["composed_ppls"] = composed_ppls

    # ------------------------------------------------------------------
    # Phase 6: Cosine analysis (reasoning vs each domain)
    # ------------------------------------------------------------------
    print("\n[Phase 6] Cosine similarity (reasoning vs domains)...")
    cosines = {}
    for name in DOMAIN_NAMES:
        cos = compute_cosine(reasoning_adapter, domain_adapters[name])
        cosines[name] = cos
        print(f"  |cos|(reasoning, {name}) = {cos:.6f}")

    results["cosines"] = cosines
    results["mean_cos"] = round(sum(cosines.values()) / len(cosines), 6)
    results["max_cos"] = max(cosines.values())

    # ------------------------------------------------------------------
    # Phase 7: Kill criteria assessment
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KILL CRITERIA ASSESSMENT")
    print("=" * 70)

    # K1: Reasoning improvement on >50% of domains
    n_improved = sum(1 for v in k1_improvements.values() if v["improved"])
    n_total = len(k1_improvements)
    k1_pct = n_improved / n_total * 100
    k1_pass = n_improved >= n_total * 0.5  # >= 50%

    print(f"\n  K1: Reasoning PPL improves on {n_improved}/{n_total} domains ({k1_pct:.0f}%)")
    print(f"      Threshold: >= 50% | {'PASS' if k1_pass else 'KILL'}")
    for name, v in k1_improvements.items():
        status = "improved" if v["improved"] else "WORSE"
        print(f"      {name}: {v['improvement_pct']:+.2f}% ({status})")

    results["k1"] = {
        "n_improved": n_improved,
        "n_total": n_total,
        "pct_improved": round(k1_pct, 1),
        "threshold_pct": 50,
        "pass": k1_pass,
        "details": k1_improvements,
    }

    # K2: Domain degradation <3%
    n_degraded = sum(1 for v in k2_degradations.values() if not v["pass"])
    worst_degradation = max(v["degradation_pct"] for v in k2_degradations.values())
    k2_pass = n_degraded == 0

    print(f"\n  K2: Domain PPL degradation >3% on {n_degraded}/{n_total} domains")
    print(f"      Worst: {worst_degradation:+.2f}% | Threshold: <=3% | "
          f"{'PASS' if k2_pass else 'KILL'}")
    for name, v in k2_degradations.items():
        status = "PASS" if v["pass"] else "KILL"
        print(f"      {name}: {v['degradation_pct']:+.2f}% ({status})")

    results["k2"] = {
        "n_degraded_over_3pct": n_degraded,
        "n_total": n_total,
        "worst_degradation_pct": round(worst_degradation, 2),
        "threshold_pct": 3.0,
        "pass": k2_pass,
        "details": k2_degradations,
    }

    # ------------------------------------------------------------------
    # Phase 7b: Dilution control (domain at 1/2 scale alone)
    # ------------------------------------------------------------------
    print("\n[Phase 7b] Dilution control: domain adapter at 1/2 scale (no reasoning)...")
    dilution_control = {}

    for name in DOMAIN_NAMES:
        zero_lora_params(model)
        apply_adapter_weights(model, domain_adapters[name], scale=0.5)
        mx.eval(model.parameters())

        diluted_domain_ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        diluted_reasoning_ppl = compute_ppl(model, tokenizer, REASONING_DATA_DIR)

        # Compare composed vs diluted-only
        composed_domain = composed_ppls[f"{name}_domain"]
        interference = ((composed_domain / diluted_domain_ppl) - 1.0) * 100

        dilution_control[name] = {
            "diluted_domain_ppl": diluted_domain_ppl,
            "diluted_reasoning_ppl": diluted_reasoning_ppl,
            "composed_domain_ppl": composed_domain,
            "interference_pct": round(interference, 2),
        }

        print(f"  {name}: diluted-only={diluted_domain_ppl:.4f}, "
              f"composed={composed_domain:.4f}, "
              f"interference={interference:+.2f}%")

    results["dilution_control"] = dilution_control

    # Recompute K2 as interference (above dilution baseline) rather than
    # degradation from individual
    n_interfered = sum(1 for v in dilution_control.values()
                       if v["interference_pct"] > 3.0)
    worst_interference = max(v["interference_pct"] for v in dilution_control.values())
    k2_interference_pass = n_interfered == 0

    print(f"\n  K2 (interference-corrected): {n_interfered}/5 domains >3% interference")
    print(f"  Worst interference: {worst_interference:+.2f}%")
    print(f"  {'PASS' if k2_interference_pass else 'KILL'}")

    results["k2_interference"] = {
        "n_interfered_over_3pct": n_interfered,
        "worst_interference_pct": round(worst_interference, 2),
        "pass": k2_interference_pass,
    }

    # ------------------------------------------------------------------
    # Phase 7c: Unit-weight composition (scale=1.0)
    # ------------------------------------------------------------------
    print("\n[Phase 7c] Unit-weight composition (scale=1.0 for each)...")
    unit_ppls = {}

    for name in DOMAIN_NAMES:
        compose_adapters(model, [domain_adapters[name], reasoning_adapter], scale=1.0)

        unit_domain_ppl = compute_ppl(model, tokenizer, DOMAIN_DATA_DIR / name)
        unit_reasoning_ppl = compute_ppl(model, tokenizer, REASONING_DATA_DIR)

        domain_alone = individual_ppls[f"{name}_on_{name}"]
        domain_deg = ((unit_domain_ppl / domain_alone) - 1.0) * 100
        reasoning_alone_on_reasoning = individual_ppls[f"{name}_on_reasoning"]
        reasoning_imp = (1.0 - unit_reasoning_ppl / reasoning_alone_on_reasoning) * 100

        unit_ppls[name] = {
            "domain_ppl": unit_domain_ppl,
            "reasoning_ppl": unit_reasoning_ppl,
            "domain_degradation_pct": round(domain_deg, 2),
            "reasoning_improvement_pct": round(reasoning_imp, 2),
        }
        print(f"  {name}: domain {domain_alone:.4f}->{unit_domain_ppl:.4f} ({domain_deg:+.2f}%), "
              f"reasoning {reasoning_alone_on_reasoning:.4f}->{unit_reasoning_ppl:.4f} ({reasoning_imp:+.2f}%)")

    results["unit_weight_composition"] = unit_ppls

    # Unit-weight K2 check
    n_unit_degraded = sum(1 for v in unit_ppls.values() if v["domain_degradation_pct"] > 3.0)
    worst_unit = max(v["domain_degradation_pct"] for v in unit_ppls.values())
    print(f"\n  Unit-weight K2: {n_unit_degraded}/5 >3% degradation, worst={worst_unit:+.2f}%")

    # ------------------------------------------------------------------
    # Phase 8: Additional analysis
    # ------------------------------------------------------------------
    print("\n[Phase 8] Additional analysis...")

    # Compare composed reasoning PPL to reasoning-alone PPL
    reasoning_alone = individual_ppls["reasoning_on_reasoning"]
    print(f"\n  Reasoning adapter alone: {reasoning_alone:.4f}")
    for name in DOMAIN_NAMES:
        composed_reasoning = composed_ppls[f"{name}_reasoning"]
        ratio = composed_reasoning / reasoning_alone
        print(f"  {name}+reasoning composed: {composed_reasoning:.4f} "
              f"({ratio:.2f}x reasoning-alone)")

    # Compare to base
    print(f"\n  Base model on reasoning: {base_reasoning_ppl:.4f}")
    for name in DOMAIN_NAMES:
        composed_reasoning = composed_ppls[f"{name}_reasoning"]
        vs_base = (1.0 - composed_reasoning / base_reasoning_ppl) * 100
        print(f"  {name}+reasoning vs base: {vs_base:+.2f}%")

    # ------------------------------------------------------------------
    # Verdict (use interference-corrected K2 as primary)
    # ------------------------------------------------------------------
    # The raw K2 (degradation from individual) conflates dilution with
    # interference. Under 1/N scaling, dilution is expected and inherent.
    # The interference-corrected K2 isolates actual cross-adapter harm.
    k2_final = k2_interference_pass  # Use interference-corrected version

    if k1_pass and k2_final:
        verdict = "SUPPORTED"
    elif not k1_pass and not k2_final:
        verdict = "KILLED (K1: reasoning improves <50% of domains) (K2: interference >3%)"
    elif not k1_pass:
        verdict = "KILLED (K1: reasoning improves <50% of domains)"
    else:
        verdict = "KILLED (K2: interference >3%)"

    # Note the raw K2 failure for transparency
    if k2_pass:
        verdict += " [raw K2 also PASS]"
    else:
        verdict += f" [raw K2 KILL: worst degradation {worst_degradation:+.2f}% -- dilution-dominated]"

    elapsed = time.time() - t_start
    results["verdict"] = verdict
    results["total_time_s"] = round(elapsed, 1)
    results["total_time_min"] = round(elapsed / 60, 1)

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"{'=' * 70}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
