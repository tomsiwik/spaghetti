# PAPER.md — P4.D0: Domain + Format Adapter Simultaneous Composition

## Summary

Hypothesis: Domain adapters (q_proj) and format adapters (v_proj+o_proj) occupy
disjoint parameter subspaces and therefore compose without interference via additive
weight merging. **KILLED**: parameter disjointness is necessary but NOT sufficient.
Functional coupling through the attention mechanism creates catastrophic interference
at the perturbation magnitudes used by these adapters.

## Setup

- Model: Gemma 4 4B (mlx-community/gemma-4-e4b-it-4bit, 4-bit quantized)
- Domain adapters: medical/legal, q_proj rank 6, scale 6.0, all 42 layers
  - Source: P1.T2 (Finding #421)
- Format adapters: SOAP/legal-brief, v_proj+o_proj rank 16, scale 4.0, layers 30-41
  - Source: P4.C1 (Finding #480)
- Composition: additive weight merging (fuse LoRA deltas into base weights)
- Eval: N=10 questions per condition, keyword/format scoring

## Prediction vs Measurement Table

| Kill Criterion | Prediction (MATH.md) | Measurement | Status |
|---|---|---|---|
| K1249: Medical+SOAP domain ≥40% AND format ≥50pp | domain ≥30%, format ≥40pp | domain=0%, format=-10pp | **FAIL** |
| K1250: Legal+Legal-brief domain ≥40% AND format ≥60pp | domain ≥40%, format ≥60pp | domain=0%, format=0pp | **FAIL** |
| K1251: Solo degradation ≤15pp | ≤10pp (disjoint params) | SOAP: 60pp, Legal: 80pp, Medical: 100pp | **FAIL** |

All three criteria catastrophically fail. Composed model produces non-English garbage.

## Root Cause Analysis

### 1. Parameter Disjointness Confirmed (Theorem 1 Verified)

Zero overlap between adapter weight keys:
- Medical: 84 keys (42 layers × q_proj × {lora_a, lora_b})
- SOAP: 48 keys (12 layers × {v_proj, o_proj} × {lora_a, lora_b})
- Intersection: ∅

### 2. Functional Coupling Destroys Composition (Theorem 2 Caveat Dominates)

The attention mechanism creates implicit coupling:
```
attn_output = softmax(q_proj(x) @ k_proj(x)^T / sqrt(d)) @ v_proj(x)
layer_output = o_proj(attn_output) + x
```

Changing q_proj (attention pattern) AND o_proj (output projection) simultaneously
creates a compound perturbation. Each adapter was trained to work with the BASE
model's complementary projections, not the other adapter's modified projections.

### 3. Perturbation Magnitude Is the Critical Factor

| Projection | Adapter | Delta Frobenius Norm | Base Weight Norm | Relative Perturbation |
|---|---|---|---|---|
| q_proj (layer 35) | Medical | 4.74 | 74.00 | **6.37%** |
| o_proj (layer 35) | SOAP | 15.84 | 65.50 | **24.09%** |
| v_proj (layer 35) | SOAP | 0.00 | 31.88 | 0.00% |

The SOAP o_proj perturbation (24.09%) is nearly 4x the medical q_proj perturbation.
Note: v_proj learned NOTHING (all lora_b = 0). The entire SOAP effect is through o_proj.

### 4. Scaled Composition Experiment

To locate the collapse threshold, tested with SOAP adapter scaled by α:

| SOAP α | Effective o_proj Perturbation | Output Quality | SOAP Format |
|---|---|---|---|
| 0.0 (medical only) | 0% | Coherent clinical note | No |
| 0.1 | ~2.4% | Coherent clinical note | No |
| 0.25 | ~6.0% | Coherent clinical note | No |
| 0.5 | ~12.0% | Coherent, has "SUBJECTIVE (S):" | Partial |
| 1.0 (full) | ~24.1% | **Hindi garbage, model collapse** | No |

**Collapse threshold**: between 12-24% o_proj perturbation (when combined with 6.4% q_proj).

**Key observation**: even at sub-collapse α, the SOAP format effect is severely
attenuated. The format adapter's learned transformations are disrupted by the changed
attention patterns from the domain adapter, even when the perturbation is small enough
to keep the model coherent.

## Impossibility Structure

**Why this failure is structural, not parametric:**

The attention mechanism creates a functional dependency chain:
```
q_proj(x) → attention_weights → attn_output → o_proj(attn_output) → residual
```

Adapter A modifies the input (q_proj) and Adapter B modifies the output (o_proj) of
this chain. Even though A and B touch disjoint parameters, they jointly determine the
layer's output through:

  output = o_proj_B(softmax(q_proj_A(x) @ k(x)^T) @ v(x))

This is a composition of two modified functions, not an addition of two independent
perturbations. The compound effect can exceed the sum of individual effects when both
perturbations are large.

**Mathematical condition for safe composition:**

For additive composition to preserve individual adapter behavior, we need the functional
Jacobian of the attention mechanism to be approximately identity at the perturbation
magnitudes used. Specifically:

  ||∂(layer_output)/∂(q_proj_weights) × ∂(layer_output)/∂(o_proj_weights)|| ≈ 0

This cross-Jacobian is NOT zero — it's the attention mechanism's implicit coupling.
Safe composition requires EITHER:
1. Small perturbations (both adapters < ~10% relative norm), OR
2. Adapters that DON'T span the attention q→o functional chain, OR
3. Adapters trained jointly (aware of each other's perturbations)

## Verdict

**KILLED** — All three kill criteria fail catastrophically.

Parameter disjointness (Theorem 1) is verified but insufficient. The functional
dependency caveat (Theorem 2 caveat) dominates: the attention mechanism's implicit
coupling creates catastrophic interference when both q_proj and o_proj are perturbed
by the adapters' trained magnitudes.

## Implications for the Architecture

1. **Domain + format composition requires co-training**, not post-hoc merging.
   The format adapter must be trained on the domain-adapted model's hidden states.
   (This is exactly what P3.B5 proposed for a different reason.)

2. **Same-projection composition (q_proj + q_proj) may be safer** than cross-projection
   composition. Prior results (Finding #440, Grassmannian N=100) showed near-zero
   interference for same-projection adapters because they don't span the attention chain.

3. **The 24% o_proj perturbation is too aggressive.** Future format adapters should use
   lower rank or scale to keep the perturbation < 10% for composition compatibility.

## Solo Adapter Performance (for reference)

| Adapter | Solo Performance |
|---|---|
| Medical domain (q_proj) | 100% medical keyword pass rate |
| SOAP format (v_proj+o_proj) | 60% SOAP format compliance |
| Legal domain (q_proj) | 50% legal keyword pass rate |
| Legal-brief format (v_proj+o_proj) | 80% legal format compliance |

## References

1. Finding #421: LoRA r=6 q_proj achieves 22-82pp domain improvement
2. Finding #480: v_proj+o_proj SOAP +70pp, Legal +90pp
3. Finding #440: Grassmannian isolation max cos=2.25e-8 at N=100 (same-projection)
4. Hu et al. 2021 (arxiv 2106.09685) — LoRA layer selection
5. Geva et al. 2021 (arxiv 2012.14913) — attention value vectors as memories
