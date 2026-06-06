# Delta Rule Interference Ordering: Research Digest

## Hypothesis

The delta rule's retrieval-and-correction mechanism (v_t - S^T k_t) causes
composed domains to actively interfere through shared state memory, reversing
the favorable interference ordering (linear < full, ratio 0.59x) found in
the simplified gated linear recurrence variant.

## Verdict: PASS (both kill criteria)

Kill criterion 1 (interference ordering reversal, ratio >1.0x): **PASS**.
Delta rule linear/full interference ratio is 0.74x -- linear attention
layers show LESS composition interference than the full attention layer,
even with the delta rule's cross-domain retrieval mechanism. The ordering
is not reversed.

Kill criterion 2 (composition gap >+10% median): **PASS**. Delta rule
median composition gap is +0.39% across 7 seeds. All 7 gaps fall within
[-0.31%, +1.45%] -- no catastrophic failures, tight distribution.

The adversarial review's priority-1 concern (delta rule reverses interference
ordering) is empirically falsified. The delta rule is composition-compatible
at micro scale.

## What This Model Is

DeltaRuleHybridCapsuleMoEGPT implements the full GatedDeltaNet delta rule
mechanism at micro scale, extending the L2-normalized hybrid attention model:

    kv_mem_t = S_{t-1}^T @ k_t              -- retrieve stored association
    delta_t = (v_t - kv_mem_t) * beta_t      -- correction (new info only)
    S_t = g_t * S_{t-1} + k_t * delta_t^T    -- update with correction

Additional mechanisms matching real GatedDeltaNet:
- Per-head beta gating (learned update strength, sigmoid)
- SiLU output gating (RMSNorm * SiLU(z))
- L2 QK normalization (proven to eliminate instability)
- Parameterized decay: g = exp(-A * softplus(a + dt_bias))

The model has 217,112 parameters (+6.4% vs L2-normalized variant) due to
the additional beta, z, and decay projections.

## Lineage in the Arena

```
gpt (dense baseline)
 |-- capsule_moe (routed capsule groups)
      |-- hybrid_capsule_moe (3:1 linear:full, simplified)
           |-- l2_norm_hybrid_capsule_moe (+ L2 QK norm)
                |-- delta_rule_hybrid_capsule_moe (+ delta rule) <-- THIS
```

## Key References

- GatedDeltaNet (Yang et al., 2024): The delta rule for linear attention.
  Retrieval-and-correction state update avoids redundant storage.
- Qwen3.5-0.8B (2026): Production architecture using GatedDeltaNet with
  L2 QK normalization, per-dim beta, SiLU output gating, conv1d.
- qwen3_5_transformers.py (HF Transformers): Reference implementation of
  torch_recurrent_gated_delta_rule used to validate our micro implementation.
- exp_l2_norm_composition_stability (this project): L2 normalization
  eliminates catastrophic composition failures (0/25 vs 4/25).
- exp_hybrid_attention_composition (this project): Simplified variant
  showed 0.59x interference ratio (linear < full).

## Protocol

Identical to prior hybrid attention experiments:
1. Pretrain shared base on all data (300 steps)
2. Fine-tune capsule groups per domain (300 steps, attention frozen)
3. Compose by concatenating domain groups, double top-k
4. Calibrate router on mixed data (100 steps)
5. Evaluate on per-domain val sets
6. Compute per-layer interference (cosine distance between domain outputs)

Three conditions, 7 seeds each:
- **full_attn**: all 4 layers full attention (control)
- **l2_norm_3_1**: simplified linear + L2 norm (baseline)
- **delta_rule_3_1**: delta rule linear + L2 norm (test)

## Empirical Results

### Composition Gap Summary (7 seeds)

| Condition | Gap mean | Gap median | Gap std | Gap min | Gap max |
|-----------|----------|-----------|---------|---------|---------|
| full_attn | +0.58% | +0.43% | 1.21% | -0.88% | +2.57% |
| l2_norm_3_1 | -0.43% | -0.50% | 0.57% | -1.36% | +0.38% |
| delta_rule_3_1 | +0.51% | +0.39% | 0.54% | -0.31% | +1.45% |

### Per-Seed Composition Gaps

| Seed | Full Attn | L2 Norm | Delta Rule |
|------|-----------|---------|------------|
| 0 | -0.36% | -0.65% | +0.81% |
| 1 | -0.88% | -0.27% | +0.64% |
| 2 | +0.43% | -1.36% | +0.25% |
| 3 | +0.02% | +0.09% | +1.45% |
| 4 | +1.77% | -0.50% | +0.36% |
| 5 | +0.53% | -0.66% | -0.31% |
| 6 | +2.57% | +0.38% | +0.39% |

No catastrophic failures in any condition across all 21 runs (7 seeds x 3
conditions). The L2 normalization (present in both l2_norm and delta_rule
conditions) fully eliminates the instability found in the original hybrid
attention experiment.

### Kill Criterion 1: Interference Ordering

Per-layer mean interference (cosine distance, 7 seeds):

| Layer | Type | Full Attn | L2 Norm | Delta Rule |
|-------|------|-----------|---------|------------|
| 0 | linear | 0.2003 | 0.3944 | 0.3133 |
| 1 | linear | 0.4049 | 0.5526 | 0.4496 |
| 2 | linear | 0.5287 | 0.5287 | 0.5774 |
| 3 | full | 0.7872 | 0.6319 | 0.6896 |

Interference ratio (linear layers 1,2 vs full layer 3):

    L2 norm (simplified): 0.86x
    Delta rule:           0.74x
    Threshold:            1.0x
    PASS: 0.74x <= 1.0x

The delta rule actually shows LOWER interference ratio (0.74x) than the
simplified variant (0.86x), contrary to the hypothesis. The retrieval-
and-correction mechanism does not amplify cross-domain interference --
it may even help by computing targeted corrections rather than blindly
adding associations.

### Kill Criterion 2: Composition Gap

    Delta rule median gap: +0.39%
    Threshold: +10%
    PASS: +0.39% <= +10%

The delta rule composition gap is comparable to full attention (+0.43%)
and slightly larger than L2 norm simplified (-0.50%). All three conditions
show gaps well under 2% -- composition works reliably.

## Key Findings

1. **The delta rule does NOT reverse interference ordering.** Linear
   attention layers with the delta rule show 0.74x the interference of the
   full attention layer. This is actually better (lower) than the simplified
   variant's 0.86x measured with the same 7-seed set. The adversarial
   review's priority-1 concern is empirically falsified.

2. **Composition works with the delta rule.** Median gap +0.39% across 7
   seeds with zero catastrophic failures. The delta rule adds no composition
   risk beyond what the simplified variant already showed (when L2 normalized).

3. **The delta rule adds modest overhead.** +6.4% parameters (217K vs 204K)
   from the beta, z, and decay projections. Training throughput is ~2.5x
   slower (38K tok/s vs 95K tok/s for full attention) due to the sequential
   recurrence over T=32 timesteps. At macro scale, chunk-based implementation
   would eliminate most of this overhead.

4. **L2 normalization remains the critical stabilizer.** Both linear attention
   conditions (simplified and delta rule) benefit from L2 QK normalization.
   Zero catastrophic failures across 14 L2-normalized runs (7 seeds x 2
   conditions). This confirms the L2 norm finding transfers to the delta rule.

5. **The interference ratio varies with seed set.** The original hybrid
   attention experiment measured 0.59x (5 seeds), this experiment measures
   0.86x for L2 norm and 0.74x for delta rule (7 seeds). The directional
   finding (linear < full) is consistent; the exact ratio depends on the
   seed set. This is expected at micro scale with high variance.

## Comparison to Prior Experiments

| Experiment | Condition | Median Gap | Interference Ratio | Seeds |
|------------|-----------|-----------|-------------------|-------|
| hybrid_attention | simplified, no L2 | +1.27% | 0.59x | 5 |
| l2_norm_attention | simplified, L2 | -0.33% | N/A (25 seeds, no interference) | 25 |
| **this experiment** | simplified, L2 | -0.50% | 0.86x | 7 |
| **this experiment** | delta rule, L2 | +0.39% | 0.74x | 7 |
| **this experiment** | full attention | +0.43% | N/A (baseline) | 7 |

## Micro-Scale Limitations

1. **Sequential recurrence at T=32**: The delta rule's retrieval-correction
   is implemented as a Python-level loop over 32 timesteps. At macro scale
   (T=4096+), the chunk-based implementation (torch_chunk_gated_delta_rule)
   would be used instead. The mathematical equivalence is exact, but training
   dynamics at longer sequences may differ because the state accumulates
   more associations before being queried.

2. **No conv1d preprocessing**: Real GatedDeltaNet applies causal conv1d
   to the QKV projections before the recurrence. This provides local mixing
   that could affect how domains interact through the state. Omitted at
   micro scale because the sequence length (32) is shorter than typical
   conv1d kernel sizes (4).

3. **d_h=16 head dimension**: The recurrent state is d_h x d_h = 16x16 = 256
   entries. At macro scale (d_h=128 or 256), the state has 16K-65K entries,
   potentially storing many more associations and creating richer cross-domain
   interference patterns.

4. **7 seeds**: Sufficient for the clear PASS on both criteria (ratio 0.74x
   well under 1.0x; gap 0.39% well under 10%). The interference ratio
   estimate has high variance (individual layer values range 0.09-0.84)
   but the aggregate is stable.

5. **Character-level toy data**: Domains (a-m vs n-z names) may not create
   the strong specialization that real domains (e.g., Python code vs English
   text) would. Real domains could produce more distinctive associations in
   the state, amplifying the delta rule's cross-domain retrieval effects.

6. **Interference metric is indirect**: Cosine distance between capsule pool
   outputs measures how differently domains behave through each layer, but
   does not directly measure the delta rule's retrieval of cross-domain
   associations. A more direct metric would measure ||kv_mem_cross|| vs
   ||kv_mem_within|| during composition, but this would require instrumenting
   the attention forward pass.

## What Would Kill This

**At micro scale:**
- Running with 25+ seeds and finding the interference ratio drifts above
  1.0x (currently 0.74x with 7 seeds)
- Finding that specific seed initializations create pathological delta rule
  dynamics where one domain's corrections systematically degrade the other's
  associations

**At macro scale:**
- Long sequences (T=4096+) where the state accumulates many more associations,
  potentially creating richer and more harmful cross-domain retrieval patterns
- Real domain pairs (code vs text) that produce strong, incompatible
  associations in the state memory -- character-level names may not stress
  the delta rule's retrieval mechanism enough
- The conv1d preprocessing (omitted here) interacts with the delta rule to
  create localized interference patterns not captured at T=32
- Per-dimension beta gating (tested here as per-head scalar) at full per-dim
  granularity could change the interference dynamics

## Implication for the Macro Architecture

The combined findings from three hybrid attention experiments establish:

1. **Gated linear recurrence is composition-compatible** (simplified, 0.59x)
2. **L2 QK normalization eliminates instability** (0/25 failures)
3. **The delta rule does not break composition** (0.74x, gap +0.39%)

This means the full GatedDeltaNet mechanism -- as used in Qwen3.5 production
models -- is compatible with the capsule composition protocol. The remaining
untested components are conv1d preprocessing (likely neutral) and per-dimension
beta gating (tested as per-head here, likely neutral at per-dim granularity).

The macro architecture can confidently use hybrid attention (3:1 linear:full)
with GatedDeltaNet for linear layers without expecting composition to break.
This was the priority-1 risk identified by adversarial review, and it is
now mitigated at micro scale.
