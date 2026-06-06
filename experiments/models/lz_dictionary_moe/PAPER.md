# LZ Dictionary MoE: Research Digest

## Hypothesis

Expert MLPs can be factored into references to a shared "dictionary" of low-rank
sub-modules plus per-expert residual deltas, achieving equivalent quality to
independent experts at reduced total parameter count.

**Falsifiable**: If dictionary-composed experts are >3% worse than independent
experts at the same total parameter count, OR dictionary utilization is <30%
(most entries unused), the approach is killed.

---

## What This Model Is

`DictionaryMoEGPT` replaces independent expert MLPs with **dictionary-composed
experts**. Each expert is expressed as a weighted combination of shared low-rank
sub-modules (the "dictionary") plus a small unique residual:

    expert_i(x) = sum_j alpha_{i,j} * dict_j(x) + delta_i(x)

This is inspired by LZ77/LZ78 compression, where repeated patterns are stored in
a dictionary and referenced by pointers. Here:
- **Dictionary entries** = shared low-rank MLP sub-modules (d -> r -> d)
- **Alpha coefficients** = per-expert "pointers" into the dictionary (softmax-normalized)
- **Delta residual** = per-expert unique low-rank component (the "literal bytes")

### Architecture Details

- **Dictionary**: D=8 shared entries per layer, each a rank-r MLP (down + ReLU + up)
- **Experts**: N=4 per layer, each with D alpha coefficients + rank-r_delta residual
- **Routing**: Standard softmax top-k=2 (identical to MoEGPT)
- **Training**: End-to-end, all components jointly optimized

### Why It Exists

The project has established that expert weight matrices contain shared structure:
behavioral_dedup found 19.3% redundancy at Layer 0, and Procrustes decomposition
found 54% shared fraction. Rather than sharing entire experts (DeepSeek-MoE) or
decomposing post-hoc (Procrustes, killed due to nonlinearity), this approach
trains shared structure FROM THE START within each expert.

---

## Lineage in the Arena

```
gpt -> moe -> lz_dictionary_moe
               (dictionary-composed experts replace independent MLPs)
```

Related experiments:
- `behavioral_dedup`: Found 19.3% capsule redundancy at Layer 0 (motivating finding)
- `procrustes_decomp`: Killed -- post-hoc decomposition breaks at nonlinearities
- `lora_procrustes`: Validated shared structure for linear LoRA adapters

---

## Key References

**StructMoE** (Sarwar et al., ICML 2024): Low-Rank Experts (LoRE) dynamically
selected per-expert via secondary router. Closest prior art. Key difference: we
compose from a SHARED codebook rather than per-expert secondary selection.

**L-MoE** (2025): Soft weighted average of LoRA expert parameters. Routes and
combines expert parameters rather than activations. Our dictionary entries are
analogous but operate at the sub-network level (full low-rank MLPs, not just
adapter matrices).

**Autonomy-of-Experts** (AoE, ICLR 2025): Low-rank factorization of expert
first layer for self-routing. We factor both projections into shared dictionary
+ unique residual.

**DeepSeek-MoE**: Shared experts (always active) + routed experts. Our approach
goes deeper: WITHIN each expert, sub-components are shared via dictionary.

---

## Empirical Results

### Configuration

- d=64, 4 layers, 4 heads, 4 experts, top-k=2
- Character-level names dataset, 500 training steps, lr=3e-3
- 3 seeds (42, 123, 7)

### Models Compared

| Model | MLP Params/Layer | Total Params | Description |
|-------|-----------------|-------------|-------------|
| Dense GPT | 1 * 8d^2 | 202,112 | No MoE, single MLP |
| Standard MoE | 4 * 8d^2 | 596,352 | 4 independent experts |
| Dict MoE (small) | 8*2dr + 4*2d*r_d | 236,032 | D=8, r=32, r_delta=16 |
| Dict MoE (large) | 8*2dr + 4*2d*r_d | 432,640 | D=8, r=64, r_delta=48 |

### Results (3-seed mean +/- std)

| Model | Params | Val Loss | vs Std MoE | vs Dense |
|-------|--------|----------|------------|----------|
| Dense GPT | 202,112 | 0.5118 +/- 0.0078 | -0.6% | baseline |
| Standard MoE | 596,352 | 0.5148 +/- 0.0046 | baseline | +0.6% |
| Dict MoE (small) | 236,032 | 0.5104 +/- 0.0043 | **-0.9%** | -0.3% |
| Dict MoE (large) | 432,640 | 0.5118 +/- 0.0063 | -0.6% | -0.0% |

### Kill Criteria Check

| Criterion | Value | Threshold | Verdict |
|-----------|-------|-----------|---------|
| Quality: Dict small vs Std MoE | -0.9% (BETTER) | >3% worse | **PASS** |
| Quality: Dict large vs Std MoE | -0.6% (BETTER) | >3% worse | **PASS** |
| Dictionary utilization | 100% | <30% | **PASS** |

### Dictionary Utilization (all layers, all seeds)

| Layer | Utilization Rate | Normalized Entropy |
|-------|-----------------|-------------------|
| Layer 0 | 1.00 | 0.999 |
| Layer 1 | 1.00 | 0.999 |
| Layer 2 | 1.00 | 0.999 |
| Layer 3 | 1.00 | 0.999 |

All dictionary entries are utilized. Entropy is near-maximum (H/H_max ~ 1.0),
meaning alpha weights are close to uniform across dictionary entries.

### Key Observations

1. **Dictionary MoE with 40% of standard MoE params matches or BEATS standard MoE.**
   Dict MoE (small) at 236K params achieves -0.9% better loss than Standard MoE
   at 596K params. This is a 60% parameter reduction with no quality loss.

2. **All dictionary entries are used (100% utilization, entropy ~1.0).** This means
   the shared structure is being leveraged, though the near-uniform alpha weights
   suggest the dictionary entries have not specialized into distinct sub-functions
   at this scale.

3. **Standard MoE provides no benefit over Dense GPT at this scale.** Standard MoE
   (596K) is +0.6% worse than Dense GPT (202K). This is consistent with known
   micro-scale behavior: more parameters do not help when the task is simple.

4. **Dictionary composition acts as effective regularization.** By forcing experts
   through shared low-rank sub-modules, the model avoids overfitting that plagues
   the standard MoE's independent experts at micro scale.

---

## Micro-Scale Limitations

1. **Alpha weights are near-uniform.** At d=64 with 500 training steps, the
   composition coefficients have not differentiated. Dictionary entries may
   specialize with longer training or harder tasks. At macro scale with BPE
   tokens and diverse domains, we would expect clear specialization patterns.

2. **Standard MoE is not a strong baseline here.** At micro scale, Dense GPT
   beats Standard MoE, so the Dictionary MoE's advantage may be partly
   regularization rather than structural sharing. The proper test is at scale
   where MoE genuinely outperforms dense models.

3. **No domain-specific analysis.** This experiment uses a single domain
   (character-level names). The LZ analogy is strongest when experts serve
   different domains with shared sub-patterns -- untested here.

4. **Soft composition only.** All dictionary entries contribute to every expert
   (soft alpha via softmax). Hard selection (sparse alpha) may enable more
   efficient computation but was not tested due to the top-1 phase transition
   risk observed in the sparse_router experiment.

5. **Fixed dictionary size.** D=8 was chosen heuristically. Optimal codebook
   size likely depends on expert count, task diversity, and model capacity.

---

## What Would Kill This

### At micro scale (already tested)
- Dict MoE >3% worse than Standard MoE at same params: **NOT KILLED** (-0.9%)
- Dictionary utilization <30%: **NOT KILLED** (100%)

### At macro scale (future validation needed)
- Dictionary entries remain near-uniform after long training (no specialization)
- Dictionary-composed experts fail to match independent experts when MoE
  genuinely outperforms dense (the micro-scale MoE-vs-dense parity obscures this)
- Hard/sparse alpha selection causes phase transition (like k=1 routing kill)
- Codebook size scaling: if optimal D grows as O(N) rather than O(1), the
  sharing benefit disappears at large N

### Theoretical kill
- If expert weight matrices have full rank with no shared low-rank structure,
  dictionary decomposition is strictly worse (forces low-rank bottleneck).
  The behavioral_dedup evidence (19.3% redundancy) argues against this.
