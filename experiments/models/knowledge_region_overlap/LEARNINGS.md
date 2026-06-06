# LEARNINGS: Knowledge Region Overlap Mapping

## Core Finding

PPL improvement sets are degenerate for ternary base + LoRA (every adapter improves
every sample), but specialization sets (argmin PPL) reveal clear domain expertise
structure. Cosine similarity saturates at >0.986 (uninformative); L2 relative
difference (0.037-0.168) is the correct compatibility metric for adapter disagreement.

## Why This Happened

### 1. Universal PPL improvement is a ternary-specific phenomenon

The ternary base model (BitNet-2B-4T) has uniformly high per-token cross-entropy
across all domains. Any continuous perturbation (LoRA at any scale >= 1.0) reduces
PPL everywhere because the base model's quantization noise dominates domain-specific
signal. This is consistent with RILQ (2412.01129, AAAI 2025), which showed that
quantization error in sub-4-bit models spreads across the full rank — the error
surface is "rank-insensitive," meaning even a random low-rank correction provides
some benefit.

**Key insight:** This is NOT true for FP16 base models. The degenerate improvement
sets are a property of the ternary quantization, not of LoRA in general. A future
experiment on FP16 bases would likely produce non-trivial improvement sets.

### 2. Cosine similarity saturation at intermediate layers

At layer 15/30, all adapter pairs produce hidden states with cosine similarity
>0.986. This is a known phenomenon: "LoRA vs Full Fine-tuning: An Illusion of
Equivalence" (2410.21228) showed that LoRA modifications are concentrated in a
few "intruder dimensions" while leaving most singular vectors unchanged. Since
cosine similarity is dominated by the high-dimensional unchanged subspace, it
saturates near 1.0 even when the adapters disagree substantively.

Layer-wise LoRA fine-tuning (2602.05988) further showed "saturation of events"
where model predictions are fully constructed at specific layers — layer 15 may
be past the saturation point for domain differentiation, compressing all adapters
into a similar representational manifold.

### 3. Specialization structure is real and informative

Despite degenerate improvement sets, the argmin-PPL specialization sets show clear
structure: medical/code/math achieve 49-50/50 own-domain specialization, math
generalizes across 84/250 samples (strongest cross-domain), and legal partially
specializes at 34/50 (though scale=4.0 confounds this). This matches the "Standing
Committee" pattern from MoE literature — a core set of strongly specialized experts
plus peripheral generalizers.

## Confirming Evidence

- **RILQ** (2412.01129): Quantization error is rank-insensitive in sub-4-bit models,
  explaining why any LoRA perturbation helps — the error surface is flat enough that
  any low-rank correction provides marginal improvement across all inputs.

- **LoRA-LEGO** (2409.16167): Each rank in LoRA functions as an independent "Minimal
  Semantic Unit" with permutation invariance. This supports our finding that
  specialization operates at the rank level — adapters that win argmin do so because
  specific ranks encode domain-relevant corrections.

- **Finding #68** (our project): Weight-space orthogonality != data-space
  orthogonality. The high cosine similarity we observed (>0.986) does NOT mean
  adapters are functionally equivalent — they can disagree substantively in the
  dimensions that matter for prediction, as shown by L2 differences of 4-17%.

- **Finding #189** (our project): Energy gap argmin routing collapses at N=24 due
  to implicit calibration assumptions. Our specialization sets show the same pattern:
  math absorbs 26 finance + 8 legal samples, acting as a "dominant expert" that
  wins argmin across domains.

## Contradicting Evidence

- **LoRAuter** (Adsul et al., 2026): Found that simple linear merging of LoRA
  adapters can SURPASS individually fine-tuned adapters (70.95% on PIQA vs 46%
  single-task). This contradicts our implicit assumption that overlap regions are
  problematic — linear composition in overlaps might actually be beneficial, not
  harmful. However, LoRAuter used FP16 bases, not ternary.

- **Activation-Guided Consensus Merging** (2505.14009): Showed that activation
  mutual information is a better compatibility metric than static weight comparison.
  Our use of hidden state cosine similarity at a single layer is a much weaker
  version of this — activation MI across all layers would give richer structure.

- **MC-SMoE**: Demonstrated that "neuron permutation alignment" creates structured
  expert groups that merge safely. This suggests our adapter overlap structure might
  be an alignment artifact, not a genuine domain boundary.

## Alternative Approaches

### For defining knowledge regions (replacing PPL improvement sets):

1. **Task-aware vector embeddings** (2602.21222): Embed training examples with
   sentence encoders, define regions via similarity kernels. No PPL computation
   needed — operates in embedding space.

2. **Mutual information probing**: Measure MI between expert activations and
   linguistic categories. Defines regions by syntactic/semantic structure, not
   PPL thresholds.

3. **Gated-LPI (Log-Probability Increase)**: Neuron-level attribution decomposing
   log-prob increase across individual neurons. Maps knowledge at finer granularity
   than sample-level PPL.

### For measuring compatibility (replacing cosine similarity):

1. **Activation-Guided Consensus Merging** (2505.14009): Layer-specific merging
   coefficients from activation MI. Inversely weights layers by similarity —
   automatically focuses on layers where adapters disagree.

2. **Activation-Informed Merging** (2502.02421): Integrates activation-space
   information into merging decisions. Complementary to any merging method.

3. **Sub-MoE output cosine**: Cluster experts by output (not hidden state) cosine
   similarity. Measures functional disagreement, not representational similarity.

### For layer-wise adapter analysis:

1. **Mediator** (2502.04411): Layer-wise parameter conflict tracking. Average
   low-conflict layers, route high-conflict layers via task-specific experts.
   Directly applicable to our architecture.

2. **Layer-wise LoRA** (2602.05988): Identify which layers contribute most to
   representation change. Could reveal whether layer 15 is optimal or if earlier
   layers capture more domain differentiation.

## Implications for Next Experiments

### 1. The sheaf framework needs corrected inputs

The sheaf-theoretic analysis (Hansen & Ghrist, 2110.03789) was never tested because
the cover was degenerate. Two corrections are needed before retrying:
- **Cover definition**: Use specialization sets (argmin) or top-k, not improvement sets
- **Compatibility metric**: Use L2 norm or activation MI, not cosine similarity

### 2. Multi-layer analysis is critical

Single-layer (layer 15) extraction likely misses where domain differentiation happens.
Layer-wise analysis (2602.05988) or activation MI (2505.14009) across all 30 layers
would reveal the optimal extraction point. The saturation phenomenon means earlier
layers may be more informative.

### 3. Equal-scale comparison is needed

Finance (scale=1.0) and legal (scale=4.0) cannot be compared with medical/code/math
(scale=20.0). Finding #235 showed ternary adapters have binary on/off scale behavior,
so the specialization structure may change dramatically at equal scales.

### 4. The three-level proxy chain extends

Finding #236: PPL doesn't predict MMLU accuracy (r=0.08).
Finding #238: MMLU accuracy doesn't predict behavioral quality.
Finding #240: PPL improvement sets don't predict specialization structure.

Each level in the proxy chain (PPL -> accuracy -> behavior) loses information.
Future experiments should measure what matters directly (behavioral quality, task
completion) rather than proxies (PPL, cosine, improvement sets).

## Recommended Follow-Up

### Priority 1: Sheaf cohomology with corrected cover (exp_sheaf_cohomology_dim)

- **Motivation**: Finding #240 identified that specialization sets + L2 norm are
  the correct inputs; the sheaf framework was not tested, only its prerequisites.
- **Literature**: Hansen & Ghrist (2110.03789) + sheaf theory survey (2502.15476)
- **Change**: Use S_i = {x : adapter i in top-2} as cover, L2 relative diff as
  compatibility metric. Extract from multiple layers (5, 10, 15, 20, 25).
- **Kill criterion**: If H^1 = 0 everywhere (no cohomological obstruction), bridge
  adapters are unnecessary and the sheaf framework adds no value over simple routing.

### Priority 2: Activation-MI adapter compatibility mapping

- **Motivation**: Cosine saturates at layer 15 (this experiment); ACM (2505.14009)
  showed activation MI is strictly more informative.
- **Literature**: ACM (2505.14009), AIM (2502.02421)
- **Change**: Compute pairwise activation MI across all 30 layers, identify which
  layers carry domain-specific information.
- **Kill criterion**: If MI is uniform across layers and pairs (no structure), then
  adapter differences are distributed, not localized.

### Priority 3: Equal-scale specialization retest

- **Motivation**: Finance dominated, legal partially specialized, but both at
  lower scales than the top 3. Finding #235 (binary scale behavior) suggests
  the landscape changes at equal scales.
- **Change**: Run specialization mapping with all adapters at scale=20 (or
  a scale within the "on" regime for all).
- **Kill criterion**: If specialization disappears at equal scales (all adapters
  tie on most samples), then routing is unnecessary — composition is trivially safe.
