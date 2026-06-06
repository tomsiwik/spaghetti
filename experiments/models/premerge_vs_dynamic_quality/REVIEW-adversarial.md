# Peer Review: premerge_vs_dynamic_quality

## NotebookLM Findings

Skipped -- the experiment is already killed and self-diagnosed. The review below covers the same ground a NotebookLM deep review would surface.

## Mathematical Soundness

**MATH.md derivations are correct.** The core analysis is straightforward:

1. Pre-merge applies each expert at 1/N strength. Dynamic top-1 applies the selected expert at full strength. The signal ratio is N/k. This is trivially correct.

2. The quality gap bound -- "gap is bounded by specialization_gap * (1 - 1/N)" -- is stated informally but is approximately right under the assumption that LoRA deltas are small perturbations to base weights. The precise bound depends on the nonlinearity of the loss landscape, which is not derived. At 0% specialization this does not matter.

3. The FLOPs accounting (Section: Computational Cost) is correct for this architecture.

4. The worked example (Section: d=64, N=5, r=8) correctly applies the quadrature rule for orthogonal norms. The sqrt(5) factor for the pre-merge perturbation norm is right.

**No hidden assumptions detected.** The math is honest about what it does and does not prove.

## Novelty Assessment

**Low novelty, but that is appropriate for a methodology experiment.** This is essentially a micro-scale replication of the comparison implicit in LoRA Soups (Ostapenko et al., 2024) and Model Soups (Wortsman et al., 2022), both of which the paper cites. TIES-Merging (Yadav et al., 2023) is cited but not tested -- this is noted as a limitation, not overclaimed.

The experiment does not claim novelty. It claims to test a specific hypothesis within the SOLE framework. That is appropriate.

## Experimental Design

This is where the experiment breaks down, and the paper is honest about it. Several issues:

**1. The comparison is vacuous (acknowledged).** Oracle vs base gap is 0.0-0.1%. When experts contribute nothing, all composition strategies are equivalent by construction. The experiment correctly identifies this but still reports the kill. This is the right call -- the K2 kill is technically triggered but the evidence is not meaningful.

**2. The LoRA merging in `merge_loras` is mathematically incorrect for top-k routing.**
Lines 415-431 of `premerge_vs_dynamic.py` merge LoRA adapters by weighted-averaging the A and B matrices separately:

```python
merged['A1'][l] += w * lora['A1'][l]
merged['B1'][l] += w * lora['B1'][l]
```

The resulting delta is `(w1*A1 + w2*A2)(w1*B1 + w2*B2)`, which is NOT `w1*A1*B1 + w2*A2*B2`. The correct approach for weighted LoRA composition is to either (a) merge the full-rank deltas `w_i * A_i * B_i` and then apply, or (b) apply each LoRA separately and sum the outputs. The code for pre-merge (`premerge_deltas`, lines 371-381) correctly operates on full-rank deltas `dW1, dW2`. But the dynamic top-2 path uses the incorrect `merge_loras` on factor matrices.

At top-1 (k=1) this bug is invisible because there is only one adapter (w=1.0). At top-2, it introduces cross-terms `w1*w2*(A1*B2 + A2*B1)` that should not exist. Given 0% specialization, this bug has no observable effect on the results, but it means the top-2 numbers are not trustworthy even in principle.

**3. Per-sample routing is correct but expensive.** The `evaluate_ntp_loss_topk` function (lines 434-453) routes each sample individually using the sample's embedding, which is the right design. However, the embedding used for routing is the mean bag-of-words embedding from the frozen base model (`model.embed`), which collapses all sequence order information. At d=64 with V=32, these embeddings likely have poor discriminative power between domains, contributing to the routing being ineffective. This is a micro-scale limitation, not a design flaw.

**4. Domain selection is always the first N domains in a fixed order.** Line 572: `selected_domains = ALL_DOMAINS[:N]`. This means N=5 always uses the 5 code domains, N=8 adds 3 reasoning domains, etc. The experiment never tests a mixed selection. This is a minor issue since all domains show 0% specialization anyway, but at macro scale this ordering bias would need to be addressed.

**5. Controls are adequate for the hypothesis.** Base, oracle, pre-merge, top-1, and top-2 are all tested. The oracle serves as the upper bound. Multiple seeds (3) provide robustness. The N sweep from 5 to 20 covers the stated range.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry for `exp_premerge_vs_dynamic_quality` correctly records:
- K2 triggered (dynamic routing max +0.06% gap)
- K1 not triggered (pre-merge within 5%)
- Caveat about 0% specialization making the comparison vacuous
- Recommendation for macro-scale retest

This is properly recorded. The experiment is listed as `status: active` in HYPOTHESES.yml despite being killed -- this should be updated to `status: killed` for consistency.

## Integration Risk

Pre-merge vs dynamic routing is a serving strategy question, not a mechanism question. The experiment correctly identifies that the answer depends on expert specialization strength, which can only be determined at macro scale. The VISION.md architecture already plans for both strategies (pre-merge for small N, dynamic for large N), which is the right approach given the inconclusive micro results.

## Macro-Scale Risks (advisory)

1. **The top-2 LoRA merging bug must be fixed before macro retest.** Merge full-rank deltas, not factor matrices.

2. **Domain selection ordering bias.** Macro retest should use random domain subsets or stratified sampling across clusters.

3. **The crossover prediction (N~20 at 10% specialization) is back-of-envelope.** It assumes linear dilution, no interference, and uniform specialization across experts. Real experts will have heterogeneous quality. The macro test should measure per-domain gaps, not just means.

4. **Bag-of-words routing embeddings will not scale.** At macro scale, use the model's hidden state (or a dedicated router head), not mean token embeddings.

## Verdict

**PROCEED** (as killed experiment with methodology validated)

The experiment was correctly killed by K2 at micro scale. The paper is unusually honest about the vacuousness of its results -- it explicitly states that 0% specialization makes all comparisons meaningless, and it correctly identifies this as a micro-scale limitation rather than a mechanism failure. The methodology, infrastructure, and kill criteria logic are sound and ready for macro-scale retest.

Two issues to fix before macro reuse:

1. **Fix the top-k LoRA merging bug in `merge_loras`.** Merge full-rank deltas `w_i * A_i * B_i`, not factor matrices `w_i * A_i` and `w_i * B_i` separately. This produces incorrect cross-terms for k>1.

2. **Update HYPOTHESES.yml status from `active` to `killed`** with a note pointing to this review.

The directional finding -- that quality gap is proportional to specialization gap -- is the correct theoretical prediction and should be the hypothesis tested at macro scale. The projected crossover at N~20 with 10% specialization is plausible but needs empirical validation.
