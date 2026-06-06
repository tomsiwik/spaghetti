# Peer Review: Split Domain Specialization

## NotebookLM Findings

Skipped -- the experiment is already researcher-killed on both criteria with clear negative results. Deep review not warranted for a kill-validation pass.

## Mathematical Soundness

### Convergence speed formalization (Section 2.2 of MATH.md)

The definition `S_conv(L) = min{s : L(s) <= 1.01 * L(S_final)}` is well-defined given a discrete evaluation schedule. The speedup formula `(S_conv(indep) - S_conv(split)) / S_conv(indep)` correctly computes relative improvement.

**Minor issue**: convergence step is quantized to the eval_every=25 grid. With only 16 evaluation points over 400 steps, the resolution is 6.25% per grid step. A "10% speedup" threshold requires distinguishing 125 vs 139 steps, but the measurement granularity is 25 steps. The actual per-seed results (150 vs 125, 75 vs 75, 150 vs 175) show this quantization effect. This does not invalidate the kill -- the mean is exactly 0.0%, far from the 10% threshold -- but it means the measurement is coarser than the threshold implies.

### Jaccard formalization (Section 2.3)

The binary Jaccard on capsule activation sets is correctly defined. The aggregation (mean across layers and children) is straightforward. The threshold (J < 0.95) is reasonable for detecting non-trivial specialization.

**The activation profiling code is correct but aggressive**: `profile_leaf_activations` accumulates active capsule indices across 20 batches of 32 samples. A capsule that fires on even a single token across 640 samples counts as "active." This biases Jaccard toward 1.0 because any capsule that fires rarely on domain A but frequently on domain B still enters both sets. The MATH.md Assumption 1 and PAPER.md Limitation 3 both acknowledge this. This is a real methodological limitation, not fatal for the kill decision but worth noting: a frequency-weighted metric (e.g., cosine similarity of activation frequency vectors) would be more sensitive.

### Information-theoretic bound (Section 3.2)

The estimate `I(domain; bigram) ~ log2(26) / L_avg ~ 0.78 bits/position` is a rough upper bound, not a tight calculation. The actual mutual information depends on the conditional distribution of bigrams given domain, which was not computed. But the qualitative conclusion (low signal beyond first character) is sound for character-level name data with a-m/n-z split.

### Section 4.1-4.2 analysis

The argument that split children inherit ~50% relevant features (random partition across domains) is qualitatively correct. The formula `S_conv(split) ~ S_conv(indep) * (1 - alpha * 0.5)` is a hand-wave, not a derivation -- alpha is undefined and the linear approximation is unjustified. But this is presented as intuition, not a theorem, and the experiment directly tests the claim, so this is acceptable.

**Overall mathematical soundness: adequate for a micro-experiment.** No errors in the derivations that are presented as formal. The informal arguments are directionally correct.

## Novelty Assessment

This experiment has minimal novelty claims and makes none. It is a follow-up ablation testing a specific limitation identified in the parent experiment (split_leaf_actual). The question "does split initialization help domain-specific fine-tuning?" is a natural next step from "does split initialization help mixed fine-tuning?" (answered: no advantage, by parent experiment).

The Jordan & Jacobs (1994) HME citation is appropriate. No prior art was found that specifically tests weight-inheritance vs random-init for domain specialization in capsule-style MoE architectures at this scale. The closest work would be knowledge distillation or progressive growing literature, but the experiment does not claim novelty over those.

**No reinvention concern.** The experiment correctly builds on split_leaf_actual infrastructure.

## Experimental Design

### Does it test what it claims?

Yes. The experiment directly compares split vs independent initialization under domain-specific fine-tuning, measuring both convergence speed and activation specialization. The two kill criteria match the HYPOTHESES.yml entry exactly.

### Are the controls adequate?

**Good**: The independent condition loads the same base weights for non-target layers, reinitializes only the target leaf pair, and uses identical training protocol (same alternating schedule, same LR, same steps, same frozen layers). The independent condition seeds at `seed + 1000` for the random reinitialization, which is appropriate.

**Good**: Three seeds provide minimal but acceptable statistical coverage for a micro-experiment.

**Concern -- training protocol conflates two effects**: The alternating-batch scheme (odd steps domain A, even steps domain B) trains BOTH children on BOTH domains. The gate starts at 50/50 (zero-initialized for independent, inherited for split) and must learn to route. The paper's comment at line 85-92 of `run_experiment.py` acknowledges this: "since we start from 50/50 routing, both children see both domains equally." This means the experiment tests whether the gate learns domain routing AND whether capsules specialize, simultaneously. A cleaner design would force routing (child 0 always sees domain A, child 1 always sees domain B) and separately test whether specialization emerges. However, the forced-routing design would not test the realistic scenario where routing must be learned, so the chosen design is defensible for the stated hypothesis.

**Concern -- convergence metric uses mixed validation**: `curve_mixed` averages val_loss_A and val_loss_B. This is correct for measuring overall quality but could mask domain-specific convergence differences. If split converges faster on domain A but slower on domain B, the mixed metric would show no advantage. The per-domain curves (curve_A, curve_B) are collected but not used in the kill criteria. This is a minor design choice that does not invalidate the kill.

### Could a positive result be explained by a simpler mechanism?

Not applicable -- the result is negative.

### Kill decision validity

The kill is clean:
- KC1: 0.0% mean speedup, threshold was >10%. Not close.
- KC2: J=0.975, threshold was <0.95. Not close.
- Both conditions (split AND independent) show near-identical behavior, ruling out implementation bugs that would differentially affect one condition.
- The researcher correctly identifies this as a data/scale limitation rather than a mechanism failure.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry `exp_split_domain_specialization` has:
- `status: killed` -- correct
- `kill_criteria` match KC1 and KC2 from PAPER.md -- correct
- `depends_on: [exp_split_leaf_actual]` -- correct lineage
- Evidence summary accurately reflects the experimental results

The consecutive_kills counter shows 2, with this as the last killed node. The hypothesis graph is consistent.

## Macro-Scale Risks (advisory)

1. **The a-m/n-z domain split is too weak for this test.** At macro scale with genuinely distinct domains (code vs medical vs legal), the Jaccard metric would have actual signal to work with. The micro kill does NOT imply domain specialization is impossible -- it implies this particular micro task cannot test it.

2. **Binary Jaccard remains a weak metric even at macro scale.** With 128+ capsules, a frequency-weighted specialization index (e.g., mutual information between domain identity and per-capsule activation magnitude) would be more informative than binary fire/no-fire Jaccard.

3. **Gate learning dynamics are untested.** The alternating-batch scheme did not produce learned domain routing at micro scale (both children see both domains equally). At macro scale, this could either resolve naturally (stronger domain signal) or remain a bottleneck. Explicit domain-conditioned routing should be compared.

4. **The convergence speed test should use wall-clock time, not step count, at macro scale.** Split children inherit features and may require fewer steps but each step could be more expensive if the inherited features create harder optimization landscapes.

## Verdict

**PROCEED** (with the kill)

The kill decision is correct and well-justified. Both kill criteria fail by wide margins. The experimental design is sound within micro constraints. The methodology (split vs independent, alternating domains, Jaccard profiling) is correctly implemented and the controls are adequate.

The researcher's interpretation is appropriately scoped: this is a negative result about domain specialization at micro scale with character-level name data, not a refutation of domain specialization in general. The constructive conclusion -- that split's value lies in function-preserving tree growth (validated by parent experiment), not in accelerating domain specialization -- is well-supported.

No revisions needed. The experiment should remain killed, and the finding should inform macro-scale experiment design: use genuinely distinct domains and frequency-weighted specialization metrics.
