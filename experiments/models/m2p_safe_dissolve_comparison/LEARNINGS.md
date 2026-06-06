# LEARNINGS: m2p_safe_dissolve_comparison

**Experiment:** exp_m2p_safe_dissolve_comparison  
**Finding:** #366 (supported)  
**Date:** 2026-04-07

---

## Core Finding

S3 (selective routing) is the Pareto winner among adapter promotion strategies: it merges all adapters into an enriched base and routes parity-class domains to the original base at inference, achieving 5/5 domain protection with zero merge overhead and only 2× memory cost. Naive loss-gating (S1) structurally degenerates to "do not promote" at micro scale — 0/10 adapters merged — because cross-domain interference exceeds τ=5% for every adapter on at least one domain.

---

## Why This Happened

**S1 failure (loss-gating = do-nothing at micro scale):** Each of the 10 cross-domain adapters raises at least one domain's loss above τ=5% when merged individually. The greedy sequential evaluation rejects all 10. This matches the Interference Additivity finding (#353): cross-domain interference is additive in the weight direction, so even a single adapter trained on domain A adds a ΔW that degrades domain B when base_B is already near-parity. At micro scale with 10 fully cross-domain adapters, no single adapter passes the gate. The negative result is scale-scoped: at larger scale with better-trained or more domain-specific adapters, some may pass.

**S4 failure on parity (null-space = wrong space):** The SVD null-space projection removes adapter directions that activate the competent domain's hidden states (Eckart-Young-Mirsky). This works in hidden-state space. But parity's SFT delta is only 0.0327 nats — the parity SFT barely moves the weights at all. The weight-space interference from cross-domain adapters is NOT aligned with the hidden-state null-space computed from parity's activation pattern. The projection misses it entirely, allowing +760% parity degradation despite SVD. This is a fundamental mismatch: hidden-state geometry ≠ weight-space interference geometry for near-trivially-competent domains.

**S3 success (routing = structural, not learned):** S3 explicitly routes parity-class domains (base_loss < τ) to the original (un-enriched) base at inference. This is a hard guarantee: the enriched copy's weight changes cannot affect parity if it never routes there. The result is structural protection without any quality cost for non-parity domains. The 2× memory cost (two base copies) is the price — acceptable on M5 Pro 48GB with a 4B model.

**Why parity keeps failing:** This is the third experiment (after #363, #364) where parity-class domains break. The structural reason is now clear: when SFT delta < 0.05 nats, ANY weight change to the base can push the domain below parity. Strategies that assume "small delta = safe" (S2: headroom scaling, S4: null-space projection) both fail because the interference is not proportional to the adapter's original domain influence.

---

## Confirming Evidence

- **Task Arithmetic (2212.04089, Ilharco et al.):** Merging task vectors (ΔW) via addition shows interference in the negative transfer direction. Our S3 fix is consistent with their "negation" concept: exclude a task vector from the merge if it would degrade a held-out domain.
- **TIES-Merging (2306.01708, Yadav et al.):** Identifies sign conflicts and magnitude pruning as the main causes of merge failure. Parity's near-zero SFT delta means its task vector has near-zero magnitude, making it maximally sensitive to sign conflicts from other adapters. Consistent with our S4 failure analysis.
- **Finding #353 (cross-domain graph):** Interference Additivity — N-way merges produce interference ~ Σ pairwise terms. Explains why every single adapter exceeds τ when cross-domain composition is involved.

---

## Contradicting Evidence

- **DARE (2311.03099, Yu et al.):** Random adapter weight pruning before merge dramatically reduces interference in LLM benchmarks. DARE was not tested here (would be S5). If random sparsification can reduce the interference below τ, S1 loss-gating might succeed with DARE preprocessing. This directly contradicts our conclusion that "S1 structurally degenerates to do-nothing" — it may only degenerate without sparsification.
- **AdapterFusion (2005.00247, Pfeiffer et al.):** Learned routing (attention-based fusion) achieves better multi-task quality than static routing (S3) by conditioning on input, not domain label. Static routing (S3) ignores within-domain variance. For macro scale with real queries, learned routing may outperform S3's hard domain-label rule.

---

## Alternative Approaches (Published Evidence)

1. **DARE + Loss-gating hybrid (2311.03099):** Apply random weight pruning (p=0.5–0.9) to cross-domain adapters before evaluating S1 loss gate. DARE reduces parameter magnitude by ~50-90%, potentially bringing interference below τ. Would convert S1 from "do-nothing" to "sparse-merge" at micro scale.

2. **Orthogonal Gradient Surgery (2001.12242, Yu et al.):** Project each adapter's update onto the orthogonal complement of all other adapters' update directions during merge, not during training. Conceptually similar to S4 but operates in weight gradient space, not hidden-state activation space — directly addresses the geometry mismatch identified as S4's failure mode.

3. **Fisher-weighted averaging (2111.09832, Matena & Raffel):** Merge adapters weighted by Fisher information diagonal, down-weighting parameters that are important for competent domains. Unlike S4 which projects away dangerous directions, this scales contributions without hard exclusion. Particularly relevant for parity-class domains where the Fisher magnitude for parity params would be high (small SFT delta = high curvature).

---

## Implications for Next Experiments

1. **Parity-guard as first-class design pattern.** Parity-class domains (SFT delta < 0.05 nats) must be explicitly identified and excluded from merge evaluation BEFORE any promotion strategy is applied. This is not a threshold trick — it is a structural pre-check. S3 enforces this at inference time; a cleaner approach would enforce it at merge time.

2. **Hybrid S3+S4 is the next natural experiment.** Route parity-class domains to original base via S3's structural routing, then apply S4 null-space projection for non-parity (hard) domains. S4 achieves 90.66% on non-parity domains with negligible cost overhead. This hybrid could yield 5/5 protection at S4 quality levels. Untested as of this experiment.

3. **DARE preprocessing to unblock S1.** If DARE sparsification (2311.03099) reduces cross-domain adapter magnitude below τ, loss-gating becomes viable. This would allow selective merging rather than full enrichment — lower memory cost than S3. Deserves a follow-up.

4. **Macro scale S1 behavior is unknown.** The S1 negative result is explicitly scoped to micro scale with 10 fully cross-domain adapters. With Qwen3-4B-scale adapters trained on real domains, some adapters may pass loss-gating. Do not generalize the negative result to macro without testing.

---

## Recommended Follow-Up

**exp_m2p_hybrid_s3s4** (P1): Test the hybrid strategy: S3 structural routing for parity-class domains + S4 null-space projection for non-parity domains. Motivation: S3 achieves 5/5 protection but at S0 median quality (89.17%); S4 achieves 90.66% on non-parity but destroys parity. Combining them should give 5/5 protection at S4 quality level. MATH.md prediction: hybrid should achieve ≥90.5% median (5/5 protected) vs S3's 89.17%. Cite: DARE (2311.03099) + Finding #366. Kill criteria: hybrid median > S3 median at same protection level.
