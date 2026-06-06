# Peer Review: Tiny Routing Heads N=24

## Experiment Type
Frontier extension (N=5 proven result extended to N=24)

## Hack Detector
- Fix count: 1 (same binary head mechanism as N=5, no new tricks). CLEAN.
- Is MATH.md a proof or a description? Description dressed in equations. There is no Theorem/Proof/QED block. Cover's theorem and VC dimension are cited but never composed into a formal guarantee. The "proof" is: "Cover says 24 << 5120 so separability holds." This is a one-line invocation, not a derivation.
- Metric used as evidence: Top-1 routing accuracy + routed PPL vs uniform PPL. Top-1 accuracy is a reasonable proxy. PPL comparison is the correct behavioral outcome.
- Kill criteria source: K584 (60% threshold) is partially derived from the argument that below 60%, top-2 selection cannot concentrate weight on relevant adapters. K585 (routed < uniform) is directly behavioral. K586 (10% overhead) is an engineering constraint. Reasonable but see critique below.

## Self-Test Audit

1. **One-sentence impossibility property:** "Base model hidden states in R^2560 have sufficient dimensionality for 24 linearly separable regions (Cover's theorem)." This is ONE property. However, it guarantees the WRONG thing -- it guarantees per-head separability, not cross-head ranking. The experiment killed on ranking, not separability. The impossibility property was aimed at the wrong failure mode. FLAG.

2. **Cited theorems:** Cover's Function Counting Theorem (1965), VC Dimension (1971), Universal Approximation (1989). All real. However, Cover's theorem guarantees that N random points in R^d are linearly separable with high probability when N << 2d. The application here conflates "domain centroids are separable" with "domain DISTRIBUTIONS are separable by a binary classifier." These are different problems. Cover's theorem applies to point separation, not distribution discrimination with overlapping supports. PARTIALLY VALID.

3. **Specific predictions:** Average accuracy >70% (measured 87.2%, pass), min accuracy >50% (measured 77.7%, pass), top-1 >60% (measured 39.6%, fail), overhead ~11% (measured 6.8%, pass), routed PPL < uniform (fail). Predictions are specific and falsifiable. PASS.

4. **Falsification condition:** "The framework is wrong if base model hidden states do NOT cluster by domain at N=24." But the hidden states DO cluster (87.2% average head accuracy proves this), and the experiment still killed. The falsification condition was too narrow -- it did not anticipate the actual failure mode (cross-head ranking collapse). The framework was wrong in a way the falsification condition did not cover. PARTIALLY VALID.

5. **Hyperparameters added:** 0 new. Reusing N=5 settings. PASS.

6. **Hack check:** No new fixes. Single mechanism at larger scale. PASS.

## Mathematical Soundness

The MATH.md is a well-structured frontier-extension document that correctly identifies the proven result (N=5, Finding #54) and the mathematical gap (scaling to N=24 with overlapping domains). For a frontier extension, this is adequate framing.

**What holds:**
- Cover's theorem citation is correct: 24 << 5120 means random points are almost surely linearly separable in R^2560
- VC dimension argument is valid: a linear classifier in R^2560 can shatter far more than 24 points
- Universal approximation for 2-layer MLP with h=32 is valid
- FLOPs analysis (0.165% overhead) is correctly computed
- The "readout problem, not representation learning problem" framing is insightful

**What does not hold:**

1. **The critical gap: separability does not imply rankability.** Cover's theorem guarantees that each head CAN find a separating hyperplane for its domain. But 24 independent binary classifiers optimized with BCE loss produce uncalibrated logits. The theorem says nothing about whether the maximum-scoring head will be the correct one. This is the detection-vs-ranking distinction that PAPER.md correctly identifies post-hoc, but MATH.md should have predicted a priori. The mathematical framework had a blind spot.

2. **The false positive rate analysis was missing from MATH.md.** A simple calculation could have predicted the failure: if each head has FPR = p, then for N heads, the expected number of false positives per input is (N-1)*p. At N=24, p=0.13: expected false positives = 23*0.13 = 2.99. The probability that at least one false positive outscores the true positive depends on score distributions, but with ~3 competitors, failure is plausible. This calculation should have appeared in the predictions.

3. **The 1-vs-23 class imbalance was acknowledged in assumptions (Section E, point 3) but not analyzed quantitatively.** MATH.md says "Could fail if the positive class is surrounded by many similar negatives" but does not compute the probability of this happening or predict how many heads would degenerate.

**Missing from MATH.md that should have been there for a frontier extension:**
- A probabilistic model of cross-head ranking accuracy as a function of N, given per-head accuracy a and false positive rate p
- Prediction of how many heads would learn "reject everything" due to class imbalance
- Analysis of when decentralized routing (independent heads) provably fails vs centralized routing

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| Avg head accuracy >70% | 87.2% | Match |
| Min head accuracy >50% | 77.7% | Match |
| Top-1 routing >60% | 39.6% | FAIL by 1.5x |
| Overhead ~11% | 6.8% | Match (better) |
| Routed PPL < uniform | Routed 10.13 > uniform 10.08 | FAIL |

The table is present and honest. The disconnect between per-head accuracy (prediction matched) and routing accuracy (prediction failed by 1.5x) is the key finding and is well-analyzed in PAPER.md.

## NotebookLM Findings

Skipping NotebookLM deep review for this killed experiment -- the failure is clear-cut and the analysis is thorough enough to review directly.

## Novelty Assessment

**Prior art:** The PAPER.md correctly cites LoRAuter (2601.21795) which uses embedding-based routing scaling to 1500+ adapters, and MoLoRA (2603.15965) which uses shared per-token routing. Both use centralized routing, which this experiment's failure validates as the correct approach at scale.

**Delta over existing work:** The main contribution is the empirical demonstration that decentralized binary heads fail at N=24 despite high per-head accuracy, with a clear mechanistic explanation (false positive cascade). This is a useful negative result. The detection-vs-ranking distinction and the identification of the N threshold where decentralized routing breaks down are genuinely informative for future routing design.

**What was already known:** That independent binary classifiers have calibration problems when used for ranking is well-established in the ML literature (isotonic regression, Platt scaling, etc.). The experiment rediscovers this in the adapter routing context, which has value for the project even if not novel to the field.

## Macro-Scale Risks (advisory)

1. At macro scale (N=50, N=100), the false positive cascade worsens quadratically. Any future routing approach MUST use centralized scoring or calibrated ranking.
2. The adapter quality problem (individual PPL ~= base PPL) is a confound. At macro scale with properly specialized adapters, routing accuracy matters more. The experiment cannot distinguish "routing fails because heads are bad" from "routing fails because adapters are indistinguishable."
3. The 7 heads that learned "reject everything" suggest that at larger N, the fraction of degenerate heads will increase, making decentralized approaches progressively worse.

## Additional Critiques

### The Adapter Quality Confound Is Underweighted

The results show avg_individual_ppl = 10.09 vs avg_base_ppl = 10.06. The adapters provide essentially zero specialization. This means:
- Routed PPL vs uniform PPL is comparing noise to noise
- Even perfect routing would show minimal PPL improvement
- The K585 kill criterion (routed PPL < uniform) is testing routing quality through a lens of adapters that have nothing to offer

This does NOT invalidate the kill on K584 (routing accuracy is independently meaningful), but it makes the K585 failure less informative than it appears. The 0.42% PPL difference (10.13 vs 10.08) is within noise for adapters that provide 0.3% change from base. PAPER.md mentions this in limitations but should have weighted it more heavily.

### The Kill Threshold for K584 Could Be Questioned

The 60% threshold is justified as "below 60%, top-2 selection cannot concentrate weight on relevant adapters." But top-2 accuracy is 54.2%, which means more than half the time at least one correct adapter is in the top-2. The argument for 60% as a hard kill is reasonable but not derived from a formal bound. For a frontier extension, this is acceptable.

### Root Cause Analysis Is Excellent

The PAPER.md analysis of the false positive cascade is the best part of this experiment. The progression from "high per-head accuracy" to "low routing accuracy" to "independent binary classifiers solve DETECTION not RANKING" to "decentralized routing needs centralized arbitration at scale" is a clean chain of reasoning. The confusion matrix patterns (agriculture as a false-positive attractor, reject-everything heads) provide concrete mechanistic evidence.

### The Impossibility Structure Is Correctly Identified

"Independent binary classifiers cannot solve competitive ranking without calibration" is the right structural insight. This rules out an entire class of routing approaches and points clearly toward shared/centralized routers.

## Verdict

**KILL -- Confirmed**

The experiment is correctly killed. The kill criteria were hit cleanly (K584: 39.6% < 60%, K585: routed PPL worse than uniform). The root cause analysis is thorough and identifies a genuine structural limitation of decentralized binary routing at scale.

**Strengths of this experiment:**
1. Clean frontier-extension framing with explicit proven result (N=5) and gap (N=24)
2. Honest prediction table with clear hits and misses
3. Excellent post-hoc mechanistic analysis of WHY the failure occurred
4. Correctly identifies the impossibility structure (detection != ranking)
5. No hack accumulation -- same mechanism tested at scale, found wanting
6. Finding #191 correctly recorded as killed, status capped at killed (appropriate for frontier extension that fails)

**Weaknesses:**
1. MATH.md lacks a formal proof (no Theorem/Proof/QED). For frontier extension, this is less severe but still a gap. A simple probabilistic model of false positive cascade could have predicted failure BEFORE running the experiment.
2. The self-test falsification condition targeted the wrong failure mode (hidden state clustering rather than cross-head ranking).
3. The adapter quality confound (adapters barely differ from base) makes K585 less informative than presented.
4. Cover's theorem is applied to distributions, not points, which is a subtle misuse (though the conclusion -- separability is easy -- is still directionally correct).

**For future work:** The impossibility structure (independent binary classifiers fail at ranking without calibration) should be formalized as a theorem with a proof. Specifically: given N independent binary classifiers with accuracy a and FPR p, derive the probability that argmax over scores selects the correct classifier as a function of N. This would give a precise N_max where decentralized routing fails, turning this killed experiment into a proven negative result.

---

## Audit-Rerun Closure Review (2026-04-18)

Reviewer on `experiment.done exp_tiny_routing_heads_n24: KILLED (audit-rerun closure)`.

**State:** `git status` shows only PAPER.md modified (+109 lines append-only). MATH.md, run_experiment.py, results.json, LEARNINGS.md untouched. DB status=killed with 2026-04-18 evidence row. KC IDs 584/585/586 consistent across DB ↔ MATH.md ↔ PAPER.md ↔ results.json. `results.json["verdict"]="KILLED"` matches PAPER.md verdict (lines 21, 272).

**Adversarial checklist:** (a)-(s) all PASS.
- (a-c) verdict consistency: results.json KILLED = PAPER KILLED = DB killed. No PROVISIONAL qualifier.
- (d) no `is_smoke` flag present.
- (e) `git log MATH.md` shows single commit — no KC swap.
- (f) K585 (routed vs uniform PPL) and K584 (top-1 accuracy) are non-tautological behavioral metrics.
- (h) no composition math bugs (`add_weighted_adapter`, `sum(lora_A`, `shutil.copy` all absent).
- (i) `LORA_SCALE=20` present at line 43 — inherited from upstream `exp_real_data_25_domain_adapters` adapters, NOT set for this routing-head experiment. Closure's Thm C1 (zero-headroom) is scale-independent: the argument rests on `|individual_ppl − base_ppl| ≈ 0.03` being smaller than estimation noise, which persists regardless of scale. Kill-direction preserved. Acknowledged honestly in closure §Antipattern.
- (k-l) no shutil.copy, no hardcoded pass dict.
- (m) BitNet-b1.58-2B-4T in MATH.md matches code.
- (o) N_val = 20 × 24 = 480 samples, well above 15.
- (r) prediction-vs-measurement table present (PAPER.md lines 13-19).

**Closure theorems:**
- **Thm C1 (formal kill-invariance on K585):** avg_individual_ppl 10.09 vs avg_base_ppl 10.06 → adapter specialization signal ≈ 0.03 PPL units, below estimation noise over 20 val samples per domain. Any routing mechanism (BCE-fixed or otherwise) interpolates between base and individual adapters; headroom < noise floor. **Sound** — same abstract structure as mem-021 (CEILING-HEADROOM COLLAPSE) at the behavioral-metric ceiling.
- **Thm C2 (directional kill on K584):** BCE fix would optimistically lift per-head acc 87→90%, FPR 0.13→0.10 → expected FPs = 23·0.10 = 2.3 → top-1 ≈ 30-50% < 60% K584 threshold. Honestly framed as directional, not formally invariant. Failure direction.
- **Thm C3 (structural uncalibration):** cites original REVIEW §"Impossibility Structure" (independent binary classifiers cannot solve competitive ranking without calibration). BCE fix addresses detection, not calibration. Kill-mechanism preserved. Consistent with LEARNINGS "four routing kills all lack cross-expert calibration."

**Code-bug acknowledged, fix deferred:** MLX `nn.losses.binary_cross_entropy(mx.sigmoid(logit), target)` double-applies sigmoid (defaults to `with_logits=True`). Properly routed to `exp_followup_bce_with_logits_routing` which exists in DB as the clean isolated venue. No rerun needed here — kill is invariant.

**Verdict: KILL (reaffirmed).**

**Route:** `review.killed` → Analyst.

**Open threads for analyst:**
- **Promote mem-021 CEILING-HEADROOM COLLAPSE to 3-instance confidence.** Instances: (1) oracle-router ceiling test-time [exp_depth_routed_adapters], (2) orthogonality ceiling training-time [exp_flat_lora_training], (3) adapter-specialization ceiling behavioral-metric [this experiment]. Three abstract instantiations of "mechanism layered on baseline at the mechanism's theoretical ceiling → zero headroom."
- Candidate closure-rule Finding distinct from #191: "Any routing mechanism proposing to beat uniform-weight composition where individual adapter PPL ≈ base PPL has zero K2-style headroom; behavioral PPL-comparison KCs are structurally unreachable." The adapter-specialization-ceiling rule.
- No new Finding strictly required — mem-021 promotion may suffice. Analyst judgment.
- Low-priority code-bug fix in line 331 belongs in `exp_followup_bce_with_logits_routing`, not here.

**Backlog state after this iteration:** P=1 open remaining: `exp_p1_t5_user_local_training` (last P=1, macro). P=2 active open: ~73 → ~72 (this P=2 closed). Nothing stuck in `experiment list --status active`.
