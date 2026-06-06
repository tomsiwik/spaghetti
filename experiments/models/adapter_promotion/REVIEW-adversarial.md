# Peer Review: adapter_promotion (exp_adapter_promotion)

## Experiment Type
Guided exploration (Type 2)

**Claimed framework:** NRE Norm Preservation (Finding #275) + Structural Orthogonality (Finding #126)
**Claimed unknown:** Does the medical adapter retain >=70% of its standalone benefit after NRE composition with 4 other domain adapters?

## Hack Detector
- Fix count: 1 (NRE composition only). No stacking. CLEAN.
- Is MATH.md a proof or a description? **Description dressed in equations.** There is a "Retention Lemma" with no Theorem/Proof/QED block. The derivation (Section II) is a sequence of approximate equalities, not a proof. The word "approximately" and the symbol "~" do all the heavy lifting. This is a mechanism description with math notation, not a formal proof.
- Metric used as evidence: PPL (retained benefit percentage). PPL is a proxy, but reasonable for this question. The behavioral outcome (does the medical adapter remain useful?) is reasonably captured by PPL on domain-specific validation data.
- Kill criteria source: K828 threshold (70%) is explicitly stated as empirical product requirement, not derived from proof. K829 (1.5x) is loosely derived from NRE norm preservation. Honest about this.

## Self-Test Audit

MATH.md has a self-test section (Section V), but it is incomplete and does not follow the standard 6-question format:

1. **One-sentence impossibility property:** Not stated as a single property. Section V discusses what would falsify P1 and P2 separately, plus a future fix. This is a multi-part discussion, not a crisp impossibility property. **PARTIAL.**
2. **Cited theorems:** Finding #275 (NRE), Finding #126 (structural orthogonality), Finding #330 (scale behavior). These are internal findings, not external theorems. No external mathematical theorem is cited (no Johnson-Lindenstrauss, no Davis-Kahan, no Welch bound). The "retention lemma" is self-derived but not proven. **WEAK** -- relies entirely on internal findings without grounding in established mathematics.
3. **Predicted numbers:** Yes. eta >= 0.45 (orthogonality bound), K828 threshold 70%, K829 threshold 1.5x. These are specific and falsifiable. **PASS.**
4. **Falsification condition:** "If eta_medical < 0.70" with two candidate explanations. But the falsification targets the experiment outcome, not the math. A proper falsification would be: "The proof is wrong if adapters are orthogonal (cos < 0.03 per Finding #126) AND retention is below 1/sqrt(N)." **PARTIAL.**
5. **Hyperparameter count:** scale=20 is the sole hyperparameter, but it is NOT acknowledged as a free choice. MATH.md never questions whether scale=20 is appropriate for this experiment, despite Finding #330 already showing scale=20 causes OOD degradation. **FAIL** -- the most critical hyperparameter is unexamined.
6. **Hack check:** Not present as a numbered self-test item. **MISSING.**

## Mathematical Soundness

### Section II: "NRE Retention Lemma" -- Step-by-Step

**Step 1:** "For unit vectors with cos(DW_i, DW_j) ~ 0: ||(1/N) sum DW_i||_F ~ sigma_mean / sqrt(N)"

This approximation is correct for orthogonal vectors of equal norm. For unequal norms, the expression is sqrt(sum(sigma_i^2))/N, not sigma_mean/sqrt(N). These coincide only when all norms are equal. The MATH.md acknowledges adapters may have unequal norms (Section III) but uses the equal-norm formula anyway. **Minor issue** -- the direction is correct, the constant may differ.

**Step 2:** "||DW_composed||_F = sigma_mean * (sigma_mean / ||mean||) ~ sigma_mean * sqrt(N)"

This is the NRE rescaling: the composed norm is rescaled from ||mean|| to sigma_mean. Since ||mean|| ~ sigma_mean/sqrt(N), the rescaled norm is sigma_mean * sqrt(N). This is correct by construction of NRE.

**Step 3:** "<DW_composed, e_i> ~ (sigma_i / N) * (sigma_mean / ||mean||) ~ sigma_i / sqrt(N)"

This step assumes the inner product of the rescaled mean with unit vector e_i is (sigma_i/N) * rescale_factor. The (sigma_i/N) comes from the mean: <(1/N) sum DW_j, e_i> = sigma_i/N (since <DW_j, e_i> = 0 for j != i, and sigma_i for j = i). The rescale factor is sigma_mean/||mean|| ~ sqrt(N). So the projection is (sigma_i/N) * sqrt(N) = sigma_i/sqrt(N). **Correct under stated assumptions.**

**Step 4:** "eta_i = sigma_i/sqrt(N) / sigma_i = 1/sqrt(N)"

Straightforward. For N=5: eta = 0.447. **Correct.**

### Critical Gap: The derivation is internally consistent but NOT a proof.

The entire Section II is a sequence of approximations under the assumption of perfect orthogonality and equal norms. It does not:
- State formal conditions under which the approximations hold
- Bound the error terms
- Connect eta (directional retention in weight space) to PPL retention (the actual measurement)

The leap from "eta_i ~ 1/sqrt(N) directional retention" to "PPL benefit retention ~ eta" (Section III, first-order approximation) is a linear Taylor expansion of the loss around W_base. This linearization is unjustified for scale=20 adapters, which produce perturbations large enough to cause 72.8% PPL degradation EVEN IN ISOLATION. The linear regime assumption is catastrophically wrong at this scale.

### The Fundamental Error: Scale=20 is Outside the Linear Regime

**This is the most important finding of the review.**

The experiment uses LORA_SCALE = 20.0 (line 42 of run_experiment.py). At this scale:

1. **The medical adapter alone DEGRADES medical PPL from 6.107 to 10.553** (-72.8%). This means the adapter is not "beneficial" -- it actively hurts performance on its own domain.

2. MATH.md's entire framework assumes the adapter provides a benefit that is then partially retained under composition. When the adapter provides no benefit (indeed, it provides anti-benefit), the retention calculation is undefined. The code correctly detects this: `if (base_ppl - promoted_ppl) > 0 else 0` returns 0 because the denominator is negative.

3. The related experiment `expert_promotion` (at scale=5) shows the medical adapter IMPROVES medical PPL: 6.058 -> 5.249 (13.4% improvement). **Scale=5 is in the linear regime; scale=20 is not.**

4. Finding #330 already established that scale=20 with N=5 causes catastrophic MMLU degradation (-42pp). Finding #328 showed scale=20 degrades OOD. MATH.md cites Finding #330 in its prior results but then proceeds to use scale=20 anyway, without any justification for why scale=20 would be safe in this experiment when it has already been shown to be catastrophic.

**Verdict on mathematical soundness:** The derivation is correct in the linear regime but the experiment is run in a regime where the linearization is known to fail. The math predicts eta >= 0.45; the experiment gets 0%; and the explanation is not that the math is wrong but that the preconditions for the math are violated by the choice of scale=20.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Analysis:

| Prediction | Expected | Measured | Assessment |
|------------|----------|----------|------------|
| P1: Medical retention eta >= 0.45 | 45-70% | 0% | KILLED. But the premise is broken: the solo adapter makes PPL worse, so "retention" is undefined. |
| P2: Cross-domain <= 1.5x base PPL | <= 1.5x | 2.08-2.45x | KILLED. Massive degradation. |
| P3: Medical has highest norm | sigma_medical >= sigma_mean | Not measured | N/A -- informative only |

The table exists but the predictions are meaningless because the experiment's precondition (adapter provides benefit) is violated. The correct observation is not "retention = 0%" but "the adapter hurts performance; composition makes it hurt less (10.553 -> 9.216)."

**Missing critical measurement:** Solo PPL for each of the 5 adapters on their own domain at scale=20. If all 5 adapters degrade their own domains (which is likely given medical does), then the composition result is simply "NRE averaging of 5 harmful perturbations produces a less-harmful perturbation." This would be a completely different finding than what was hypothesized.

## NotebookLM Findings

NotebookLM was not used for this review. The issues are sufficiently clear from direct analysis.

## Novelty Assessment

**This experiment has essentially zero novelty.** It is a direct application of NRE composition (Finding #275) with scale=20, which was already shown to fail in Finding #330 and Finding #328. The "promotion" concept (permanently attaching one adapter) is interesting but is not what this experiment tests -- it tests uniform NRE composition of all 5 adapters, the same thing that Finding #330 already evaluated.

The related experiment `expert_promotion` already demonstrates the correct approach: promote at scale=5 (not 20), then train new adapters on the promoted base. That experiment is more novel and more rigorous.

**Prior art within the project:** Finding #330, Finding #328, and `expert_promotion` together already cover the territory this experiment explores.

## Code Correctness Issues

### Issue 1: Phase 1 result is misinterpreted
The results show `benefit_pct: -72.8`, meaning the adapter makes things WORSE. PAPER.md describes this as "benefit: -72.8%" but then proceeds to frame the K828 analysis as if the adapter had a benefit that vanished. The correct framing is: "the adapter at scale=20 actively degrades the medical domain."

### Issue 2: Retained benefit formula fails silently
Line 146: `retained_benefit = ... if (base_ppl - promoted_ppl) > 0 else 0`. When the adapter hurts performance, the denominator is negative, and the code returns 0. This is correct guard logic, but it means the 0% reported in results.json and PAPER.md obscures the real story: composition partially mitigates the damage (10.553 -> 9.216).

### Issue 3: promote_idx mismatch
Line 142: `attach_adapter(model, frozen_A, composed, 0, LORA_SCALE)` uses index 0, while line 118: `attach_adapter(model, frozen_A, adapter, promote_idx, LORA_SCALE)` uses promote_idx (also 0, since PROMOTE_DOMAIN="medical" is DOMAINS[0]). This is consistent but only by accident -- if DOMAINS order changed, Phase 2 would use a different A-matrix than Phase 1 for the composed adapter.

### Issue 4: No solo-adapter controls for other domains
The experiment measures base PPL and composed PPL for code/math/legal/finance, but never measures solo-adapter PPL for these domains at scale=20. Without this control, we cannot distinguish "composition causes degradation" from "scale=20 adapters individually cause degradation."

## Macro-Scale Risks (advisory)

Not applicable. This experiment was killed, and the root cause (scale=20 is outside the linear regime) is already established across multiple findings.

## Summary of Issues (by severity)

### BLOCKING (either of these alone justifies KILL)

1. **The experiment runs at scale=20, where the solo medical adapter DEGRADES its own domain by 72.8%.** The entire mathematical framework assumes the adapter provides a benefit that composition partially retains. When the adapter provides no benefit, the framework is inapplicable. This is not a "failed prediction" -- it is a violated precondition.

2. **Finding #330 and Finding #328 already established that scale=20 causes catastrophic degradation.** Running this experiment at scale=20 without mathematical justification for why scale=20 would work here is a waste of compute. The MATH.md acknowledges Finding #330 but does not address the contradiction.

### NON-BLOCKING

3. MATH.md has no Theorem/Proof/QED block. The "Retention Lemma" is a derivation sketch, not a proof.
4. Self-test is incomplete (missing hack check, missing one-property formulation).
5. No solo-adapter controls for non-medical domains prevent proper failure analysis.
6. The "guided exploration" framing (what is the unknown?) conflates two questions: "does NRE retain adapter benefit?" (answered by the math at eta=0.45) and "does the medical adapter provide benefit at scale=20?" (answered by phase 1: no).

## Verdict

**KILL -- justified and well-documented.**

The kill is correct but for a slightly different reason than stated. The stated kill reason (K828 FAIL: 0% retention, K829 FAIL: 2.08-2.45x degradation) is technically accurate but misses the root cause. The real finding is:

**At scale=20, the medical adapter DEGRADES its own domain by 72.8%. NRE composition cannot retain a benefit that does not exist. The entire premise of "adapter promotion via composition" is invalid at this scale.**

The experiment `expert_promotion` already demonstrates the correct approach: promote at scale=5, where the adapter actually helps, and train new adapters on the promoted base. That experiment should be the focus of further work, not this one.

### Impossibility structure for the finding record

The mathematical impossibility is: NRE composition preserves norm but averages direction. At scale=20, each adapter's delta is large enough that ΔL ~ -<grad, ΔW> fails as a first-order approximation -- the higher-order terms dominate and cause PPL degradation. Composition of N deltas, each individually harmful, produces a harmful composition. No amount of norm rescue can fix this because the problem is directional (the adapter points in a harmful direction at this scale), not scalar (the adapter has the wrong magnitude).

This is a known consequence of the nonlinear regime: when ||E||_op / delta >= 1 (where delta is the spectral gap), the Davis-Kahan bound becomes vacuous and subspace rotation can be arbitrary. The `expert_promotion` MATH.md explicitly calculates this: at scale=20, sin(theta) <= 1.0 (vacuous), confirming the failure is inevitable.

---

## Audit-Rerun Closure Review (2026-04-18)

**Re-review trigger:** researcher submitted the experiment under the
`audit-2026-04-17-rerun, lora-scale` sweep as a *closure* (no rerun executed;
structural theorems argue the kill is robust to the scale fix).

### Adversarial checklist re-run against the closure artifacts

- (a) `results.json.verdict == "KILLED"` matches DB `status=killed` ✓
- (b) `all_pass=false` matches `killed` ✓
- (c) PAPER.md verdict line consistent (KILLED throughout) ✓
- (d) `rerun_executed=false` is flagged honestly in `results.json.source`;
  not disguised as a full run — no `is_smoke` dishonesty ✓
- (e) MATH.md git timestamp (Apr 17 10:52) pre-dates all closure artifacts
  (Apr 18 09:40). K828/K829 were **not** relaxed post-hoc ✓
- (f) K828 (retention) and K829 (cross-domain catastrophe) measure distinct
  quantities; no tautology ✓
- (g) K-IDs 828/829 in results.json match DB and MATH.md ✓
- (h) Code uses `pierre.compose_adapters` (NRE averaging) — not the buggy
  `sum(lora_A)` / `add_weighted_adapter("linear")` pattern ✓
- (i) `LORA_SCALE = 20.0` IS hard-coded (antipattern). **Closure survives:**
  C1 derives η ≤ 1/√N as a scale-invariant ratio (scale cancels in the
  retention fraction), so the scale-fix cannot rescue K828. The antipattern
  is correctly identified as cosmetic here.
- (j)-(l), (m) not applicable (no routing, no shutil, no hardcoded pass,
  model path is runtime-selected consistently)
- (m2) MLX idioms present (`mx.eval`, `mx.clear_cache`, `mx.set_memory_limit`,
  `mx.reset_peak_memory`); skills evidence ok ✓
- (r) PAPER.md has prediction-vs-measurement table ✓

### Closure theorems — soundness

- **C1 (orthogonality retention ceiling):** MATH.md §II derives η = 1/√N
  under orthogonality (Finding #126). Scale enters as a common multiplicative
  factor on every ΔW_i, so the retention ratio η is scale-invariant. For N=5,
  η ≤ 0.447 < K828's 0.70. Sound.
- **C2 (sibling covers correct mechanism):** `exp_expert_promotion` tests the
  merge-then-compose alternative and is SUPPORTED (Finding #333) at scale=5.
  The research question already has a SUPPORTED answer via an architecturally
  distinct mechanism. Sound.
- **C3 (K829 alone insufficient):** `all_pass` requires both K828 and K829.
  Scale-fix may recover K829 but leaves K828 blocked by C1. Sound.

### Fit to closure-rule family

Fourth kill-robust closure of the 2026-04-17 sweep following
`exp_depth_routed_adapters`, `exp_mlp_only_per_token_routing`, and
`exp_ridge_router_single_pass_e2e`. Generalisation `base-ceiling-blocks-routing`
/ `ap-oracle-ceiling-blocks-headroom` holds: a structural upper bound on the
composition operator below the kill threshold cannot be rescued by a
hyperparameter fix.

### Verdict (closure)

**PROCEED — KILL stands.** Closure theorems are internally consistent,
mathematically sound under the cited findings, and correctly scope-limit the
lora-scale fix. Artifacts complete (MATH.md, run_experiment.py, results.json,
PAPER.md with Audit-Rerun Closure addendum, REVIEW-adversarial.md, LEARNINGS.md
with closure section). No further rerun needed.
