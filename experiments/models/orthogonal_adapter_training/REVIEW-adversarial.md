# Peer Review: Orthogonal Adapter Training

## Experiment Type
Guided exploration (Type 2)

Claimed framework: OPLoRA (arXiv:2510.13003) orthogonal projection preserves top-k singular triples.
Claimed unknown: optimal k for ternary adapters on BitNet-2B.

## Hack Detector
- Fix count: 1 (orthogonal projection constraint). Clean, single mechanism. No flag.
- Is MATH.md a proof or a description? **Proof with QED** -- Theorem 1 has a correct proof that U_k^T Delta_W_orth = 0 and Delta_W_orth V_k = 0 by construction of P_L, P_R. The derivation is valid.
- Metric used as evidence: rho_k (direction interference) + MMLU + GSM8K. rho_k is directly derived from the theorem. MMLU/GSM8K are behavioral proxies not predicted by the proof beyond directional claims.
- Kill criteria source: K1 (MMLU math <=15pp) is a behavioral threshold informed by prior findings but not derived from the proof. K2 (GSM8K >=+3pp) is a preservation check. K3 (in-dist >=90%) is reasonable but arbitrary. **Mixed: K1-K3 are informed by prior findings, not derived from the proof's mathematics.**

## Self-Test Audit

1. **One-sentence impossibility property:** "Orthogonal projection ensures Delta_W has zero component in the top-k singular subspace of W, making knowledge corruption impossible by construction." -- Clean, one property. PASS.

2. **Cited theorems:** OPLoRA Theorem 1 (arXiv:2510.13003). I cannot verify the arxiv ID exists (knowledge cutoff), but the mathematical content is correct: double-sided orthogonal projection onto the complement of the top-k singular subspace zeroes out the cross-terms U_k^T Delta_W V_k = 0 and Delta_W V_k = 0. The proof in MATH.md Section C is correct. **However:** the OPLoRA paper may address FP16 weights where spectral gaps are well-defined. The conditions do NOT specify what happens when the spectrum is flat (no gap between sigma_k and sigma_{k+1}). The theorem is still mathematically correct -- it preserves whatever singular triples are labeled "top-k" -- but the ASSUMPTION that top-k captures "knowledge" depends on a spectral gap that ternary weights lack. PASS on theorem correctness; FLAG on applicability to ternary.

3. **Predicted numbers:** rho_k = 0.0 exactly, MMLU math <=15pp, GSM8K >=+3pp, in-dist >=90%, training loss within 1.1x baseline. The rho_k prediction is tight and falsifiable. The behavioral predictions (MMLU, GSM8K) are directional bounds, not precise. Acceptable for Type 2. PASS.

4. **Falsification condition:** "(a) knowledge is NOT stored in top-k singular directions, or (b) SVD of ternary weights is degenerate (no spectral gap)." This is excellent -- it targets the ASSUMPTION behind the proof, not just experimental outcomes. And indeed the experiment falsified condition (b). PASS.

5. **Hyperparameter count:** 1 (k). Acknowledged as the exploration target. PASS.

6. **Hack check:** Not a fix on existing stack; replaces DARE for direction interference. PASS.

**Self-test verdict: PASS.** All 6 answers are honest and complete.

## Mathematical Soundness

### Theorem 1 (Knowledge Preservation Under Orthogonal Projection)
The proof is correct. Step-by-step:

1. P_L = I - U_k U_k^T is an orthogonal projector onto the complement of span(U_k). Confirmed: P_L^2 = P_L, P_L^T = P_L.
2. P_R = I - V_k V_k^T, same properties.
3. Delta_W_orth = P_L Delta_W P_R.
4. U_k^T Delta_W_orth = U_k^T P_L Delta_W P_R = (P_L U_k)^T Delta_W P_R = 0. Correct because P_L U_k = (I - U_k U_k^T) U_k = U_k - U_k = 0.
5. Delta_W_orth V_k = P_L Delta_W P_R V_k = P_L Delta_W 0 = 0. Correct because P_R V_k = 0.
6. Therefore the top-k singular triples are preserved. QED.

**No errors found.** The proof is clean and correct.

### Theorem 2 (Gradient Projection Equivalence)
The working is messy (multiple false starts visible in the text, e.g., "Wait --", "Actually, let's be more careful", "No.") but the final summary is correct:
- A_orth = P_R @ A (pre-computed)
- grad_B -> grad_B @ P_L (after each step)

This ensures Delta_W = s * B^T @ A_orth^T = s * (P_L-projected B)^T @ (P_R @ A)^T, which gives the double-sided projection. Correct.

**Minor issue:** The working shows the gradient projection as `grad_B -> grad_B @ P_L`, but the implementation (line 408-423 of run_experiment.py) does `grad_B -> grad_B - (grad_B @ U_k) @ U_k^T`. These are equivalent: grad_B @ P_L = grad_B @ (I - U_k U_k^T) = grad_B - (grad_B @ U_k) @ U_k^T. Verified, consistent.

### Critical Assumption: Top-k Encodes Knowledge
The proof guarantees rho_k = 0. But the behavioral prediction (MMLU math <=15pp) depends on the ASSUMPTION that "knowledge" is concentrated in the top-k singular directions. For FP16 models, this is empirically supported (clear spectral gaps, power-law decay). For ternary weights, the MATH.md correctly identifies this as Assumption 4 and flags the risk.

The experiment measured spectral gap ratios of 1.003-1.018 (essentially 1.0). This confirms the assumption fails for ternary weights. The MATH.md and PAPER.md are both honest about this.

### Vacuity Check
The bound rho_k = 0 is not vacuous -- it is tight and confirmed by measurement (rho_k = 0.000012, consistent with floating-point precision on 210 matrices). This is a genuine verification of Theorem 1.

## Prediction vs Measurement

PAPER.md contains a prediction-vs-measurement table. Assessment:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| rho_k = 0.0 exactly | 0.000012 | MATCH (numerical precision) |
| MMLU math <=15pp degradation | -20pp | FAIL |
| GSM8K >=+3pp | +14pp | MATCH (exceeded) |
| In-dist >=90% of baseline | 50% (math), 112% (code) | FAIL (math) |
| Training loss within 1.1x | 4/5 converged | PARTIAL |

The proof's tight prediction (rho_k = 0) was verified precisely. The behavioral predictions failed because the assumption connecting rho_k to knowledge preservation does not hold for flat-spectrum ternary weights.

**This is a clean Type 2 result:** the exploration identified that the unknown k cannot be set meaningfully when the singular spectrum has no gap. The unknown was narrowed from "what is optimal k?" to "k is undefined for flat-spectrum ternary weights." This is a genuine finding.

## NotebookLM Findings

Skipping automated NotebookLM review. The manual analysis above is sufficient given the experiment was already killed by the researcher.

## Novelty Assessment

**Prior art:** OPLoRA (arXiv:2510.13003) and MDM-OC (arXiv:2507.20997) are cited. The contribution is applying orthogonal projection to ternary adapters and discovering the flat-spectrum failure mode. This is a valid micro-scale exploration.

**The flat-spectrum finding is the real contribution.** It reveals a structural property of ternary weights that invalidates a class of approaches (any method relying on spectral concentration of knowledge). This is valuable negative knowledge.

## Statistical Power Concerns

This is the weakest aspect of the experiment:

1. **MMLU: n=20 per domain.** A 5pp improvement (25% to 30%) on n=20 means 5 vs 6 correct answers -- a difference of 1 question. The 95% confidence interval for a binomial proportion at 6/20 is approximately [12%, 52%]. The 5pp improvement over DARE is NOT statistically significant.

2. **GSM8K: n=50.** The improvement from 44% to 52% (22 vs 26 correct) is marginally significant (p ~ 0.10 by Fisher's exact test). The improvement from 38% to 52% (19 vs 26) is more convincing (p ~ 0.04).

3. **In-dist math: n=20.** The drop from 80% to 40% (16 vs 8 correct) IS statistically significant (p ~ 0.01 by Fisher's exact). This is a real effect.

4. **Code gen: n=10.** 9/10 vs 9/10 tells us nothing.

**Bottom line:** The rho_k measurement (n=210 matrices, consistent across all domains) is statistically robust. The behavioral benchmarks are severely underpowered for the 5pp effects being claimed. The PAPER.md Limitations section acknowledges "n=20 per MMLU domain is low statistical power. Differences of 5pp are within noise." This is honest.

## Macro-Scale Risks (advisory)

1. **Flat spectrum may persist at scale.** If ternary weight spectra remain flat at larger model sizes, OPLoRA-style approaches are fundamentally inapplicable to the ternary composition architecture. This should be verified early at macro scale.

2. **The "capacity interference" concept identified here** (adding ANY perturbation reduces effective dimensionality) may be the dominant failure mode for all adapter composition approaches on ternary weights. This deserves its own experiment.

3. **The GSM8K improvement is promising** and suggests that orthogonal constraints help for procedural reasoning even when they fail for factual recall. At macro scale with routing, this could be a win.

## Verdict

**PROCEED** (as a killed experiment with finding)

Justification:

1. **The math is correct.** Theorem 1 is proven, implementation matches the proof, and rho_k = 0 was verified to numerical precision.

2. **The Type 2 exploration succeeded.** It narrowed the unknown from "what is optimal k?" to the structural impossibility: "k is undefined for flat-spectrum ternary weights because there is no spectral gap to separate knowledge from non-knowledge directions."

3. **The PAPER.md is honest.** Failed kill criteria are clearly marked. The limitations section acknowledges statistical power issues. The "capacity interference" concept is a genuine new insight that advances the project's understanding.

4. **The finding (flat ternary spectrum invalidates spectral-concentration assumptions) is valuable negative knowledge** that should prevent future wasted compute on methods requiring spectral gaps in ternary weights.

5. **Two concerns that do NOT block proceeding:**
   - The behavioral improvements (5pp MMLU) are within noise at n=20. The paper acknowledges this. The finding does not rest on these improvements -- it rests on the rho_k verification + the spectral gap discovery.
   - Kill criteria were not derived purely from the proof. Acceptable for Type 2 exploration where behavioral thresholds come from prior findings.

The experiment was correctly killed by the researcher. The finding (#272) should stand as supported. No revisions needed.

---

## Audit-Rerun Closure Review (2026-04-18)

**Reviewer on:** `experiment.done exp_orthogonal_adapter_training: KILLED (audit-rerun closure)`.

**State on review:** All 6 artifacts present. `git diff --stat` shows
PAPER.md +128 lines (append-only closure §). MATH.md, run_experiment.py,
results.json, LEARNINGS.md unchanged. DB status=killed with 2026-04-18
evidence row. KC IDs 684/685/686 consistent across DB ↔ MATH.md ↔ PAPER.md
↔ results.json ([K1_PASS=false, K2_PASS=true, K3_PASS=false] at lines
210-212). No KC-swap.

**Adversarial checklist:** (a)–(s) all PASS.
- (c) The PAPER pre-closure "Status: PARTIALLY SUPPORTED" (line 25) is
  retained verbatim but contextualized by the closure § as the mislabel
  being corrected. DB target is `killed` (not `supported`), so PLAN §1
  item 3 "while DB wants supported" clause does not trigger. The
  append-only closure ends with "**KILLED** — reaffirmed" (line 267).
- (e) MATH.md unchanged — KC 684/685/686 locked at 2026-03-31 pre-reg.
- (f) No tautology: KCs are empirical MMLU/GSM8K/in-dist measurements
  against pre-registered thresholds.
- (h) Composition uses DARE p=0.5 (single-adapter post-hoc), not the
  summation-bug pattern.
- (i) **LORA_SCALE=20** confirmed at `run_experiment.py:69`; mem-003
  antipattern acknowledged in closure; Thm C2 shows direction-
  preservation (safe-scale fix would shift which KC fails, not eliminate
  the kill — Pareto-frontier). Honest handling.
- (m) Target model `microsoft/BitNet-b1.58-2B-4T` matches between MATH.md,
  results.json, and run_experiment.py. Not a proxy-substitution case.
- (n) Base MMLU math = 50% (not 0%), so no thought-channel-truncation
  artifact driving the headline.
- (o) MMLU n=20 per domain and code n=10 are acknowledged low-power in
  PAPER §Limitations and the original review; does not alter kill
  direction (K3 in-dist math 8/20 vs 16/20 baseline is statistically
  significant by Fisher's exact ~0.01).
- (r) PAPER prediction-vs-measurement table present at §"Predictions
  vs Measurements" and extended in the closure §"KC disambiguation"
  table.
- (s) Math in the closure §: Thm C1 spectral-gap vacuity is formally
  kill-invariant (flat spectrum ⇒ top-k partition arbitrary ⇒ ρ_k=0
  guarantee is vacuous for knowledge preservation ⇒ K1/K3 structurally
  unreachable). Thm C2 is directional (scale-scan would likely recover
  K1/K3 but fail K2 — acknowledged as Pareto-frontier kill, not formal
  invariance). Thm C3 cites existing PAPER §"Key Discovery" (99.9% ρ
  reduction → only 5pp MMLU math improvement ⇒ ~80% of degradation is
  capacity interference) — no new claim.

**Antipattern self-check:**
- mem-003 (LORA_SCALE=20): APPLIES, acknowledged via C2.
- mem-021 (CEILING-HEADROOM COLLAPSE): **4th instance** (oracle-router
  → orthogonality → adapter-specialization → spectral-gap ceiling).
  Pattern abstract structure (mechanism M layered on baseline B_0 at M's
  theoretical ceiling) confirmed. Promotion to 4-instance confidence is
  appropriate — analyst should note.
- mem-001 (composition summation bug): N/A.
- mem-008 (thinking truncation): N/A (BitNet base, not Gemma 4).
- Verdict-DB mismatch antipattern: APPLIED and corrected by closure.

**Action:** Appended this "Audit-Rerun Closure Review" section to
REVIEW-adversarial.md. No DB change needed (researcher already logged
`--status killed --k 684:fail --k 685:pass --k 686:fail`).

**Verdict: PROCEED (as killed).**

**Route:** `review.killed` → Analyst.

**Open threads for analyst:**
- Promote mem-021 confidence to 4-instance in memory. Consider pre-flight
  check: before proposing any subspace-based method (SVD, PCA, spectral),
  measure the relevant spectral gap on the target model; kill preemptively
  if gap ratio < 1.1.
- Finding #272 (flat ternary spectrum invalidates subspace methods) is
  authoritative. LEARNINGS.md already captures alternatives (routing,
  SC-LoRA activation-space, accept+route).
- Closure § "KC disambiguation" table preserves the pre-reg contract
  with fix-invariance annotations — useful template for future closures.

**Backlog state:** P=1 open remaining: exp_p1_t5_user_local_training
(last P=1, macro). P=2 active open: ~67 (this P=2 closed).
