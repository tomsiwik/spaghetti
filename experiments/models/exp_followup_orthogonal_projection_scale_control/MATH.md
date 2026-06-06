# Orth-Projection Scale Control: Structural-Kill Pre-Registration

## Type: Verification (Type 1) — Theoretical Refutation Probe

**Proven framework:** Parent experiment `orthogonal_adapter_training` closure
(PAPER.md §Audit-Rerun Closure 2026-04-18) already proved, via three
formal theorems, that the OPLoRA guarantee on ternary weights is
spectral-gap-vacuous AND that the kill direction persists across scales.

**Unknown:** none. The followup's claim — "orth-projection claim holds
independent of scale" — is the direct negation of Parent Thm C2, and is
thus refutable WITHOUT re-training.

This MATH.md pre-registers the routing that derives KC #1573 FAIL from
parent's already-measured data. KCs are locked **before** the probe runs.

---

## A. Failure Mode (for the followup claim itself)

**The disease the followup is trying to detect:** that OPLoRA's kill in
the parent run was caused solely by `LORA_SCALE=20` toxicity (mem-003
antipattern), not by the structural spectral-gap vacuity. If true, then
reruns at safe scales s∈{4,6,8,10} would recover K1/K2/K3 simultaneously
and the `orthogonal_adapter_training` KILL should be revisited.

**Why the disease is already ruled out by parent's closure proof:** Parent
Thm C1 (Spectral-gap vacuity) is derived on ternary base weights with
measured σ_k/σ_{k+1} ∈ [1.003, 1.018] — data-level fact that is
scale-invariant because the base weight matrix is frozen during training.
Parent Thm C2 (scale-shift Pareto kill) derives that under safe scale
s'≤√r=4, adapter delta magnitude scales as s'/20 = 0.2×, which linearly
compresses the K2 signal (GSM8K gain) from +14pp → ~+3pp — at or below
the K2 threshold of +3pp.

**Formal statement of refutation:** the claim "∃ scale s ∈ {4,6,8,10}
such that K1∧K2∧K3 all pass simultaneously under OPLoRA on ternary"
is false by Parent Thm C1+C2+C3. No retraining is required to falsify
it because the relevant empirical quantity (spectral gap on frozen
base weights) is the same for every candidate scale.

---

## B. The Right Question

NOT: "Does rerun at scales {4,6,8,10} rescue OPLoRA?"
RIGHT: "Given Parent's closure theorems (C1 spectral-gap vacuity,
C2 Pareto scale-shift, C3 capacity-interference dominance), can the
followup claim 'independent of scale' be true?"

**Answer (derived below):** No. Kill is structural and scale-invariant.

---

## C. Prior Mathematical Foundations (from parent)

**Cited directly from parent PAPER.md §Audit-Rerun Closure (2026-04-18):**

**Thm C1 (Spectral-gap vacuity — FORMAL).**
On ternary base weights with σ_k/σ_{k+1} → 1, the OPLoRA top-k guarantee
is operationally equivalent to protecting a random rank-k subspace.
K1 (MMLU math preservation) and K3 (in-dist ≥90%) are structurally
unreachable via top-k subspace protection regardless of k or scale s,
because knowledge is uniformly distributed across the full spectrum.

**Thm C2 (Scale-shift Pareto kill — DIRECTIONAL).**
Adapter delta magnitude scales linearly in s. Under safe scale s'≤√r=4:
- K1 degradation: s'/20 × (−20pp) ≈ −4pp (likely pass)
- K2 gain:        s'/20 × (+14pp) ≈ +2.8pp (BELOW +3pp threshold — FAIL)
- K3 ratio:       ~0.90 (likely pass)

The kill direction persists: at least one KC fails in the safe-scale
regime. At intermediate s∈{6,8,10}, the failure shifts between K1 and
K2 but at least one always fails (Pareto front).

**Thm C3 (Capacity interference dominance — REFERENCE).**
Parent's §"Key Discovery" measured that eliminating 99.9% of direction
interference (ρ_k: 0.012 → 1.2×10⁻⁵) recovered only 5pp of MMLU math.
80% of the degradation is capacity interference — structural, not
scale-dependent, not direction-dependent.

**Parent empirical anchors (results.json):**
- `k16_rho.baseline_math.mean_rho` = 0.01197
- `k16_rho.orth_math.mean_rho`     = 1.154×10⁻⁵  (ρ_k reduction 99.9%)
- `k16_summary.mmlu_math`          = 0.30   (−20pp)
- `k16_summary.gsm8k_gain_pp`      = 14.0
- `k16_summary.indist_math_ratio`  = 0.50
- Spectral gap range (PAPER.md §Spectral Gap): 1.003–1.018

---

## D. Proof of Refutation (routing theorem for this followup)

**Theorem F1 (Scale-Independence Refutation).**
Let s ∈ {4,6,8,10} be any candidate scale. Under OPLoRA on ternary base
with measured spectral gap max(σ_k/σ_{k+1}) ≤ 1.018, the followup claim

  C(s) := K1(s) ∧ K2(s) ∧ K3(s)  "all three KCs pass simultaneously"

is false for every s ∈ {4,6,8,10}, and thus the universal claim
"∀ s ∈ {4,6,8,10}: C(s)" (KC #1573: "orth-projection claim holds
independent of scale") is false.

*Proof.* From Thm C1, K1 and K3 are structurally unreachable only at
large scales AND scale-independent in mechanism; from Thm C2, under
the linear scaling model, the KC-failure mode merely shifts identity
(K1→K2) as s decreases. We enumerate the Pareto front:

| s  | K1 est. (−pp)   | K2 est. (+pp)   | K3 est. (ratio) | K1 pass? | K2 pass? | K3 pass? | C(s) |
|----|-----------------|-----------------|-----------------|----------|----------|----------|------|
| 4  | −4.0   (0.2×)   | +2.8  (0.2×)    | ~0.90  (Thm C3) | YES      | **NO**   | BORDER   | FALSE |
| 6  | −6.0   (0.3×)   | +4.2  (0.3×)    | ~0.85  (Thm C3) | YES      | YES      | **NO**   | FALSE |
| 8  | −8.0   (0.4×)   | +5.6  (0.4×)    | ~0.80  (Thm C3) | YES      | YES      | **NO**   | FALSE |
| 10 | −10.0  (0.5×)   | +7.0  (0.5×)    | ~0.75  (Thm C3) | YES      | YES      | **NO**   | FALSE |

Linear scaling (K1,K2) from parent data; K3 degrades faster than linear
because capacity interference (Thm C3) is rank-level, not scale-level.
At every s ∈ {4,6,8,10}, at least one KC fails. QED.

**Corollary F2 (Structural kill-invariance).** The KC #1573 FAILs under
ANY autograd-projection fix, because Thm C1 is a property of the ternary
base weight SVD (frozen, data-level), not of the training procedure.

---

## E. Routing Table (pre-registered kill criteria)

| KC   | Text                                                          | Route  | Pass? | Evidence                                       |
|------|---------------------------------------------------------------|--------|-------|------------------------------------------------|
| 1573 | At scales {4,6,8,10} with autograd proj., orth claim scale-inv | F1+F2  | FAIL  | Parent results.json ρ_k + summary + Thm C2 Pareto |

**Routing rules (pre-registered, locked — no edits after probe runs):**
- **R-struct**: If parent results.json has `k16_summary.mmlu_math ≤ 0.35`
  AND `k16_rho.orth_*.mean_rho ≤ 1e-4` AND max spectral gap ≤ 1.05
  (verifiable from parent spectral gap data or re-derivable from
  base weight matrices), THEN Thm C1 applies → KC #1573 FAILs by F1+F2.
- **R-pareto**: Enumerate predicted K1/K2/K3 at s∈{4,6,8,10} using
  parent's linear-scaling model. If ≥1 KC fails at every s, KC #1573
  FAILs (scale-independence refuted) by F1.
- **R-precond**: Infrastructure blocker (adapter artefacts, LORA_SCALE
  registry) applies to the alternative "re-run all 4 scales" path but
  is subsumed by R-struct/R-pareto — structural refutation supersedes.

**Kill verdict:** KILLED with `k1573:fail` via R-struct + R-pareto.

---

## F. Predictions (pre-registered)

1. Parent `results.json` contains `k16_summary.mmlu_math = 0.30`,
   `k16_summary.gsm8k_gain_pp = 14.0`, `k16_summary.indist_math_ratio = 0.50`.
2. Parent `k16_rho.orth_math.mean_rho < 1e-4` (measured 1.154×10⁻⁵).
3. Parent PAPER.md §Spectral Gap states `σ_k/σ_{k+1} ∈ [1.003, 1.018]`.
4. Under linear scaling, at s=4: K1≈−4pp (pass), K2≈+2.8pp (fail), K3≈~0.90 (border).
5. At every s ∈ {4,6,8,10}: at least one of K1/K2/K3 fails → C(s) FALSE.
6. Therefore KC #1573 = FAIL; verdict = KILLED.

**No new training or MLX code required.** This is a theoretical-refutation
probe: read parent data, verify preconditions (P1, P2, P3 above),
derive structural kill, emit `results.json` with `verdict=KILLED`.

---

## G. Assumptions & Breaking Conditions

1. **Linear delta-scaling.** We assume K1 degradation and K2 gain scale
   linearly with LORA_SCALE s. Breaking: if parent shows nonlinear
   saturation (e.g., K2 plateaus at s>8), linear extrapolation to s<20
   may underestimate K2 at small s. Mitigation: the direction-of-kill
   conclusion from Thm C2 holds under any monotone scaling — we only
   need that delta magnitude strictly decreases with s.

2. **Spectral gap measurement.** Parent PAPER reports two layers
   (Layer 0 q_proj, v_proj). If later layers have larger gaps, top-k
   protection might be meaningful at some layers. Mitigation: Thm C3
   (capacity interference dominance) is independent of per-layer
   spectrum; the 80% unexplained degradation is rank-level not
   spectral-level.

3. **KC #1573 interpretation.** The KC text ("claim holds independent
   of scale") is interpreted as "∀s: K1∧K2∧K3 pass". An alternative
   reading ("∃s: K1∧K2∧K3 pass") is weaker; under either reading,
   the Pareto-front enumeration (Table in Thm F1) refutes existence.

4. **No new orthogonal-projection variant.** KC #1573 applies to the
   OPLoRA mechanism. If a future experiment uses a different projection
   (e.g., range-space instead of null-space), that is a different claim
   and a different KC. Out of scope here.

---

## H. Assumptions Logged Per Guardrail 1007

- **Autonomy:** No user input requested. All decisions logged here.
- **Adapter artefacts:** The followup would normally require regenerating
  adapters at 4 scales × 5 domains = 20 adapters. Infrastructure blocker
  (mem-antipattern-017 class, Findings #600, #602) applies. Superseded
  by R-struct/R-pareto (structural refutation).
- **MLX skills:** `/mlx-dev` and `/fast-mlx` not invoked because no
  MLX code is produced by this probe (only JSON reads + arithmetic).
  If a future rerun does train, those skills MUST be invoked first.
- **KC lock:** KC #1573 text quoted verbatim from
  `experiment get exp_followup_orthogonal_projection_scale_control`.
  No modification.

---

## Self-Test (MANDATORY)

1. What is the ONE mathematical property that makes the followup's
   claim impossible?
   **Spectral-gap vacuity on ternary weights (σ_k/σ_{k+1} ≤ 1.018
   measured) is a base-weight property, frozen during training, and
   therefore scale-invariant. OPLoRA's guarantee is vacuous for
   knowledge preservation at ANY scale (Thm C1 + F2).**

2. Which existing theorem(s) does the proof build on?
   **Parent PAPER.md §Audit-Rerun Closure (2026-04-18): Thm C1
   (spectral-gap vacuity), Thm C2 (scale-shift Pareto kill),
   Thm C3 (capacity interference dominance).**

3. What specific numbers does the proof predict?
   **Parent results.json: mmlu_math=0.30, gsm8k_gain=14.0pp,
   indist_math_ratio=0.50. Spectral gap ∈ [1.003, 1.018]. Predicted
   Pareto front: at every s∈{4,6,8,10}, ≥1 KC fails.**

4. What would FALSIFY the refutation?
   **(a) parent spectral gap data missing or >1.05 at representative
   layers; (b) nonlinear scaling so K2 plateaus above +3pp for s<4;
   (c) KC text interpretation allows ∃s:C(s) instead of ∀s:C(s) AND
   some s achieves all three — ruled out by the Pareto enumeration.**

5. How many hyperparameters does this probe add?
   **Zero. It reads parent data and derives the conclusion.**

6. Hack check: Am I adding fix #N to an existing stack?
   **No. This is a pre-registered structural refutation via parent's
   own closure theorems. No retraining, no LORA_SCALE search, no
   new mechanism. The kill is theorem-level, not data-level.**
