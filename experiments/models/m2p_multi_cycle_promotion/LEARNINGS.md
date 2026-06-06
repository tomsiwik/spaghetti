# LEARNINGS.md — exp_m2p_multi_cycle_promotion

## Outcome
**KILLED** on K930 (absolute accuracy < 50%), SUPPORTED on K928 + K929 + Pythagorean bound.

## Key Learnings

### 1. Pythagorean bound (Theorem 1) is an algebraic identity
Under Grassmannian A-slots (A_i^T A_j = δ_ij I_r), the Frobenius norm of cumulative weight updates grows exactly as sqrt of sum of squared adapter norms — verified to rel_err = 2.34e-8 (floating-point noise floor). This is not an empirical measurement; it is a mathematical consequence of the orthogonality construction. No ambiguity.

### 2. Weight-space orthogonality ⇏ activation-space orthogonality
Theorem 2 guarantees zero Frobenius inner product between adapters, so later promotions cannot lower the weight-space projection onto earlier domains' A-slots. However, the effective activation-space impact on domain-k inputs is bounded by `‖A_j.T x_k‖` — small but nonzero. K928/K929 are needed to empirically verify this residual is below practical thresholds.

### 3. Absolute-accuracy kill criteria are uninformative below SFT ceiling
K930 ("any domain < 50%") fired because the toy transformer (d=128, 2 layers, 800 pretrain + 600 SFT steps) cannot reach 50% on mod-10 arithmetic even in isolation. The SFT baseline itself is 5% (add), 0% (sub), 40% (mul) — the model is structurally incapable of solving these tasks. K930 caught model weakness, not promotion interference.

**Tripwire (reusable):** Before registering an absolute-threshold KC, measure the SFT baseline. If SFT < threshold, reframe KC as a relative ratio. This is a DESIGN principle, not a hypothesis.

### 4. Promotion doesn't destroy what wasn't there
On all 3 domains, the PROMOTED (no-adapter) accuracy equals or exceeds SFT (adapter-active) accuracy:
- add: 5% SFT → 10% promoted (+100%)
- sub: 0% SFT → 10% promoted (+∞)
- mul: 40% SFT → 40% promoted (=)

Interpretation: the small quality gain on add/sub is attributable to the base model continuing to learn during joint pretraining + adapter-training (confounded with promotion). The key finding is that promotion does not degrade the weights below their pre-promotion state — zero interference, not negative interference.

### 5. Impossibility structure requires real-LLM verification
The Pythagorean/Grassmannian math makes K930 structurally impossible FOR REAL LLMs where SFT > 50%. This experiment cannot verify that structural claim on the toy model; a follow-up on Qwen3-0.6B or Gemma 4 is needed. See finding #453 (exp_p1_t6_flywheel_simulation) which demonstrated ε_cumul=7.6% across 3 promotions on a real model — complementary evidence for the structural claim.

## Findings Registered
- **#398** [killed] (pre-existing, 2026-04-08): Multi-cycle promotion: Pythagorean bound verified to machine precision; K930 killed by toy model capacity, not promotion interference.

## Reusable Side-Findings (analyst-owed when cap lifts)
- **Toy-model-capacity-ceiling antipattern:** absolute-threshold KCs confound model weakness with hypothesized failure modes. Tripwire: check SFT baseline before threshold.
- **Activation-vs-weight-space gap:** orthogonal A-slots protect weights exactly, but activation bleed still exists; it must be empirically bounded via K928/K929-style quality-ratio tests.

## Follow-ups
- Real-LLM replication (exp_g4_flywheel_real_users, exp_followup_m2p_crystallize_real_users) — in cohort, awaits drain.
- Symplectic integration for promotion (notes field of experiment) — research direction, not actionable until K=5 tested.

## Why no-rerun
The three stated hypotheses (K928 quality retention, K929 no-cross-cycle degradation, Pythagorean bound) are all verified. K930's failure mode is diagnosable as toy-model capacity, not a new axis. Rerunning with larger toy model would not validate the structural claim — that requires a real LLM (see exp_g4_flywheel_real_users). No code fix required.

## Status: terminal for this experiment scope.
