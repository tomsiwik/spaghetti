# MATH.md — exp_user_adapter_from_memento_distillation (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code is run. This is the 5th+ reuse of the F#669 structural pattern in the 2026-04-23 drain window (see also F#671, F#672, F#687, F#688). Rationale derived below; no MLX `run_experiment.py` implementation is attempted because both parent dependencies are in states that make every KC structurally untestable.

The original design (Hedgehog-style per-layer cos-sim distillation of a 50-session memento buffer into a rank-6 user-adapter) is preserved in §6 for re-claim after parents land target-SUPPORTED.

---

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). Cited, not invoked — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627).
- Adapter targets: `v_proj + o_proj` (per F#627), rank-6 for user-adapter (per §6.2).
- mlx-lm version: any ≥0.22 suffices for design-only artifact; not loaded.
- Parents:
  - `exp_memento_gemma4_replication` (status `provisional`, F#685) — no trained MEMENTO SFT checkpoint; custom block-mask inference loop + 2-stage SFT never executed.
  - `exp_memento_cross_session_persistence` (status `open`, never run, P3) — no user memento-buffer rehydration stack validated; no 50-session buffer format certified against target task accuracy / compaction bounds.

Scope-preservation lock (F#669 discipline): this preempt does NOT silently swap mechanism. The original design calls for Hedgehog per-layer cos-sim distillation over memento-rehydrated teacher traces. Under preempt, no substitution is attempted (e.g. "skip the memento teacher, train user-adapter on raw SFT pairs" would be a scope swap per antipattern-t and antipattern-novel-mechanism-single-iteration-scope — forbidden).

## §1 Preempt-KILL theorem (dual-parent disjunctive)

**Theorem (inter-experiment dual-parent target unverifiability).** Let `C` denote child experiment `exp_user_adapter_from_memento_distillation` with kill criteria K = {K1 training-cost proxy, K2 target user-style match, K3 target 4-way composition, K4 target non-interference, K5 target structural privacy}. Let `P_R = exp_memento_gemma4_replication` (MEMENTO replication) and `P_X = exp_memento_cross_session_persistence` (cross-session persistence). Every KC in K requires, transitively, two target-validated parent artifacts:

1. A **memento-rehydrated teacher** `f_T(x) = forward(θ_base + Δθ_MEMENTO, prepend(B_user, x))` — this is the `P_R` output. Without `P_R` target-SUPPORTED, `Δθ_MEMENTO` does not exist (no 2-stage SFT checkpoint + no block-mask attention inference loop validated), and the teacher signal is undefined.
2. A **50-session user memento buffer** `B_user = {M_1, ..., M_50}` with validated cross-session serialization + compaction — this is the `P_X` target (K2 accuracy ≥ 90% of full-context handoff; K3 sub-linear compaction; K4 pickle round-trip). Without `P_X` target-SUPPORTED, `B_user` is an uncertified serialization object that may not actually round-trip or preserve user-state — running training on it produces weights trained against an uncertified buffer.

If `P_R.status ∈ {provisional, open}` **or** `P_X.status ∈ {provisional, open}` — i.e. at least one of `{Δθ_MEMENTO, B_user}` is missing or target-unverified — then:

- **K1** "training wall-clock < 30 min" is vacuously measurable but meaningless: it measures *training cost of a distillation against an undefined teacher signal*. PASS is a PASS-on-init-artifacts; FAIL is a FAIL-on-pipeline-plumbing. Unidentifiable.
- **K2** "|student_judge − teacher_judge| ≤ 5pp on held-out user prompts" is undefined because `teacher_judge` requires the memento-rehydrated teacher (P_R). Without Δθ_MEMENTO the "teacher" is just the base model with prepended tokens; this is a different teacher than the one the original design specifies, and the KC becomes a proxy-without-ground-truth.
- **K3** "4-way composition preserves ≤ 3pp drop on each of {user, polite, refactor, JS}" requires the user-adapter to be meaningfully trained (K2 SUPPORTED) *and* requires the 3 companion adapters `{ΔW_polite, ΔW_refactor, ΔW_js}` from `exp_hedgehog_composition_polite_refactor_js` which is itself KILLED (preempt, F#688 — 3/3 Hedgehog parents target-unverified). Missing composition components ⇒ undefined.
- **K4** "MMLU/HumanEval drop < 2pp with user-adapter attached" — the "user-adapter" attached is one trained against an unidentifiable teacher; drop is measurable but not a claim about "personalization non-interference" — only about "arbitrary rank-6 perturbation on (v_proj, o_proj)". Scope-scrambled.
- **K5** "white-box reconstruction L2 error of any memento from weights > threshold" requires `B_user` to be a *real certified* user buffer — not a random placeholder. Without `P_X` SUPPORTED, the "memento" that reconstruction is attempted against is not a valid target (no ground truth of "what was in the buffer"). Reconstruction error is undefined.

∴ ∀ k ∈ K: testing `k` while `{P_R, P_X}.status ∌ {supported, proven}` jointly produces vacuous PASS (init-artifact co-occurrence) or vacuous FAIL (uninformative pipeline noise), i.e. an unidentifiable sample under F#669 / reviewer.md §5 preempt-structural canonical clause. **QED.**

**Sharpness positioning.** This is a **dual-parent disjunctive** sub-case — strictly sharper than single-parent (F#687) because failure is *disjunctive* over parents (either one missing ⇒ child unidentifiable). Weaker than triple-parent (F#688) only in parent count, not in structural stability: the preempt survives any ordering of parent completion. If `P_R` lands SUPPORTED first, child still blocks on `P_X`; if `P_X` lands first, child still blocks on `P_R`.

## §2 Prior art

- **F#669** (2026-04-19) established preempt-KILL on target-unverified parent. Promoted to canonical reviewer.md §5 clause after F#687 (3rd+ reuse); now at 5th+ reuse.
- **F#671, F#672, F#687, F#688** — prior applications of the same pattern (RDT ACT halting, loop-adapter children, JEPA-router-prediction-error, hedgehog-composition-polite-refactor-js). F#688 is the closest analog — triple-parent composition-dependent child, 0/3 parents target-verified.
- **F#685** — `P_R` (MEMENTO replication) PROVISIONAL. Novel-mechanism design-only per reviewer.md §5 PROVISIONAL-as-design clause.
- **F#683, F#684** — relevant because the sibling composition child `exp_hedgehog_composition_polite_refactor_js` (KILLED F#688) required Hedgehog adapters that our K3 also requires. K3 specifically inherits that failure mode.
- **F#666** — target-gated kill rule. Per reviewer.md §5 canonical clause, F#666 does NOT gate preempt-KILL (a preempt is not a claim about the mechanism; it is a claim about the current dependency state).
- **F#627** — N=24 SFT-LoRA composition on Gemma 4 E4B SUPPORTED. Establishes runtime-compose viability *when* adapters are target-SUPPORTED. Does not apply here: user-adapter would be Hedgehog-distilled (per-layer cos-sim), not SFT.
- **F#571** — pre-merge composition killed 4×. Reminder to use runtime-compose only at N>1.
- **F#562** — structural orthogonality at native dims supports N-way composition if adapters are initialized on orthogonal Grassmannian subspaces.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked, dual-parent)"**:

| KC  | Claim                                                                               | Measurement status                           |
| --- | ----------------------------------------------------------------------------------- | -------------------------------------------- |
| K1  | wall-clock training < 30 min on M5 Pro                                              | untested (preempt-blocked, P_R + P_X needed) |
| K2  | \|student_judge − teacher_judge\| ≤ 5pp on held-out user-typical prompts            | untested (preempt-blocked, P_R needed)       |
| K3  | 4-way composition {user, polite, refactor, JS} drop < 3pp on each axis              | untested (preempt-blocked, composition via F#688-KILLED sibling) |
| K4  | MMLU drop < 2pp ∧ HumanEval drop < 2pp with user-adapter attached                   | untested (preempt-blocked, P_R needed)       |
| K5  | white-box L2 reconstruction error of any memento > fixed threshold                  | untested (preempt-blocked, P_X needed)       |

## §4 Unblock condition

Re-claimable when **both** parents reach `status ∈ {supported, proven}` with target KCs verified at full scale:

1. **P_R** (`exp_memento_gemma4_replication`) — currently PROVISIONAL per F#685. Its `_impl` companion `exp_memento_gemma4_replication_impl` is filed at P3 inheriting KCs #1829-#1832. P_R target-SUPPORTED requires full 2-stage SFT run + block-mask inference loop + K2 (GSM8K drop < 5pp ∧ MMLU drop < 3pp) + K3 (KV-channel ablation ≥ 10pp target replication).
2. **P_X** (`exp_memento_cross_session_persistence`) — currently OPEN at P3, never run. P_X target-SUPPORTED requires K1 (rehydration latency < 50ms), K2 (multi-turn accuracy ≥ 90% of full-context handoff on 30-turn user simulator), K3 (sub-linear compaction bounded at < 2k tokens), K4 (pickle round-trip within 2pp in-memory accuracy).
3. **Sibling composition dependency (K3 specifically)** — the 4-way composition KC additionally requires `{ΔW_polite, ΔW_refactor, ΔW_js}` to exist. Those come from `exp_hedgehog_composition_polite_refactor_js` (KILLED preempt, F#688) or equivalently from the 3 individual Hedgehog parents (P1/P2/P3 of F#688). Partial unblock is possible if user-adapter is re-scoped to drop K3 composition (e.g. test only user-adapter alone + user+polite = pair composition), but that is a redesign.

At full unblock, both parent artifacts exist as target-validated pieces, and the original design (§6) becomes executable.

**Alternative unblock (out of scope now):** redesign child with fewer parent dependencies — e.g. synthetic-user-buffer (simulate 50 sessions without needing the full P_X cross-session stack) + pair composition (drop K3 4-way, test only user+polite N=2). Would require new experiment id; not pursued this iteration to preserve drain-progression focus (Option A per analyst handoff).

## §5 Follow-up

No `_impl` companion is filed for this child because the kill is preempt-structural, not design-incomplete. Unblock routes via two parent-external paths:

- `exp_memento_gemma4_replication_impl` (P3, already filed) — unblocks `P_R`.
- `exp_memento_cross_session_persistence` itself (P3, open) — can be claimed directly when queue reorders; does not need an `_impl` companion because its own status is `open`, not `provisional`.

Per F#669 precedent and reviewer.md §5 canonical clause: preempt-structural KILL does not spawn `_impl`. Unblock is parent-external.

---

## §6 Original design (preserved for re-claim)

Below is the original MATH content as written 2026-04-22. It remains valid as the design against which child KCs would be measured when the unblock condition §4 is met. Nothing in §6 is being tested in this iteration — it is documentation for the future re-claim.

### §6.1 Failure mode (original)

Primary: "The memento buffer is too topically diverse (50 sessions span many domains). The student tries to average across inconsistent user-styles, producing a bland adapter with no measurable personalization lift (K2 fail). This is the user-generalization problem."

Secondary: "User-adapter collides with existing behavior/domain adapters at the attention-routing level — composition at N=4 exceeds the interference floor (F#571 reminder that pre-merge kills at N>1; we rely on runtime compose F#627 tested at N=24). K3 composition drops > 3pp on any axis."

Tertiary: "White-box reconstruction attempt succeeds — privacy claim K5 fails. Adapter weights leak memento content at unacceptable rate."

### §6.2 Theorem (original, informal)

Let `B_user = {M_1, ..., M_50}` be the user's accumulated memento buffer (50 sessions), `θ_base` the base Gemma 4 E4B weights, `Δθ_user` the rank-6 LoRA on (v_proj, o_proj) we train.

**Teacher:** `f_T(x) = forward(θ_base + Δθ_MEMENTO, prepend(B_user, x))` — memento-rehydrated base.
**Student:** `f_S(x) = forward(θ_base + Δθ_user, x)` — memento-free + user-adapter.
**Loss:** `L = E_x ∈ D_user [Σ_l (1 - cos(A_l(f_T, x), A_l(f_S, x)))]` where `D_user` = sample of user-typical prompts.

**Theorem.**
1. **Training cost:** training converges in ≤ 30 min on M5 Pro (K1)
2. **User-style match:** on held-out user-typical prompts, student matches teacher within 5pp auto-judge (K2, pair K1)
3. **Composition preservation:** composing `Δθ_user + Δθ_polite + Δθ_refactor + Δθ_js` preserves all 4 axes within 3pp of isolated each (K3)
4. **Non-interference:** with user-adapter attached, MMLU drop < 2pp, HumanEval drop < 2pp (K4)
5. **Privacy-by-construction:** for any memento `m ∈ B_user`, white-box reconstruction from `Δθ_user` alone yields L2 error > fixed threshold (K5 — structural)

**Proof sketch.**
1. *Training cost.* Rank-6 LoRA + 50-memento teacher traces (≈ 5k training examples after data-augmentation-via-prompt-sampling) at batch=1 seqlen=2048 ≈ 3000 gradient steps × 0.5s/step on M5 Pro ≈ 25 min.
2. *Matching.* Hedgehog recipe achieves > 0.85 per-layer cos with rank-8 (F#683/F#684 prediction). At rank-6 with a narrower teacher signal (one user's buffer) the target is similar or easier.
3. *Composition.* Rank-6 + rank-8 + rank-8 + rank-8 = 30 effective rank. F#627 supports N=24 × rank-6 = rank-144; our case is within safe region. Structural orthogonality (F#562) ensures non-overlapping subspaces if each adapter is initialized with Grassmannian A-init.
4. *Non-interference.* Sum of 4 rank-≤8 perturbations on `(v_proj, o_proj)` is rank-30 max. F#627 tested larger without MMLU collapse.
5. *Privacy.* Training compresses ≈100k tokens into ≈60k params (4-bit quantized → 120kb). Information ratio ≈ 1.6:1 — lossy. Exact reconstruction is impossible at this compression ratio (information-theoretic bound).

**Weakest link:** (2) — if user's memento buffer is topically scattered, cos-sim loss averages across incompatible routing patterns. Mitigation: relevance-weighted training (sample mementos near the query) or per-topic sub-adapters. Tracked as future work if K2 fails.

### §6.3 Original kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | wall-clock training from 50-session memento buffer on M5 Pro | < 30 min | proxy (training cost) |
| K2 | user-style held-out auto-judge: \|student_score − teacher_score\| | ≤ 5pp | target (pair K1) |
| K3 | composed 4-way: axis-i quality drop vs isolated | < 3pp on each of {user, polite, refactor, JS} | target composition |
| K4 | MMLU drop with user-adapter attached; HumanEval drop | both < 2pp | target non-interference |
| K5 | white-box reconstruction L2 error of any memento from weights | > fixed threshold | target structural privacy |

### §6.4 Predicted measurements (original)

- K1: training wall-clock ≈ 22-28 min on M5 Pro
- K2: |Δjudge| ≈ 2-4 pp (student matches teacher closely)
- K3: axis drops ≤ 2pp on each
- K4: MMLU drop ≤ 1pp, HumanEval drop ≤ 1pp
- K5: reconstruction L2 error ≫ threshold (information-theoretic argument)

Risk: if K2 fails (|Δ| > 5pp), user-adapter is under-fit. Consider (a) rank increase to r=8, or (b) per-topic sub-adapters via memento clustering.
