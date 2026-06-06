# MATH.md — exp_hedgehog_composition_polite_refactor_js (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. This is the 4th+ reuse of the F#669 structural pattern in the 2026-04-23 drain window (see also F#671, F#672, F#687). Rationale derived below; no MLX `run_experiment.py` implementation is attempted because all three parent dependencies are in states that make every KC structurally untestable.

The original design (composition theorem for Hedgehog adapters on Gemma 4 E4B) is preserved in §6 for re-claim after parents land target-SUPPORTED.

---

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). Not invoked — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627).
- Adapter targets: `v_proj + o_proj` (per F#627, rank-8 per MATH.md §6.2).
- mlx-lm version: any ≥0.22 suffices for design-only artifact; not loaded.
- Parents:
  - `exp_hedgehog_behavior_adapter_politeness` (status `provisional`, F#683) — no trained ΔW_polite.
  - `exp_hedgehog_procedural_adapter_refactor` (status `provisional`, F#684) — no trained ΔW_refactor.
  - `exp_hedgehog_domain_adapter_js` (status `open`, never run) — no trained ΔW_js.

## §1 Preempt-KILL theorem

**Theorem (inter-experiment triple-parent target unverifiability).** Let `C` denote child experiment `exp_hedgehog_composition_polite_refactor_js` with kill criteria K = {K1 structural proxy, K2 target triple-axis judge, K3 target ablate-polite, K4 target ablate-refactor, K5 target non-interference}. Let `P1 = exp_hedgehog_behavior_adapter_politeness`, `P2 = exp_hedgehog_procedural_adapter_refactor`, `P3 = exp_hedgehog_domain_adapter_js`. Every KC in K requires runtime composition `W_comp = W_base + ΔW_polite + ΔW_refactor + ΔW_js` where each `ΔW_i` is a *target-SUPPORTED trained Hedgehog LoRA adapter* — i.e. K1/K2 for its own parent must have passed (per F#666 target-gated kill).

If `P1.status ∈ {provisional, open}` or `P2.status ∈ {provisional, open}` or `P3.status ∈ {provisional, open}` — i.e. at least one `ΔW_i` is missing, untrained, or Hedgehog-target-unverified — then:
- **K1** "composed per-layer cos vs ideal teacher" is undefined. The cosine is computed between the composed-attention trajectory and a 3-prompt-concatenated teacher; both operands require all three adapters to be loadable tensors with teacher-validated structure. Missing any one → undefined operand.
- **K2** "politeness Δ ≥+15pp AND refactor ≥ baseline-lora AND JS-idiomatic ≥ base+JS-LoRA, simultaneously on n≥100" requires each axis to have a target-validated contributor. Politeness (P1 target K2) is untested per F#683. Refactor (P2 target K2) is untested per F#684. JS-idiomatic (P3 target K2) has never been measured. A triple conjunction over three untested axes is not a measurement; it is a sample from the joint distribution of initialization noise. Vacuous.
- **K3 (ablate-polite)** sets `α_polite = 0` and re-measures. Setting zero on a zero-trained adapter is a no-op — the "ablation" changes nothing measurable. Claim "diagonal dominance" is undefined because there is no diagonal.
- **K4 (ablate-refactor)** identical structural defect to K3 for P2.
- **K5 (non-interference on polite-only prompts)** requires the full composition to be populated with target-SUPPORTED adapters; with 3/3 parents unverified, "the composition" is not a defined object.

∴ ∀ k ∈ K: testing `k` while `{P1, P2, P3}.status ∌ {supported, proven}` jointly produces vacuous PASS (init-artifact co-occurrence) or vacuous FAIL (uninformative), i.e. an unidentifiable sample under F#669. **QED.**

**Sharper than F#669:** the triple-parent case strictly subsumes single-parent because failure is *conjunctive* — the child is unidentifiable even if only 1 of 3 parents is unverified. Here all three are. The preempt is stable under any ordering of parent training.

## §2 Prior art

- **F#669** (2026-04-19) established preempt-KILL on target-unverified parent. Promoted to canonical reviewer.md §5 clause after F#687 (3rd+ reuse).
- **F#671, F#672, F#687** — prior applications of the same pattern (RDT ACT halting, loop-adapter-composition children, JEPA-router-prediction-error).
- **F#683, F#684** — PROVISIONAL status of the two already-attempted parents. Both design-only (graceful-failure run_experiment.py, no MLX training loop executed). Both have `_impl` companions filed at P3.
- **F#666** — target-gated kill rule. Per reviewer.md §5 canonical clause, F#666 does NOT gate preempt-KILL (a preempt is not a claim about the underlying mechanism; it is a claim about the current dependency state).
- **F#627** — N=24 SFT-LoRA composition on Gemma 4 E4B supported. Establishes runtime-compose as viable *when* adapters are target-SUPPORTED. Does not apply here: these are Hedgehog (per-layer cos-sim distilled) adapters, not SFT.
- **F#571** — pre-merge composition killed 4×. Reason to avoid merging at all; runtime-compose only.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked, triple-parent)"**:

| KC  | Claim                                                                     | Measurement status                  |
| --- | ------------------------------------------------------------------------- | ----------------------------------- |
| K1  | composed per-layer cos vs ideal-teacher > 0.70 on JS refactor prompts     | untested (preempt-blocked)          |
| K2  | politeness +15pp ∧ refactor ≥ LoRA ∧ JS ≥ base+JS-LoRA, n≥100             | untested (preempt-blocked)          |
| K3  | ablate-polite: politeness -15pp, refactor/JS drop <3pp                    | untested (preempt-blocked)          |
| K4  | ablate-refactor: refactor -10pp, polite/JS drop <3pp                      | untested (preempt-blocked)          |
| K5  | non-JS polite prompts: composed politeness within 2pp of polite-alone     | untested (preempt-blocked)          |

## §4 Unblock condition

Re-claimable when **all three** parents reach `status ∈ {supported, proven}` with target KCs verified at full scale:

1. **P1** (`exp_hedgehog_behavior_adapter_politeness`) — K2 auto-judge politeness Δ must be SUPPORTED at n≥100 (not the current design-only PROVISIONAL).
2. **P2** (`exp_hedgehog_procedural_adapter_refactor`) — K2/K3 Fowler-judge refactor quality must be SUPPORTED at n≥100 on held-out prompts.
3. **P3** (`exp_hedgehog_domain_adapter_js`) — K2 JS-specific benchmark accuracy must be SUPPORTED (HumanEval-JS or analogous).

At that point, all three ΔW_i exist as target-validated trained adapters, and the child KCs in §3 become measurable. The original composition design (§6) becomes executable.

**Alternative unblock:** re-design child to not require triple-parent target-SUPPORTED (e.g. two-adapter pair composition as a less-headline test of the thesis). Currently out of scope.

## §5 Follow-up

No `_impl` companion is filed for this child because the kill is preempt-structural, not design-incomplete. Unblock routes via:
- `exp_hedgehog_behavior_adapter_politeness_impl` (P3, already filed)
- `exp_hedgehog_procedural_adapter_refactor_impl` (P3, already filed)
- `exp_hedgehog_domain_adapter_js` itself (P3, still open) — this is the cleanest standalone entry.

Per F#669 precedent: preempt-structural KILL does not spawn `_impl`. Unblock is parent-external.

---

## §6 Original design (preserved for re-claim)

Below is the original MATH content as written 2026-04-22. It remains valid as the design against which child KCs would be measured when the unblock condition §4 is met. Nothing in §6 is being tested in this iteration — it is documentation for the future re-claim.

### §6.1 Failure mode (original)

Primary degenerate behavior: "Composition at N=3 exceeds the interference floor — one or more adapters' attention-routing perturbations overwhelm the others, producing either (a) politeness collapse (refactor wins, style becomes neutral), (b) refactor collapse (JS adapter dominates, producing idiomatic but non-refactored code), or (c) base-task collapse (combined perturbation exceeds the residual-stream budget, MMLU drops > 5pp)."

Secondary: "Cross-leakage — removing polite adapter causes refactor AND JS quality to drop ≥ 3pp (non-zero off-diagonal in the ablation matrix), indicating capability entanglement at the attention-routing level."

### §6.2 Theorem (original, informal)

Let `ΔW_p, ΔW_r, ΔW_j` be the three trained LoRA deltas on `(v_proj, o_proj)` of Gemma 4 E4B. Let `W_comp = W_base + ΔW_p + ΔW_r + ΔW_j`.

**Theorem.** For prompts in `D_JS-refactor`, running inference with `W_comp` produces outputs with:

1. **Politeness score** (auto-judge) ≥ base + ΔJudge_p − 5pp
2. **Refactor quality** ≥ standalone refactor adapter − 5pp
3. **JS-idiomaticity** ≥ standalone JS adapter − 5pp
4. **Ablation diagonal dominance:** removing `ΔW_i` reduces axis-`i` quality by ≥ 10pp while off-diagonal drops < 3pp.
5. **Non-target tasks** (MMLU, non-code prompts) drop < 3pp vs base.

**Proof sketch.**
1. *Additivity.* Attention-routing perturbations from independently-trained adapters on orthogonal-initialized subspaces (F#562) sum approximately linearly when magnitudes are bounded (LoRA r=8 each).
2. *Capability retention (1-3).* If standalone K1 cos > 0.85 for each adapter, the composed K1 cos against the ideal combined teacher degrades gracefully by triangle inequality.
3. *Diagonal dominance (4).* Each adapter has no gradient signal to encode other axes.
4. *Non-target bound (5).* Sum of three rank-8 perturbations is rank-24 (or less); F#627 supports N=24 with MMLU drop < 2pp.

QED sketch. Weakest link: step 1 (orthogonal subspaces) — empirical verification required.

### §6.3 Original kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | composed per-layer cos vs ideal-teacher (3-prompt concat) on JS refactor prompts | > 0.70 | structural proxy |
| K2 | politeness Δ ≥ +15pp AND refactor-judge ≥ baseline-lora AND JS-idiomatic ≥ base+js-LoRA | all 3 simultaneously | target multi-axis |
| K3 | ablate-polite: politeness drop ≥ 15pp, refactor/JS drop < 3pp | both conditions | target ablation (F#666 pair K2) |
| K4 | ablate-refactor: refactor-judge drop ≥ 10pp, polite/JS drop < 3pp | both conditions | target ablation |
| K5 | on non-JS non-refactor polite prompts, composition matches polite-alone within 2pp | Δ ≤ 2pp | target non-interference |

### §6.4 Predicted measurements (original)

- K1: composed cos ∈ [0.70, 0.82]
- K2: politeness +22pp, refactor-judge roughly-matched, JS-idiomatic +3pp
- K3: ablate-polite → politeness collapses to +3pp, refactor -1pp, JS -1pp (clean subspaces)
- K4: ablate-refactor → refactor-judge -12pp, politeness -1pp, JS -2pp
- K5: polite prompts (non-code) Δ ≤ 1.5pp
