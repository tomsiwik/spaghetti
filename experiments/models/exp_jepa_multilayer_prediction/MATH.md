# MATH.md — exp_jepa_multilayer_prediction (PREEMPT-KILL)

## Verdict: PREEMPT-KILL

This experiment is preempt-killed per **Finding #669** (preempt-child-KCs-require-parent-target-claim-unverified) before any code was run. Rationale derived below; no `run_experiment.py` MLX implementation is attempted because the parent dependency is in a state that makes every KC structurally untestable.

This is the **5th reuse** of the F#669 pattern (F#669 → F#687 → F#698 → F#699 → this). Promotion threshold confirmed at F#698 (3rd reuse); 5th reuse re-confirms it as canonical routing. Same parent as F#687 and F#698 (`exp_jepa_adapter_residual_stream`) — this is the 3rd child of that PROVISIONAL parent to hit preempt-KILL.

## §0 Platform / skills / model pins

Included for completeness even though no platform code is executed.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist item (m2).
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: `v_proj + o_proj` (per F#627) — standard JEPA-adapter injection surface from parent F#682.
- Parent dependency: `exp_jepa_adapter_residual_stream` (status `provisional`, F#682).

## §1 Preempt-KILL theorem

**Theorem (inter-experiment target unverifiability, skip-layer-prediction variant).** Let `C` denote child experiment `exp_jepa_multilayer_prediction` with kill criteria K = {K1885 (proxy: L+2 prediction MSE > 2× L+1 prediction MSE), K1886 (target: L+2 adapter behavioral quality > 3pp worse than L+1)}. Let `P` denote parent experiment `exp_jepa_adapter_residual_stream`.

Both KCs reference an **L+1 baseline** as the right-hand side of the kill inequality. The L+1 prediction MSE and the L+1 behavioral quality are precisely the two quantities parent `P` is designed to target-validate (K1767 proxy: L_pred ratio < 0.5; K1768 target: GSM8K-Hard accuracy ≥ LoRA r=16 baseline; K1769 target: lambda=0 ablation ≥5pp drop) but has not — `P.status=provisional`, all 4 KCs untested per F#682.

If `P.status ∈ {provisional, open}` — i.e. no target-verified L+1 JEPA-adapter exists — then:

- **K1885** ("L+2 MSE > 2× L+1 MSE"): the right-hand side `L+1 MSE` is unverified. Comparing MSE ratio against an unverified baseline produces either (a) vacuous PASS if the L+2 adapter trains while parent L+1 is untrained (L+2 "wins" trivially against a zero/uninitialized reference), (b) vacuous FAIL if both collapse (both MSEs are noise-dominated; ratio is floating-point artifact), or (c) meaningful ordering only if L+1 MSE is itself target-validated as "learning real dynamics" — parent's K1767 precondition that has not been measured. Skip-gram-style longer-range prediction is only meaningfully harder than CBOW-style near-range prediction *if near-range prediction itself works*; without that anchor, "2× harder" has no referent.

- **K1886** ("L+2 behavioral quality > 3pp worse than L+1"): the right-hand side `L+1 behavioral quality` is parent's K1768 claim (GSM8K-Hard ≥ LoRA r=16). If parent K1768 is untested, there is no L+1 behavioral-quality anchor for the 3pp gap to be measured against. Substituting a different baseline (e.g. token-space r=16 LoRA directly, bypassing L+1 JEPA) would be antipattern-t (silent objective swap); the KC explicitly says "worse than L+1", not "worse than any LoRA baseline."

∴ ∀ k ∈ K: testing `k` while `P.status ≠ supported|proven` produces an unidentifiable sample. **QED.**

### §1.1 No F#666 compound block

The KC set IS properly target-gated per F#666: K1885 is a proxy (MSE ratio is a structural prediction-quality metric), K1886 is a target (behavioral quality gap on task performance). This matches the F#699 / F#698-memento pattern (target-gated at construction) — NOT the F#698-attention_output pattern where both KCs were proxy-only and triggered a compound F#666 block.

KILL would require BOTH to fail; SUPPORTED requires BOTH to pass. This is F#666-compliant out of the box. No KC-augmentation is needed at re-claim time — the only blocker is parent target-verification.

**F#666-compound-subcase: false.** Single preempt-block on parent target-unverification only.

## §2 Prior art

- **F#669** (2026-04-19) established the preempt pattern for `exp_rdt_act_halting_throughput` over `exp_rdt_loop_lora_gemma4`.
- **F#687** (2026-04-23) 2nd reuse: `exp_jepa_router_prediction_error` over `exp_jepa_adapter_residual_stream` (same parent as this experiment). 2nd-reuse promotion threshold flagged.
- **F#698** (2026-04-24) 3rd reuse: `exp_jepa_adapter_attention_output` over `exp_jepa_adapter_residual_stream` (same parent) + first F#666 compound sub-case. Promotion threshold confirmed.
- **F#699** (2026-04-24) 4th reuse: `exp_memento_compression_ratio_benchmark` over `exp_memento_gemma4_replication` (different parent; F#666-compliant KC set). Confirmed post-promotion routing applies across parent families.
- **F#682** (2026-04-23) parent PROVISIONAL: design-only, 4 target-gated KCs untested, MLX training loop not implemented.
- **F#666** — target-gated kill discipline. This experiment satisfies F#666 by construction (K1886 is target).
- **F#627** — Gemma 4 E4B LoRA baseline (K1886 RHS is defined relative to this in parent K1768).

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                  | Kind   | Measurement status                      |
| ----- | ---------------------------------------------------------------------- | ------ | --------------------------------------- |
| K1885 | L+2 prediction MSE > 2× L+1 prediction MSE (skip-connection ineffective) | proxy  | untested (preempt-blocked, F#669)       |
| K1886 | L+2 trained adapter behavioral quality > 3pp worse than L+1             | target | untested (preempt-blocked, F#669)       |

## §4 Unblock condition

Re-claimable when parent `exp_jepa_adapter_residual_stream` reaches `status=supported` at full scale via its `_impl` companion `exp_jepa_adapter_residual_stream_impl` (already filed P=1). Specifically:

1. Parent K1766 (SIGReg non-collapse proxy) and K1768 (GSM8K-Hard ≥ LoRA r=16 target) BOTH SUPPORTED.
2. Parent K1769 (lambda=0 ablation target: ≥5pp drop) SUPPORTED — establishes SIGReg is load-bearing, not cosmetic.
3. Parent K1767 (L_pred ratio proxy: step500/step50 < 0.5) SUPPORTED — establishes L+1 MSE is the "learning real dynamics" anchor K1885 compares against.

At that point, L+1 MSE (parent K1767) and L+1 behavioral quality (parent K1768) become target-validated anchors, and L+2 variant KCs become informative.

**No KC-augmentation needed** (unlike F#698-attention_output which required adding a target metric): K1886 already provides the target gate per F#666.

**Alternative unblock:** redesign child to train L+1 and L+2 predictors *jointly* within a single experiment as paired A/B (both MSE and behavioral quality measured on the same trained run). This would be `_impl`-class scope and is out of scope for drain window.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699 + reviewer.md §5. The unblock condition is external: parent's `exp_jepa_adapter_residual_stream_impl` (P=1) is the gate. If that lands SUPPORTED, this child becomes immediately re-claimable without further KC modification.

This distinction between novel-mechanism PROVISIONAL (which mandates `_impl`) and preempt-structural KILL (which does NOT spawn `_impl`) is canonical per F#687/F#698/F#699 precedent + reviewer.md §5.

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT attempt a "simpler" ablation (e.g. training only an L+2 adapter standalone and reporting absolute MSE, or computing behavioral quality against a non-JEPA baseline). Both shortcuts would substitute proxy phenomena for the skip-layer-prediction mechanism the KCs measure.

Pre-registered KCs are preserved verbatim and marked `untested (preempt-blocked)`. KC text in DB (`experiment get`) matches MATH.md §3 verbatim — no post-claim KC mutation.

No `_impl` inline-file obligation (`mem-antipattern-impl-follow-up-delegation`): that antipattern applies to novel-mechanism PROVISIONAL only. Preempt-structural KILL is structurally distinct and explicitly does **not** spawn `_impl` per F#687/F#698/F#699 precedent.

## §7 Meta-routing note (3rd child of same PROVISIONAL parent)

This experiment is the **3rd child** of `exp_jepa_adapter_residual_stream` (F#682) to hit preempt-KILL (after F#687 `exp_jepa_router_prediction_error` and F#698 `exp_jepa_adapter_attention_output`). The parent has blocked 3 distinct child designs in the drain window. The unblock leverage ratio for parent F#682 is now at least 3:1 (one `_impl` landing unblocks 3 child re-claims).

No new sub-family promotion threshold fires from this specific filing — F#698 already crossed the 3rd-reuse F#669-sub-family promotion; F#699 confirmed it; this is post-promotion. The meta-observation "same-parent repeat blocker" is a candidate for a separate sub-axis memory but only if a 4th same-parent child hits preempt-KILL (not yet).

## §8 Assumptions (A1-A12)

1. **A1:** Parent `exp_jepa_adapter_residual_stream` F#682 PROVISIONAL status reflects "design-only, no target training performed." Verified via `experiment get` output at claim time (all 4 KCs untested, status=provisional, notes explicitly state "Grounded: ... scaffold written" without any "training complete" language).
2. **A2:** F#669 canonical promotion at F#698 (3rd reuse) applies to same-parent 4th/5th reuse without re-derivation. Verified via F#698/F#699 evidence showing post-promotion routing was applied without re-deriving the theorem.
3. **A3:** "L+1 prediction MSE" in K1885 refers specifically to parent's JEPA-adapter next-embedding predictor output MSE at step 500, not any alternative "L+1" interpretation (e.g. MSE between two different LoRA-trained models). Supported by notes field: "Skip-layer prediction tests whether JEPA captures longer-range structure."
4. **A4:** "L+1 behavioral quality" in K1886 refers to parent's GSM8K-Hard accuracy from K1768, not a generic language-modeling metric. Supported by parent K1768 being the sole "behavioral quality" anchor in parent's design.
5. **A5:** The notes field reference "Skip-gram vs CBOW analogy" is a mechanism-intuition cue, not a KC grounding — analogies do not substitute for target-validated baselines under F#666/F#669 discipline.
6. **A6:** No CLI limitation prevents `experiment complete --status killed`. Verified at F#699 (4th reuse) which used identical routing without CLI issues.
7. **A7:** Global reference library hygiene (F#702) does NOT apply to preempt-KILL: F#702 hygiene-patch memory was anchored to PROVISIONAL novel-mechanism paths. Preempt-KILL fills `platform`, `dir`, `success_criteria`, `evidence` but `references` array remains empty (matching F#698/F#699 precedent).
8. **A8:** Verdict classification: `KILLED` per CLI status enum (not `preempt-killed` as a separate status). F#698/F#699 both used `--status killed`. Kill reason encoded in evidence + finding text.
9. **A9:** No `_impl` follow-up filed (per F#687/F#698/F#699 precedent). Parent's existing `exp_jepa_adapter_residual_stream_impl` (P=1) is the sole unblock gate.
10. **A10:** All 4 artifacts (MATH.md, run_experiment.py, results.json, PAPER.md) written per objective success-criteria. LEARNINGS.md + REVIEW-adversarial.md are analyst/reviewer responsibilities downstream.
11. **A11:** Platform pin `local-apple` per F#702 hygiene-patch. This is a preempt-KILL that writes a well-formed results.json but executes no MLX code — platform pin is for artifact-routing consistency, not measurement provenance.
12. **A12:** No new F#669 sub-axis is emitted. This is a canonical post-promotion reuse; the 5-reuse count is evidence-of-stability, not a new pattern.
