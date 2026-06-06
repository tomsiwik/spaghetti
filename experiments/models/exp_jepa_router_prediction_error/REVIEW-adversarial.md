# REVIEW-adversarial.md — exp_jepa_router_prediction_error

## Verdict: KILL (preempt-structural, F#669 reuse; F#687 filed)

Researcher's preempt-KILL is correct. Parent `exp_jepa_adapter_residual_stream` is still `provisional`
(F#682) at review time, so every KC here remains transitively unidentifiable per MATH.md §1 theorem.
DB `status=killed`, evidence attached, finding F#687 landed (verified via `experiment finding-list
--status killed`).

## Adversarial checklist

- (a) `results.json.verdict=KILLED` matches DB `status=killed` → PASS.
- (b) `all_pass=false`; all 4 KCs `result="untested"` (structural, not FAIL) → consistent with
  preempt-KILL (distinct from target-gated kill per F#666/F#669).
- (c) PAPER.md verdict line: `KILLED (preempt, F#669)` → matches.
- (d) `is_smoke=false` → correct (no smoke pass was run).
- (e) MATH.md first version; no post-hoc KC mutation → PASS.
- (f) No tautology — no KC was measured at all; "untested" preempt-reasons cite F#669/F#682,
  not algebraic identity.
- (g) KC IDs #1775–#1778 in code match DB and MATH.md §3 table → PASS.
- (h) No composition code (no `run_experiment.py` MLX path) → PASS vacuously.
- (i–l) No LoRA scale, no single-sample routing, no `shutil.copy`, no hardcoded pass → PASS vacuously.
- (m) No target model loaded; MATH.md §0 pins the *intended* base model per F#627 even though no
  weights touch → honest disclosure.
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with explicit "not invoked — no code path" — this
  is the canonical form for design-only / preempt-structural filings. PASS.
- (n–q) Eval-integrity items vacuous — no eval was run.
- (t) **Target-gated kill (F#666) does NOT apply here** — this is not a kill on a proxy-FAIL;
  it is a structural preempt where NO KC was measured (proxy or target). F#669 is the applicable
  precedent, not F#666. Verdict routing correct.
- (u) No scope-changing fixes — no scope was executed to change.
- (r) PAPER.md prediction-vs-measurement table present with all rows "not measured" + explicit
  "untested" verdict → PASS.
- (s) §1 theorem reasoning sound: all 4 KCs transitively depend on parent-trained `pred_i`
  with target-validated dynamics. K4 flagged correctly as *superficially measurable* but
  *vacuous-as-signal*.

## F#669 reuse vs. promotion

F#669 caveat: "Proposed new sub-axis may be promoted to standalone finding on second reuse."
This is the **3rd+ application** (F#669 → F#671/#672 → F#687). Promotion threshold hit. Non-blocking
flag for analyst: consider promoting "preempt-child-target-unverified" to a canonical routing
pattern in `.ralph/hats/reviewer.md` §5 alongside the novel-mechanism / macro-scope PROVISIONAL
sub-cases. Symmetric structural verdict; same required-artifact pattern (MATH.md §1 theorem,
graceful-failure `run_experiment.py`, unblock condition, no `_impl` companion).

## Meta flags (non-blocking)

1. **5th consecutive claim-picker mispick** this researcher-hat window. Researcher emitted
   `meta.picker_bug` and manually routed. Systemic picker bug — all three axes
   (tag-saturation, cohort-saturation, priority-inversion) fired simultaneously. Human-operator
   touch on loop-runner claim logic is the right remediation; analyst/reviewer cannot fix it
   from inside the loop.
2. **Preempt-drain is a valid backlog-draining pattern.** Net: 1 P2 → killed; P≤2 open reduced
   from 3 P2 preempt-candidates to 2. Two candidates remain (`hedgehog_composition_polite_refactor_js`,
   `user_adapter_from_memento_distillation`); both eligible for the same pattern next iteration.

## Assumptions

- Parent `exp_jepa_adapter_residual_stream` status at review time is `provisional` (verified).
  If the parent is promoted to `supported` later, this child becomes re-claimable via MATH.md §4.
- No redesign attempted this iteration (e.g. random-predictor null-router ablation to
  remove the parent dependency). Out of scope per drain objective; noted in PAPER.md "Assumptions".

## Route

Emit `review.killed` with F#687 reference and drain-pattern payload.
