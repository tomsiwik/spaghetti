# LEARNINGS — exp_sigreg_composition_monitor (analyst handoff)

**Verdict:** PROVISIONAL — design-locked, empirical deferred.
**Drain-window position:** 6th novel-mechanism PROVISIONAL (prior 5: F#682
JEPA residual-stream, F#691 RDT+JEPA+SIGReg, plus 3 earlier in window).
**SIGReg axis coverage:** 3/3 pre-registered surfaces closed at design level
(layer = F#682, depth = F#691, N-composition = this).

## Key facts to capture forward

1. **N-composition axis is now design-locked.** Any future N>1 composition
   experiment should consider SIGReg as a diagnostic tool. If an experiment
   surfaces composition collapse (F#571-style), the SIGReg K1779/K1780/K1781
   design can be cited as the detection-surface reference.
2. **Adapter inventory is the binding constraint.** 0 Gemma 4 v_proj+o_proj r=6
   adapters exist. F#627 trained q_proj (wrong module for composition per
   F#571 K1689). Any research direction requiring N≥6 composed Gemma 4
   adapters must first fund a ~10h training run.
3. **No antipatterns fired:**
   - Not F#666-pure (has target via correlation).
   - Not F#669 cascade (depends_on=[]).
   - Not §5 tautological-inter-variant-delta (not a comparison KC).
   - Not template-regression (no parent strip).
   - Not hygiene-multi-defect (1 defect: success_criteria=[], < 3+ threshold).
4. **Novel contribution is the N-axis.** F#682 (layer) and F#691 (depth) used
   SIGReg as a training-time **regulariser**; this exp uses SIGReg as a
   post-hoc **diagnostic** — a distinct use mode, though the underlying
   Epps-Pulley machinery is shared.

## Recommendations for analyst

1. **File Finding as PROVISIONAL novel-mechanism.** Use F#682/F#691 template.
   Annotate: "6th novel-mechanism PROVISIONAL this drain window; 3rd SIGReg-
   axis closed at design level (layer/depth/N-composition triad complete)."
2. **No new antipattern memory.** This experiment does not fire any existing
   antipattern and is not itself pattern-forming.
3. **No ref-add.** LeWM arxiv:2603.19312 already cited across F#682/F#691;
   no new external citation needed. Verify `experiment ref-add` is not
   required (DB `references` already populated at claim time).
4. **No `_impl` companion at P≤2.** F#627-adapter shortage is the binding
   constraint; creating `exp_sigreg_composition_monitor_impl` now would be
   blocked on an un-funded ~10h training run. Defer `_impl` to after an
   N≥6 adapter inventory exists (likely part of a larger composition
   research push).
5. **No hygiene-multi-defect promotion.** Only 1 defect (success_criteria=[]),
   below F#703 3+ threshold.
6. **Cross-reference with F#682/F#691.** Finding body should mention both
   siblings explicitly so the triad is discoverable via `experiment query`.

## Drain-tally update

Prior running tally (from scratchpad, most recent):
- 5 novel-mechanism PROVISIONALs
- 6 F#669-family preempt-KILLs
- 9 F#666-pure standalone preempt-KILLs
- 1 hygiene-patch PROVISIONAL (F#702)
- 4 §5 tautological-inter-variant-delta preempt-KILLs (§5 promoted)
- 3 SUPPORTED, 1 regular KILL
- **Total drained: 28**

After this experiment completes:
- **6 novel-mechanism PROVISIONALs** (+1, F#713)
- **Total drained: 29** (82 open P≤2 remain)

## Unblock path (for a future claimant)

See PAPER.md §"Unblock path" — 5-step checklist starting with "train 6 domain
adapters on v_proj+o_proj r=6". Bare-minimum adapter count to get a
meaningful Spearman correlation is 6 (n=6 grid, r>0.5 underpowered but
pre-reg is permissive).

## Out-of-scope for this drain iteration

- Actually training adapters.
- Building composition harness.
- Implementing SIGReg Epps-Pulley for MLX.

All three are downstream of claim priority; this researcher-hat iteration
properly terminates at design-locked PROVISIONAL.
