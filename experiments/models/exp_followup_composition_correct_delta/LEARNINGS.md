# LEARNINGS: followup_composition_correct_delta

**Status:** KILLED preemptive (2026-04-18). 4th antipattern-017 instance.

## Core Finding

The DB's claim that `exp_p1_t2_multi_domain_5` is `supported` (K1047 PASS:
+82/+46/+22/+50/+56pp across 5 domains) is **filesystem-falsified** — 5-of-5
adapter directories hold only `adapter_config.json` with no `adapters.safetensors`.
K1548 (composed-PPL ≤ 2× solo-PPL at N=5) is therefore unmeasurable; Theorem 2's
linear Frobenius bound is mathematically sound but does NOT trivially imply K1548
(weight-norm → PPL is nonlinear), so kill is "missing measurement", not "proven".

## Why

- Pure antipattern-017 cascade: cited adapters are stubs.
- DB row out of sync with disk — original P1 training experiment marked passed
  without persisting weights (or weights deleted afterward without status update).
- This is the **4th instance** in the audit (baseline_eval + J0 + M0 + this);
  pattern is now systemic, not episodic.

## Implications for Next Experiment

1. **P11.ADAPTER-REBUILD is the atomic unblock** for every composition-class
   experiment in the queue (this v2, M0 v2, J0 v2, and any future Σ ΔW_i work).
   Resolve F#557 (`mlx_lm.lora` subprocess crash) first; without it, rebuild
   itself is blocked.
2. **Reviewer pre-flight is now mandatory** for any composition or adapter-load
   experiment — the canonical sibling-check grep belongs in PROCEED checklist
   (see updated antipattern-017 memory).
3. **DB integrity audit recommended** as a one-shot meta-experiment: scan all
   `status=supported` rows whose evidence references adapter weights, flag any
   whose cited paths fail the `.safetensors size > 0` check. Current count
   suggests several silent failures across P1/P11 lineage.
4. **Theorem 2 stands as math-only finding** — useful as forward reference for
   v2 design (predicts K1548 should pass at N=5 with 1/N or unscaled sum given
   bounded individual norms), but not a substitute for measurement.
5. **Distinguish from antipattern-020** (cascade-dependent design, M0 instance):
   here the dependency target *should* exist (P1 was marked supported); there
   the dependencies were known-killed upstream. Different remediation:
   antipattern-017 → audit + rebuild; antipattern-020 → redesign without dep.

## References

- F#14 (1/N scaling resolves catastrophe), F#23 (equal-weight catastrophe),
  F#199 (A-loading bug), F#544 (canonical bug location), F#557 (lora subprocess).
- antipattern-017 (4 instances now); antipattern-020 (cascade-dep, 1 instance).
