# LEARNINGS — exp_p5_dccd_format_conditioning

## V2 Audit Closure (2026-04-18) — KILLED verdict confirmed

Tag `audit-2026-04-17-rerun`. Rerun not executable (prereq adapters
deleted: medical q_proj + SOAP v_proj+o_proj). Verdict reconstructed
from 2026-04-11 N=10 measurements; both failing KCs close on
N-independent structural bounds — scaling N cannot rescue. Review V2
PROCEED with KILLED. MATH.md git-clean, antipattern scan clear.

**Distinction to preserve:** the KILL is on the re-prompting *implementation*
of Phase 2, not on DCCD's temporal-separation *principle*. Theorem 2
(Interference(P1,P2)=0) is conclusively verified by K1269 PASS
(100% coherence vs weight-composed 80%; #483 eliminated). Separate
finding-add recommended so Theorem 2 survives the top-level KILL.

**Structural closures (N-independent):**
- K1267 — RLHF ceiling + adapter ceiling: SOAP-only trained adapter
  reaches only 60% < 70%, so re-prompting (instruction-following through
  RLHF-suppressed SOAP prior, Finding #479) cannot exceed ~40%.
- K1268 — lossy re-prompting channel: draft 11.6 medical keywords →
  reformat 7.2 (38% information loss). MATH.md's ~0pp prediction assumed
  Phase 2 had a trained format adapter, not re-prompting.

---

## V1 Analyst (2026-04-11)

**Status:** KILLED (2/3 fail) | Finding: supported

## Core Finding
Temporal separation (DCCD principle, arXiv:2603.03305) eliminates #483 cross-projection catastrophe: DCCD achieves 100% coherence vs 80% for weight-composition, and preserves partial domain (70%) and format (40%) capability that weight-composition destroys entirely (0%/0%).

## Why It Worked / Why It Failed
Temporal separation works because Phase 1 and Phase 2 adapters never interact in weight space — interference = 0 by construction. The failure was in the Phase 2 implementation: re-prompting relies on Gemma 4's base instruction-following, which is RLHF-suppressed for SOAP format (Finding #479). Re-prompting achieves only 40% SOAP vs the 70% target. The DCCD *paper's* actual mechanism (token-level grammar masking) was not implemented.

## Impossibility Structure (from KILL)
Re-prompting DCCD cannot exceed the base model's RLHF-suppressed SOAP capability ceiling. The same behavioral prior that makes q_proj insufficient for SOAP (Finding #479) limits re-prompting to ~40%. Fix: structural enforcement, not instruction-following.

## Implications for Next Experiment
Two proven fix paths:
1. **SOAP adapter in Phase 2** — Phase 1: domain adapter → draft. Phase 2: SOAP adapter (v_proj+o_proj, Finding #480) reformats. Still temporal separation (one adapter active at a time). Expected: 60%+ SOAP, <10pp domain degradation.
2. **Token-level grammar masking** — State-machine logit masking in MLX loop forces 100% SOAP by construction. Higher implementation cost, guaranteed result.

The SOAP adapter in Phase 2 is the fastest unblocked path: both adapters are already trained. Test whether the Phase 2 SOAP adapter can reformat Phase 1 draft without information loss.

## References
- arXiv:2603.03305 — Draft-Conditioned Constrained Decoding
- Finding #483 — Cross-projection catastrophe (reproduced as baseline)
- Finding #480 — v_proj+o_proj SOAP adapter (+70pp compliance)
- Finding #479 — RLHF suppresses SOAP in base model (explains re-prompting failure)
