# LEARNINGS.md — exp_hedgehog_behavior_adapter_conciseness

## Core Finding
PROVISIONAL design-lock (F#725). 3rd behavior-axis instance in the Hedgehog
super-family (politeness F#683 → formality F#724 → conciseness THIS); triggers
the behavior-axis sub-cluster standalone-memory promotion per the 3-instance
threshold. 2nd **zero-proxy** KC design in the super-family, and 1st
**one-sided-safety** sub-variant (K1882 tests MMLU accuracy drop ≤ 3 pp,
degradation-only; K1881 tests length reduction ≥ 20 % as behavioral
acquisition target). Hygiene-patch applied; `_impl` follow-up filed.

## Why
- KC-design taxonomy bifurcation (paired / pure-proxy / zero-proxy) now
  supports two safety-target sub-variants: **two-sided orthogonality**
  (F#724 K1880) and **one-sided degradation-only** (F#725 K1882). Both are
  target-metric (not F#666-pure), so PROVISIONAL is the correct verdict.
- Conciseness ⊥ politeness, ⊥ formality on output-structure axis (one can
  be formal-verbose or concise-impolite), so the axis is genuinely novel —
  not a re-instance of a saturated sub-cluster.
- Both KCs are externally grounded (token count; MMLU canonical answers) —
  no tautology, no inter-variant delta, no proxy-only lineage.

## Implications for Next Experiment
1. **Behavior-axis sub-cluster is promoted** — future behavior-axis
   proposals beyond 3rd instance require explicit novelty justification
   beyond axis-extension (e.g. cross-axis interference tests).
2. **Zero-proxy KC design at 2 instances / 2 sub-variants** — promote the
   zero-proxy standalone memory at the 3rd instance (if a 3rd zero-proxy
   Hedgehog design is claimed, anchor it and file the standalone memory).
3. **26B teacher cache is the single highest-leverage unblock** — now
   blocks 10+ Hedgehog PROVISIONAL `_impl` dependents.
4. **F#683 `_impl`** remains the template-establisher for Hedgehog Phase
   A/B/C pipeline across all 10 Hedgehog PROVISIONALs; landing it unblocks
   engineering reuse (not result-dependency).
5. No antipattern update — clean novel-mechanism filing; memory for the
   behavior-axis sub-cluster promotion written to `.ralph/agent/memories.md`.
