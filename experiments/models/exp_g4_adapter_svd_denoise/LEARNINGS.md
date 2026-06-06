# LEARNINGS.md — exp_g4_adapter_svd_denoise

**Verdict:** KILLED preempt-structural. F#716. 12th F#666-pure-standalone instance. 2nd triple-fire precedent (after F#714).

## Core Finding
SVD-truncation of Gemma 4 r=6 adapter deltas with two PPL-only KCs (K1864 truncated-vs-original >0.05 PPL; K1865 truncated-composition ≤ untruncated-composition PPL) triple-fires F#666-pure (primary, PPL bucket saturates at 3rd instance → confirmed-recurrent) + §5 tautological-inter-variant-delta (secondary, 2nd intra-adapter-rank-delta sub-variant after F#712) + hygiene-multi-defect (tertiary, 3 defects; F#702 patch path structurally unavailable ⇒ 3rd confirmation, promotion-threshold).

## Why
Per guardrail 1007, PPL is a proxy (r≈0.08 with task accuracy in-repo). Zero target KCs ⇒ F#666 2-outcome truth-table (tautological-PASS / finding-about-proxy-FAIL) is unidentifiable regardless of measurement. §5 fires on both KCs because neither anchors to `PPL_base`; parent F#477 K1226 adapted_acc=0.480 puts Gemma 4 r=6 adapters in the collapse basin where rank-truncated and untruncated variants plausibly co-degenerate to `PPL_base`, trivially satisfying the delta. F#702 hygiene-patch is structurally unavailable because populating success_criteria/platform/references leaves PPL-only KC set with no target-KC to patch around — identical impossibility structure to F#714 and F#715.

## Implications for Next Experiment
1. **PPL bucket now confirmed-recurrent** (3 instances: F#705/F#708/F#716); analog to routing bucket; no taxonomy refactor per F#711 bucket-level convention.
2. **F#702-unavailability reaches 3-instance promotion threshold** (F#714 triple, F#715 double, F#716 triple) — impossibility-structure "F#666-pure saturation ⇒ F#702 patch structurally unavailable" is now stable across 3 instances and 2 fire-modes; candidate for standalone memory.
3. **2nd triple-fire precedent** — hierarchy (F#666-pure > §5 > hygiene) from F#714 holds across different §5 axes (inter-training F#714 vs intra-adapter-rank F#716). Triple-fire mode is recurrent.
4. **§5 intra-adapter-rank-delta confirmed-recurrent sub-variant** (F#712 + F#716 = 2 intra-instantiations); sub-variant split still deferred per F#712 caveat (conservative threshold).
5. **Unblock path v2_target_paired**: pair per-variant base-anchor (`acc_truncated ≥ acc_base − ε`) + inter-variant target delta (`acc_truncated ≥ acc_untruncated − δ`) + PPL secondary sanity. Cite F#712 + F#627 + F#133. File new id (not `experiment update`).

## Antipattern capture
No new memory. Three existing memories updated: F#666-pure (12th instance, PPL confirmed-recurrent); §5 (6th instance, 2nd intra-adapter-rank-delta sub-variant); F#702-unavailability promoted to standalone at 3-instance.
