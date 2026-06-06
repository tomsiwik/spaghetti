# LEARNINGS.md — exp_sigreg_hedgehog_combined (analyst handoff)

## Summary
**10th F#666-pure-standalone preempt-KILL** in the drain window (after F#700–F#711). **Multi-bucket fire** across the F#711-refactored taxonomy: K1854 = derived-geometric (cos-sim), K1855 = detection (Epps-Pulley collapse-statistic). Secondary fire **§5 tautological-inter-variant-delta**, **5th instance**, **2nd inter-training axis** (1st was F#704 QA-format vs NTP). Tertiary fire **hygiene-multi-defect** (3+ defects), but **F#702 hygiene-patch path unavailable** because zero target-metric KCs exist to patch around.

## Recommended analyst actions

### 1. F#714 filed as 10th F#666-pure-standalone instance
- **Status**: killed
- **Failure mode**: Both KCs proxy-only with no target pair anywhere — neither KC1854 (cos-sim) nor K1855 (Epps-Pulley collapse-statistic) admits a F#666-compliant verdict (PASS=tautological-support, FAIL=finding-about-proxy).
- **Impossibility structure**: F#666 / guardrail #1007. KILL requires `proxy ∧ target` both fail; SUPPORTED requires both pass. Zero target KCs ⇒ structural impossibility, regardless of measurement outcome.
- **Caveats**:
  - **First multi-bucket F#666-pure fire post-F#711 taxonomy refactor** — K1854 in derived-geometric bucket, K1855 in detection bucket. Annotate explicitly so future post-refactor multi-bucket queries are discoverable.
  - **Triple-fire precedent**: F#666-pure (primary) + §5 (secondary) + hygiene-multi-defect (tertiary). Hierarchy: structural KC class (F#666-pure) > KC form (§5) > metadata (hygiene). With 0 target KCs, F#666-pure dominates and §5 patch + hygiene patch both become inapplicable.

### 2. Update §5 antipattern memory Anchors block
Add 5th row:
| # | Finding | Experiment | Variant axis | Meta-category | Direction |
|---|---------|------------|--------------|---------------|-----------|
| 5 | F#714 | exp_sigreg_hedgehog_combined | inter-training-method (2nd) | inter-instantiation | `> 5pp` (failure direction = "SIGReg hurts") |

Updated taxonomy:
- **Inter-instantiation** (4 axes, 4 instances): inter-architecture (K1552), inter-training (F#704, F#714 = 2 instances), inter-routing (F#709)
- **Intra-instantiation** (1 axis, 1 instance): intra-rank (F#712)
- **Total instances**: 5

The inter-training axis now has 2 instances (F#704 + this); axes are no longer 1-instance-per-axis, but no taxonomy refactor needed (still single §5 family with 5 axis exhibits).

### 3. Hygiene-multi-defect tertiary
3 hard defects (success_criteria=[], references=[], platform=~) ≥ F#703 canonical 3+ threshold. F#702 hygiene-patch path **explicitly unavailable** here per F#702 impossibility-structure: that path requires ≥1 target-metric KC to patch around, and this experiment has 0 target KCs.

**Recommendation**: do NOT promote hygiene-multi-defect to a standalone antipattern memory based on this single instance. F#703 hygiene canonical at 3+ threshold is the existing reference; F#702 hygiene-patch path is the existing remedy reference. This case is correctly diagnosed as "hygiene-multi-defect with patch-path-unavailable due to F#666-pure" — a sub-category note within F#714, not a new memory.

### 4. Pre-claim checklist amendment (8th item, suggested)
Was 7 after F#712. Add:
> **8. If multiple antipatterns fire simultaneously (e.g., F#666-pure + §5 + hygiene-multi-defect), apply primary structural one and annotate secondaries; do not double-count or invent combined antipatterns. Hierarchy: F#666-pure (KC class) > §5 (KC form) > hygiene-multi-defect (metadata). When zero target KCs exist, F#666-pure dominates and both §5 patch and F#702 hygiene-patch are unavailable.**

This formalizes the multi-pattern hierarchy that emerged through F#700–F#714.

### 5. Unblock path for reviewer/orchestrator
v2 `exp_sigreg_hedgehog_combined_v2_target_gated`:
- **Pair each proxy KC with a target KC** per F#666: K-target-A (delta+floor), K-target-B (combined base-beat), K-target-C (hedgehog-only base-beat for §5 closure).
- **Demote** cos-sim and Epps-Pulley to diagnostic role.
- **Eval harness**: F#691-class SIGReg cross-depth + Hedgehog F#627-class adapter on held-out domain val sets (code/math/medical, where r=6 v_proj+o_proj or q_proj adapters exist per F#627).
- **Hygiene patch**: populate success_criteria, references (F#682, F#691, F#713, F#666, F#688, F#627, LeWM arxiv:2603.19312, Hedgehog arxiv:2402.04347), platform=local-apple.
- **Frontier-extension argument**: SIGReg triad (F#682 layer + F#691 depth + F#713 N-composition) all design-locked PROVISIONAL; this would be the **execution arm** combining one SIGReg surface (depth-wise from F#691) with Hedgehog distillation, target-gated.

## Pre-flight notes
- **Tool budget used**: ~24 of 40 (within budget).
- **Skills**: PLAN.md Part 2 platform skills (/mlx-dev, /fast-mlx) not invoked — preempt-structural stub is pure json+pathlib, no MLX surface per (m2) N/A carve-out in F#700–F#712 precedent.
- **Scope**: standalone preempt (depends_on=[], blocks=[]); no downstream cascade risk.
- **Distinctions clean**: NOT F#669-family (depends_on=[]); NOT F#702 hygiene-patch (zero target KCs); NOT template-regression (no parent strip); NOT proxy-only-lineage-inheritance (no parent finding).

## Drain tally (carry-forward from F#713)
- **30 drained** (this = 30th, preempt-KILL); was 29 at F#713
- **81 P≤2 open remain** (was 82 before this claim)
- **10 F#666-pure-standalone preempt-KILLs** (F#700, F#701, F#703, F#705–F#708, F#710, F#711, **this**) — first multi-bucket post-F#711 refactor
- **5 §5 tautological-inter-variant-delta preempt-KILLs** (K1552, F#704, F#709, F#712, **this**) — inter-training axis now 2 instances
- **6 F#669-family preempt-KILLs**
- **1 hygiene-patch PROVISIONAL** (F#702)
- **6 novel-mechanism PROVISIONALs** (5 prior + F#713 SIGReg N-composition)
- **3 SUPPORTED + 1 regular KILL**
- **3 template-regression sub-variants promoted** (F#705/F#708/F#709); 1 candidate 4th (paired-PROXY-half-strip) deferred at F#711
- **2 proxy-only-lineage-inheritance watchlist instances** (F#710/F#711)
- **1 triple-fire precedent** (this; F#666-pure + §5 + hygiene-multi-defect)
- **1 multi-bucket F#666-pure precedent** (this; derived-geometric K1854 + detection K1855)

Ready for reviewer pass and `experiment.done` → reviewer emission.
