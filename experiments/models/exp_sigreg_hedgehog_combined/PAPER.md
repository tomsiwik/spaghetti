# PAPER.md — exp_sigreg_hedgehog_combined

**Verdict:** KILLED (preempt-structural, pre-measurement)

**Antipattern (primary):** F#666-pure-standalone — **10th instance**, multi-bucket fire across the F#711-refactored taxonomy: K1854 in derived-geometric bucket (cos-sim), K1855 in detection bucket (Epps-Pulley collapse-statistic).

**Antipattern (secondary):** §5 tautological-inter-variant-delta-ignores-base-baseline — **5th instance, 2nd inter-training axis** (1st was F#704). K1854 compares Hedgehog+SIGReg vs Hedgehog-only on a proxy delta with no per-variant base-anchor.

**Antipattern (tertiary):** hygiene-multi-defect (3+ defects: `success_criteria=[]`, `references=[]`, `platform=~`). F#702 hygiene-patch path **unavailable** because zero target-metric KCs exist to patch around.

## Claim
- **K1854**: Combined-loss Hedgehog adapter cos-sim > 0.05 worse than Hedgehog-only ("SIGReg hurts").
- **K1855**: SIGReg statistic during training shows collapse at any checkpoint.
- Mechanism (DB notes): `L = L_cos + λ · L_SIGReg` testing whether SIGReg's Epps-Pulley anti-collapse regulariser improves Hedgehog distillation stability.

## Prediction-vs-measurement

| # | Prediction | Basis | Measured | Verdict |
|---|------------|-------|----------|---------|
| P1 | K1854 likely PASS (small λ-perturbation to dominant L_cos) | Optimization continuity | Not measured (preempt) | Tautological-support per L2 — inadmissible |
| P2 | K1855 likely FAIL at some checkpoint (Epps-Pulley sensitive to early-training transients) | SIGReg's canonical use case | Not measured (preempt) | Finding-about-proxy per L2 — inadmissible |
| P3 | Cos-sim and task-accuracy decoupled on this codebase | F#688 (r ≈ 0.08 measured) | Not measured (preempt) | Confirms F#666-pure rationale |
| P4 | Adding `task_acc(combined) ≥ task_acc(hedgehog) − 5pp` AND `task_acc(*) ≥ task_acc(base) + γ` would convert KCs to F#666-admissible | F#666 / guardrail #1007 / F#166 remedy | Not measured (preempt) | Unblock path |
| P5 | Domain-specific eval on F#627-class adapters would allow non-degenerate target measurement | F#627 SUPPORTED on domain tasks | Not measured (preempt) | v2 path |

## Summary
Both KCs are proxy-only with no paired target metric anywhere. K1854 is a cos-sim similarity measure on intermediate activations (derived-geometric bucket per F#711 refactor). K1855 is an Epps-Pulley collapse-detection statistic on training-time hidden-state moments (detection bucket). Neither has a downstream task-accuracy, behavioral-outcome, or oracle-gap pair.

Per F#666 / guardrail #1007: KILL requires `proxy ∧ target` both fail; SUPPORTED requires both pass. Both branches of each KC are **structurally inadmissible**:
- **PASS** ⇒ tautological-support (proxy invariant to behavior per F#688 r ≈ 0.08 coupling)
- **FAIL** ⇒ finding-about-proxy-not-kill (proxy decoupling from target per F#666 mechanism)

This is the **10th F#666-pure-standalone preempt-KILL** in the current drain window. F#700–F#711 = 9 prior instances; this completes the 10th. Multi-bucket fire (derived-geometric K1854 + detection K1855) — first multi-bucket instance post-F#711 taxonomy refactor.

§5 fires secondarily on K1854's inter-training-method comparison (Hedgehog+SIGReg vs Hedgehog-only) without per-variant base-anchor — 5th §5 instance, 2nd on the inter-training axis (1st was F#704 QA-format vs NTP). Even with §5 base-anchor patch, the cos-sim metric remains a proxy under F#666 — §5 patch alone does not unblock; F#666-pure is the dominant fire.

Hygiene tertiary: 3 hard defects ≥ F#703 canonical 3+ threshold. F#702 hygiene-patch path is **unavailable**: that path applies only to pre-regs with ≥1 target-metric KC. With zero target KCs, hygiene-patch is inapplicable; F#666-pure preempts.

## Assumptions logged (autonomy guardrail #1008)
- "Cos-sim" in K1854 read as cosine-similarity between adapter output and teacher activation (canonical Hedgehog metric per arxiv:2402.04347), not adapter-weight cos-sim. Both readings are F#666-proxy.
- "SIGReg statistic shows collapse at any checkpoint" in K1855 read as Epps-Pulley test rejecting the null at any training checkpoint (canonical SIGReg / LeWM use per arxiv:2603.19312). Operational ambiguity does not affect the F#666-pure verdict (proxy-only regardless of operational variant).
- "λ" weight in `L = L_cos + λ · L_SIGReg` is unspecified in DB notes; unspecified hyperparameter does not affect KC structure (still proxy-only).
- F#688 PPL↔task r ≈ 0.08 coupling (this codebase) generalizes to cos-sim↔task and Epps-Pulley↔task per F#666 §canonical "structural-stability orthogonal to behavior on Gemma 4 shallow regime". Defensible reading; alternative (cos-sim and task tightly coupled) is contradicted by F#688 + F#666.
- Hygiene patch unavailability is asymmetric per F#702: 0 target KCs ⇒ no patch surface ⇒ preempt-KILL stands.

## Unblock path
File `exp_sigreg_hedgehog_combined_v2_target_gated`:
1. **Pair each proxy KC with a target KC** (per F#666):
   - K-target-A: `task_acc(combined_loss_adapter) ≥ task_acc(hedgehog_only_adapter) − 5pp` (per-pair delta with absolute floor)
   - K-target-B: `task_acc(combined_loss_adapter) ≥ task_acc(base) + γ` (base-beat per F#166)
   - K-target-C: `task_acc(hedgehog_only_adapter) ≥ task_acc(base) + γ` (per-variant base-anchor for §5 closure)
2. **Demote** cos-sim and SIGReg-statistic to **diagnostic** role, not verdict-bearing.
3. **Eval harness**: F#691-class SIGReg cross-depth + Hedgehog F#627-class adapter on held-out domain val sets (code/math/medical, where r=6 v_proj+o_proj or q_proj adapters exist on disk per F#627).
4. **Hygiene patch**: populate `success_criteria` (e.g., "K-target-A PASS AND K-target-B PASS AND K-target-C PASS"), `references` (F#682, F#691, F#713, F#666, F#688, F#627, LeWM, Hedgehog), `platform` = `local-apple`.
5. **Cite** F#682 (SIGReg layer-wise design-lock), F#691 (SIGReg cross-depth design-lock), F#713 (SIGReg N-composition design-lock — closes triad), F#688 (proxy↔task decoupling measurement), F#666 (target-gate guardrail).
