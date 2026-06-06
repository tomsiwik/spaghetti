# MATH.md — exp_sigreg_hedgehog_combined (PREEMPT-KILL, F#666-pure standalone, 10th instance)

## Claim under review
- **K1854**: "Combined loss Hedgehog adapter cos-sim > 0.05 worse than Hedgehog-only (SIGReg hurts)"
- **K1855**: "SIGReg statistic during training shows collapse at any checkpoint"

Notes (DB): tests `L = L_cos + λ · L_SIGReg` (Epps-Pulley anti-collapse regulariser combined with Hedgehog cos-sim distillation).

## Preempt-KILL conclusion (before any measurement)
**KILLED on F#666-pure-standalone (10th instance).** Both KCs are proxy-only with zero paired target metric (no task accuracy, no behavioral outcome, no oracle-gap). Per guardrail #1007, KILL requires `proxy ∧ target` both fail; SUPPORTED requires both pass. Proxy-only KCs cannot generate either verdict (proxy-PASS = tautological-support; proxy-FAIL = finding-about-proxy-not-kill). `depends_on=[]` ⇒ standalone, orthogonal to F#669 family.

**Secondary fire**: §5 tautological-inter-variant-delta (5th instance, 2nd inter-training axis after F#704). K1854 compares two training methods (Hedgehog+SIGReg vs Hedgehog-only) on a proxy delta with no per-variant base-anchor. Sub-bucket: derived-geometric (cos-sim) for K1854 + detection (collapse-statistic) for K1855 — multi-bucket fire across the F#711-refactored F#666-pure taxonomy.

**Tertiary fire**: hygiene-multi-defect (3+ defects: `success_criteria=[]`, `references=[]`, `platform=~`) per F#703 canonical 3+ threshold. Hygiene is secondary annotation — F#666-pure structural impossibility is the trigger; hygiene patch (F#702 path) is unavailable because there is **zero target-metric KC** to patch around.

## F#666 truth table (per-KC)

| KC | Form | Target metric? | Verdict path |
|----|------|----------------|--------------|
| K1854 | proxy-comparison `cos(combined) − cos(hedgehog) > 0.05` | NO (cos-sim is structural similarity proxy per F#666 §canonical) | proxy-only ⇒ inadmissible |
| K1855 | proxy-detection `SIGReg statistic indicates collapse at any checkpoint` | NO (Epps-Pulley statistic on training-time activations) | proxy-only ⇒ inadmissible |

Both KCs proxy-only, no target pair anywhere ⇒ F#666-pure-standalone.

## Five lemmas

### L1 — Both KCs are proxy-only under F#666 / guardrail #1007
F#666 canonical defines target metrics as task accuracy, behavioral outcome, or oracle-gap. K1854 measures `cos-sim(adapter_with_SIGReg, teacher) − cos-sim(adapter_without_SIGReg, teacher)` — a similarity proxy on intermediate activations, not on downstream behavior. K1855 measures the Epps-Pulley statistic on training-time hidden-state moments — a collapse-detection proxy with no paired downstream task accuracy. F#688 (measured `r ≈ 0.08` PPL↔task-quality coupling on this codebase) generalizes: cos-sim and distributional-collapse proxies are structurally orthogonal to behavior on Gemma 4 shallow regime per F#477. No claim about Hedgehog-on-tasks, Pierre-stack-quality, or composition-N can be made from either KC. Multi-bucket per F#711 taxonomy: K1854 = derived-geometric, K1855 = detection.

### L2 — F#666 truth-table inadmissibility (both branches forbidden)
**Branch K1854-PASS** (`cos(combined) − cos(hedgehog) ≤ 0.05`): tautological-support. Cos-sim is invariant to behavior; passing this proxy says nothing about whether the combined-loss adapter is better, worse, or identical on real tasks. Forbidden as a SUPPORTED verdict per F#666.
**Branch K1854-FAIL** (`cos(combined) − cos(hedgehog) > 0.05`): finding-about-the-proxy-not-a-kill. Even a 5pp cos-sim gap could correspond to identical or improved task behavior — proxy decoupling per F#666 mechanism. Forbidden as a KILL verdict.
**Branch K1855-PASS** (no collapse detected): tautological-support — Epps-Pulley non-rejection at training time does not imply any downstream task gain.
**Branch K1855-FAIL** (collapse detected at any checkpoint): finding-about-the-proxy. SIGReg-statistic collapse during training does not mean the resulting adapter fails on tasks — Pierre v3 trained without SIGReg achieves 0.41 behavioral despite collapse-prone hidden-state distributions per project memory.

### L3 — §5 secondary fire (5th instance, 2nd inter-training axis)
K1854 form `op(f(variant_i), f(variant_j)) op_2 δ` with no per-variant base-anchor matches §5 byte-for-byte. Variant axis is **inter-training-method** (Hedgehog+SIGReg vs Hedgehog-only) — same axis as F#704 (QA-format+cache-aware vs NTP-baseline), 2nd instance on this axis. Direction: `> 5pp` (failure direction = "SIGReg hurts"), matching F#704/K1552 polarity. Resolution identical to F#704: pair with per-variant base-anchor (e.g., `cos(combined, teacher) ≥ cos(base, teacher) + γ` AND `cos(hedgehog, teacher) ≥ cos(base, teacher) + γ`). Even with the base-anchor patch, the cos-sim metric remains a proxy under F#666 — §5 patch alone does not unblock; F#666-pure remains the dominant fire.

### L4 — Hygiene-multi-defect tertiary (3+ defects, F#703 threshold)
DB record shows: `success_criteria=[]` (defect 1), `references=[]` (defect 2), `platform=~` (defect 3). Three hard defects ≥ F#703 canonical 3+ promotion threshold. **Hygiene-patch path (F#702) is unavailable** because that path requires ≥1 target-metric KC to patch around (per F#702 impossibility-structure: "F#666-pure = 0 target-KCs ⇒ preempt-KILL; ≥1 target-KC + hygiene defects ⇒ hygiene-patch + _impl"). With zero target KCs, hygiene-patch is not applicable; F#666-pure preempts.

### L5 — Standalone topology, no parent dependency
`depends_on=[]`, `blocks=[]`. Standalone. Not F#669-family (no parent cascade). Not template-regression (no parent stripping — this is a fresh hypothesis combining SIGReg + Hedgehog without a measured parent in this codebase). Not proxy-only-lineage-inheritance (no parent finding to inherit proxy-only structure from). Cleanly inside F#666-pure-standalone.

## Standalone / parent topology
`depends_on=[]` (standalone). Not F#669-family. Not F#702 hygiene-patch (no target KC). Not template-regression (no parent strip). Not proxy-only-lineage-inheritance (no parent). Cleanly F#666-pure-standalone with §5 secondary + hygiene-multi-defect tertiary.

## Prediction table (what would measurement have shown if we ran it)

| # | Prediction | Grounding | If measured |
|---|------------|-----------|-------------|
| P1 | K1854 likely PASS (combined cos ≈ hedgehog cos within 5pp) | SIGReg added as small λ-weighted term; small perturbation to dominant L_cos signal | Tautological-support per L2 — inadmissible |
| P2 | K1855 likely FAIL at some checkpoint (Epps-Pulley sensitive to early-training transients) | SIGReg's own canonical use case is detecting collapse — by construction it fires often enough to be useful | Finding-about-proxy per L2 — inadmissible |
| P3 | If task-accuracy KC were added, decoupled from cos-sim verdict | F#688 PPL↔task r ≈ 0.08; F#666 §canonical | Would expose orthogonality |
| P4 | Adding `task_acc(combined) ≥ task_acc(hedgehog) − 5pp` AND `task_acc(combined) ≥ task_acc(base) + γ` would convert KC pair to F#666-admissible | F#666 / guardrail 1007 remedy | Unblock path |
| P5 | The combined-loss claim (SIGReg helps Hedgehog) is testable in a v2 with target gating | F#691 SIGReg cross-depth PROVISIONAL design + Hedgehog F#627-class adapter eval | v2 path documented |

## Unblock path (for v2 re-claim)
A valid follow-up `exp_sigreg_hedgehog_combined_v2_target_gated` would:
1. Add target KCs paired with each proxy:
   - **K-target-A**: `task_acc(combined_loss_adapter) ≥ task_acc(hedgehog_only_adapter) − 5pp` (per-pair delta with absolute floor)
   - **K-target-B**: `task_acc(combined_loss_adapter) ≥ task_acc(base) + γ` (base-beat per F#166)
   - **K-target-C**: `task_acc(hedgehog_only_adapter) ≥ task_acc(base) + γ` (per-variant base-anchor for §5 closure)
2. Restrict cos-sim/SIGReg-statistic KCs to **diagnostic** role, not verdict-bearing.
3. Use F#691-class SIGReg cross-depth + Hedgehog F#627-class adapter evaluation harness.
4. Cite F#682 (SIGReg layer-wise design-lock), F#691 (SIGReg cross-depth design-lock), F#713 (SIGReg N-composition design-lock), F#666 (target-gate guardrail), F#688 (proxy↔task decoupling measured).
5. Hygiene patch: populate `success_criteria` with operationalized SUPPORTED conditions (e.g., "K-target-A PASS AND K-target-B PASS"), `references` (F#682, F#691, F#713, F#666, F#688, F#627 + LeWM arxiv:2603.19312, Hedgehog arxiv:2402.04347), `platform` = `local-apple`.

## References
- **F#666** (conclusive, 2026-04-19, target-pair guardrail) — primary fire structural anchor
- **F#477** (killed, 2026-04-11, Gemma 4 q_proj r=6 MCQ FAIL — shallow-regime decoupling)
- **F#688** (measured PPL↔task r ≈ 0.08 — proxy-decoupling on this codebase)
- **F#704** (killed, 2026-04-24, §5 2nd instance, inter-training axis 1st)
- **F#709** (killed, 2026-04-24, §5 3rd instance, §5 promotion)
- **F#712** (killed, 2026-04-24, §5 4th instance, intra-rank sub-variant)
- **F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711** (F#666-pure-standalone instances 1-9; this = 10th)
- **F#711** (killed, 2026-04-24, taxonomy refactor execution; multi-bucket fire here: derived-geometric K1854 + detection K1855)
- **F#702** (provisional, 2026-04-24, hygiene-patch path — N/A here, requires ≥1 target KC)
- **F#703** (killed, 2026-04-24, hygiene-multi-defect 3+ canonical)
- **F#682** (provisional, 2026-04-24, SIGReg layer-wise design-lock)
- **F#691** (provisional, 2026-04-24, SIGReg cross-depth design-lock)
- **F#713** (provisional, 2026-04-24, SIGReg N-composition design-lock — completes triad)
- **F#627** (supported, 2026-04-19, Hedgehog-class r=6 adapter SUPPORTED on domain tasks)
- **F#166** (prerequisite gate — base-beat before composition / inter-variant delta)
- LeWM (arxiv:2603.19312) — SIGReg / Epps-Pulley grounding
- Hedgehog (arxiv:2402.04347) — cos-sim distillation grounding
