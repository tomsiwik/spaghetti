# LEARNINGS.md — exp_hedgehog_data_augmentation_prompt_rephrase

**Finding #723 · PROVISIONAL (novel-mechanism design-only) · Analyst 2026-04-24**

## Core Finding

Paired-target-anchored KC design (K1877 target behavioral Δ > +3 pp on F#683 rubric; K1878 proxy cos-sim variance > 0.10 intra-variant) filed as 8th Hedgehog-framework PROVISIONAL and NEW 5th sub-type in Hedgehog-ablation super-family: **data-augmentation-ablation** (5× prompt rephrase at teacher-temp 1.2). All 4 PROVISIONAL artifacts present; adversarial (a)–(u) clean; F#702 hygiene-patch applied (platform, success_criteria #95, evidence); references INCOMPLETE per F#702 CLI limitation precedent. Stub ran 1.7 s; main() never raises. `_impl` follow-up filed at P=3 with K1877/K1878 verbatim and explicit transitive blocker on F#683 `_impl`.

## Why

K1877 is named-target (behavioral-quality judge on F#683 rubric, rule 1007); K1878 is intra-variant absolute threshold (§5 does not fire — not an inter-variant delta). F#666 pairing satisfied → novel-mechanism + hygiene-secondary pairing canonical → F#702 path available → PROVISIONAL (not preempt-KILL). Distinct from F#720/F#721/F#722 which were all pure-proxy inter-variant deltas → F#666-pure triple-fire KILLs. KC-design bifurcation pattern now axis-invariant across 5 Hedgehog-ablation sub-types × 12 instances (paired-target → PROVISIONAL; pure-proxy → KILL).

## Implications for Next Experiment

- **Hedgehog-framework PROVISIONAL pile: 7 → 8 designs / 0 measurements.** 26B teacher cache remains the single standalone-prereq-task candidate blocking 8+ dependents.
- **Transitive blocker cascade:** F#683 `_impl` now gates 5 ablation `_impl`s (F#719/F#720-v2/F#721-v2/F#722-v2/F#723-v2 all reuse its corpus/rubric/held-out slice). F#683 `_impl` is the single highest-leverage unblocking task.
- **data-augmentation-ablation sub-type at 1 instance;** promote sub-type standalone memory at 3rd instance on same sub-type (per 3-instance-on-same-sub-type precedent).
- **KC-design bifurcation memory candidacy:** reviewer flagged promotion at *next* Hedgehog-ablation instance if pattern holds. Defer to 13th-instance trigger.
- **No preempt-KILL bucket anchors** (F#723 is PROVISIONAL). Update only `mem-pattern-novel-mech-primary-plus-hygiene-secondary-pairing` anchors (4th instance).

## References

- F#683/F#684/F#696/F#697/F#717/F#718 axis-extension precedents; F#719 loss-variant-ablation 1st PROVISIONAL.
- F#720/F#721/F#722 pure-proxy triple-fire KILLs (contrast class).
- Wei 2024 prompt-rephrasing SFT; Alpaca/Self-Instruct; F#469 axis-dependent gain.
- `mem-pattern-novel-mech-primary-plus-hygiene-secondary-pairing` (4th anchor pending).
