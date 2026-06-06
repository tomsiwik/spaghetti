# LEARNINGS: exp_shine_gemma4_metalora (SHINE S3)

## Core Finding
Multi-projection (q+v+o) is 7.7x better than q-only M2P; meta LoRA kills performance
when param/data ratio is ~5000:1 (26.6M params, 40 chunks). Centroid trap (cos=0.988)
persists — cos² diversity loss is insufficient for homogeneous data.

## Why
q-only LoRA actually HURTS (ratio 1.16 > 1.0); v_proj and o_proj carry format priors
(validates Finding #480). Meta LoRA overfits to extraction patterns, not context diversity.
cos² diversity penalty cannot force geometric separation when all passages share the same
optimal LoRA basin — the data must be geometrically diverse, not the loss.

## Implications for Next Experiment
**S4 design:** Drop meta LoRA, keep S2 pre-caching, add multi-projection (q+v+o).
Expected: S2's 0.134 CE ratio with v+o should outperform S3's 0.151.
Centroid trap requires InfoNCE contrastive loss with hard negatives OR diverse
training domains — not tuning λ on cos² penalty.

## Key Numbers
- S2 CE ratio: 0.134 (q-only, pre-cached memory)
- S3 CE ratio: 0.151 (meta LoRA + q+v+o — worse than S2)
- q-only ratio: 1.160 (HURTS)
- q+v+o ratio: 0.151 (7.7x improvement over q-only)
- Centroid cos: 0.998 → 0.988 (marginal; diversity loss insufficient)

## References
- arXiv:2602.06358 (SHINE) — M2P architecture
- Finding #484 — S2 baseline (86.6% CE reduction, centroid trap)
- Finding #480 — v+o projections carry format priors
- Finding #485 — S3 KILLED (this experiment)
