# LEARNINGS: exp_shine_gemma4_qa_benchmark (SHINE S4)

## Status: KILLED

## Core Finding
CE reduction (0.0804, 92%) is not document encoding — it is a universal LM shift to average English prose. QA F1 = 0.006 (0.6%) despite the best CE ratio in the SHINE series. CE and behavioral utility are orthogonal (Theorem 2 confirmed).

## Why
Reconstruction loss has no contrastive signal: it only minimizes CE(D_i | ΔW), never maximizes CE(D_j | ΔW) for j≠i. The global minimum IS the centroid LoRA. Centroid cosine = 0.9986 — all 7 documents produce identical LoRA weights. This is the same failure across all 4 SHINE stages.

## What Was Validated
- Multi-projection (q+v+o) from S3 confirmed useful — adapter generation is fast (0.133s, K1263 PASS)
- Proof-first method worked: Theorems 1–2 predicted F1 < 10%, measured 0.6%
- ICL (F1=0.196) functions as a baseline; the base model CAN answer with document in prompt

## What Was Killed
- SHINE pipeline without contrastive loss as a session adapter mechanism
- CE ratio as a proxy for document encoding quality (proven misleading)
- Meta LoRA (carried from S3 kill — confirmed not needed)

## Impossibility Structure
Without contrastive signal, the M2P loss landscape has a single global minimum at ΔW_centroid. No amount of architecture tuning (rank, layers, projections) can escape a unimodal loss surface. Only a structural change to the loss can break the centroid.

## Implications for Next Experiment
Fix requires InfoNCE contrastive loss (arXiv:1807.03748): explicitly require ΔW(D_i) to produce low CE on D_i AND high CE on D_j≠i. Also requires 100+ diverse documents (not 8 similar English prose passages). Without both changes, any SHINE variant will reproduce the centroid trap.

## References
- arXiv:2602.06358 (SHINE) — original architecture
- arXiv:1807.03748 (CPC/InfoNCE) — the fix
- Finding #486 — CE reduction ≠ document encoding
- Finding #345 — algebraic proof of centroid trap
