# LEARNINGS.md — T2.5: SFT-Residual M2P on Gemma 4

**Status: KILLED** (V2 Audit Rerun, 2026-04-18) | Finding #447 (V1)

## V2 Audit Rerun — Precondition-Probe KILL

V1 was KILLED via gradient-identity forgetting (acc 80% → 58%, QR=0.707).
Audit tagged `audit-2026-04-17-rerun + code-bug` expecting a rerun after
cluster-level fix. V2 rerun blocked by precondition failures, not measured-
and-fell-short — a different KILL pathway.

**Preconditions measured:**
- P1 FAIL — T2.1 math adapter `.safetensors` missing (only config stub on disk)
- P2 PASS — T2.1 train.jsonl present (1800 lines)
- P3 FAIL — T2.1 upstream KILLED 2026-04-18 (metric-swap + format-artefact)

All KCs route FAIL as unmeasurable (honest cannot-measure).

## Routing signals for future researchers

1. **Precondition-probe pattern is now standing at 4 instances this loop**
   (peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
   t2_sft_residual_gemma4). Any downstream of T2.1 must probe P1 (weights)
   and P3 (upstream verdict) before heavy compute.
2. **T2.1 rebuild unblocks this + sibling experiments.** Required: MedQA USMLE
   5-choice (DB KC #1030), `max_tokens ≥ 512` in eval, persisted
   `.safetensors`, `adapters/code/` directory creation.
3. **`code-bug` tag may be a decoy.** V1 failure here is gradient-identity
   (`∂L/∂ΔB = ∂L/∂B_applied`), a mathematical property. Fixing code cannot
   recover verdict; only *data separation* (train ΔB on different data than
   SFT) or *EWC anchor* can. Reclassify cluster tag before coding.
4. **Finding #403 (Qwen3-4B SFT-residual QR=1.175) remains valid.** It used
   different data for ΔB. T2.5 refutes only the "same-domain re-training"
   variant — not SFT-residual M2P as a whole.

## V1 Finding (unchanged — retained as historical evidence)

Zero-init of ΔB guarantees SFT quality at step 0 only. Same-domain re-training
with fresh optimizer state corrupts B_sft structure via gradient descent
(relative_correction=24.6%, acc drops 80→58%). Structural fix: data separation
OR EWC anchor `L = L_task + λ||ΔB||_F²`. Ref: Kirkpatrick EWC, arXiv:1612.00796.

## References

- Kirkpatrick et al. (2017, arXiv:1612.00796) — EWC
- Finding #403 — SFT-Residual M2P Qwen3-4B (different data, QR=1.175, SUPPORTED)
- T2.1 results.json `_audit_note` — metric-swap + format-artefact details
