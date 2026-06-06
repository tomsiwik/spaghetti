# PAPER: SHINE Piece C — M2P Architecture Study

**Experiment:** shine_architecture_study
**Type:** Verification (Type 1)
**Status:** Pre-run (pueue task 6, queued)
**Date:** 2026-04-14

---

## Summary

Verifies that M2P Transformer (arXiv:2602.06358, §3.4) has no Qwen-specific components
at Gemma 4 E4B production dimensions (L=42, d_model=2560, r=16) and that parameter
count stays well below 1B. Two kill criteria (K806, K807) both expected to PASS.

---

## Prediction vs Measurement Table

| Prediction (MATH.md) | Predicted Value | Measured Value | Pass/Fail |
|----------------------|----------------|----------------|-----------|
| K806: Qwen-specific features | None (PASS) | TBD | TBD |
| K807: Parameter count | < 35M (PASS) | TBD | TBD |
| lora_A output shape | (42, 16, 2560) | TBD | TBD |
| lora_B output shape | (42, 2560, 16) | TBD | TBD |
| Forward latency (L=42) | < 100ms | TBD | TBD |
| Scale-agnostic (L=10,20,42) | Yes (PASS) | TBD | TBD |
| Actual param count (Theorem 2 corrected) | ~25M (not 31.5M) | TBD | TBD |

*Note: Theorem 2 in MATH.md predicts 31.5M but double-counts output projection by 4×.
LEARNINGS.md anticipates ~25M actual. Measured count is authoritative.*

---

## Kill Criteria

**K806** (architecture agnostic): PASS if no M2P component requires Qwen-specific imports
or config fields that don't map to generic (L, M, H) parameters. Expected to PASS — M2P
uses only standard row/column self-attention on geometric grid Z ∈ ℝ^{L×M×H}.

**K807** (param count): PASS if total parameters < 1B. Expected to PASS with ~25-35M.
Full implementation would be catastrophically over-budget above 1B; this is a sanity
check, not a borderline test.

---

## Experimental Design

Three phases:
1. **Production config** (L=42, M=32, H=256, n_m2p=4): instantiate, verify shapes, count params
2. **Scale ablation** (L=10, 20, 42): confirm shape-agnosticism — same code, different L
3. **Compact config** (H=128, n_layers=2, M=16): minimum viable M2P bound for ablation budget

---

## Caveats

1. **Param count discrepancy**: Theorem 2 double-counts the 4× multiplier in attention.
   Reported measured count is authoritative; MATH.md prediction is an upper bound.
2. **No forward-pass gradient**: This experiment only checks shapes, param counts, and
   forward-pass correctness (shape + finite norm) — not learning dynamics.
3. **C2 integration not verified here**: This experiment only establishes M2P generates
   correct shapes for PoLAR injection; joint training is a separate future experiment.

---

## Findings

*TBD — pending results.json*

---

## Connection to Vision

If K806 + K807 both PASS: C2 integration (PoLAR + M2P joint architecture) is unblocked.
M2P generates W_A (42, 16, 2560) and W_B (42, 2560, 16) directly compatible with PoLAR
Stiefel retraction. This closes the gap between session context → adapter generation.

Compact config result establishes minimum viable M2P budget (~2M with rank-factored head),
useful for ablation budget allocation in C2 experiment design.
