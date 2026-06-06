# LEARNINGS.md — exp_condition_number_per_layer

**Status:** KILLED (K943 fires: mean κ = 18,130 >> 200)

## Core Finding

4-bit quantization creates near-degenerate KV projection matrices in Qwen3-0.6B:
k_proj (mean κ = 56,013) and v_proj (mean κ = 16,445) are UNSAFE, while rectangular
matrices (q_proj κ = 44, o_proj κ = 21, MLP κ < 100) remain safe.

## Why

Square GQA matrices (1024×1024) acquire near-zero singular values under 4-bit/group-64
quantization — small dynamic-range groups collapse to near-zero, driving σ_min → 0.
Rectangular matrices are protected by their overparameterized structure (σ_min bounded).

## Critical Bypass (Why M2P Survives K943)

M2P targets q_proj (κ = 44, SAFE). Grassmannian A-matrices are initialized from the
top-k singular vectors of the base weight W. M2P's signal path therefore travels
exclusively through the high-σ subspace, making effective κ_M2P << κ_full.
The degenerate low-σ directions (where κ explodes) are never accessed.

Reference: Aghajanyan et al. arXiv 2012.13255 — fine-tuned adapters align with
top singular directions of base weights.

## Implications for Next Experiment

**exp_m2p_a_matrix_alignment**: Verify M2P A-matrices cos(A, U_top) > 0.9 for
q_proj and v_proj. If confirmed, K943 KILL does not block M2P promotion — the
per-experiment effective κ is what governs error propagation, not the global matrix κ.
