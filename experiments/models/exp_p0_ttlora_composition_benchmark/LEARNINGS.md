# LEARNINGS: exp_p0_ttlora_composition_benchmark

## Core Finding
Pre-merge composition fails identically for TT-LoRA (norm ~3500) and standard LoRA (norm ~4), despite a 737x magnitude difference. The disease is perturbation **direction** (non-orthogonal subspace overlap), not magnitude.

## Why
Summing independently-trained adapters adds N-1 cross-domain perturbations per query. These cross-terms corrupt attention regardless of scale — even ε→0 keeps directions identical. Structural orthogonality (PoLAR/Grassmannian) is the only fix. TT-LoRA's tensor contraction structure also amplifies core norms multiplicatively (not additively), making Theorem 1's √P scaling assumption catastrophically wrong (predicted 0.21, measured 737).

## Impossibility Structure
W_merged = W_base + ΣᵢΔWᵢ fails because each ΔWᵢ was trained to correct W_base for domain i, not for W_base + Σⱼ≠ᵢΔWⱼ. No compression scheme fixes this — only orthogonal subspace training does.

## Implications for Next Experiment
Per-query routing (K1450 PASS, 0pp delta) completely avoids interference. The next experiment should test PoLAR/Grassmannian orthogonal adapter training to validate that structural orthogonality enables safe pre-merge. Alternatively, investigate whether the Room Model (W_combined = Σ ΔWᵢ with routing-as-matmul) sidesteps this by never summing adapters statically.

## References
- Finding #526 (KILLED): TT-LoRA pre-merge identical failure to std LoRA
- Finding #510: Original pre-merge failure diagnosis
- PoLAR/Grassmannian: structural orthogonality approach
