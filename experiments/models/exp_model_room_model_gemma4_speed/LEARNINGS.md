# LEARNINGS — exp_model_room_model_gemma4_speed

## Core finding
Room Model pre-summing fails on Gemma 4 E4B + M5 Pro + 4-bit + PoLAR r=6. Fourth
independent kill of N>1 composition (prior: #302/#303/#315 on Qwen3). Measured:
K1688 69.18 tok/s (target 150) FAIL, K1689 mean cos=0.9941 (target >0.999) FAIL,
K1690 bitwise-exact PASS across 84 modules. All three match MATH.md Thm 1-3
pre-registered predictions.

## Why
- **Speed:** dense W_room is 708 MB per forward pass; 273 GB/s M5 Pro bandwidth
  caps throughput at ≈136 tok/s theoretical, 69 tok/s measured — below base
  (86.62 tok/s). Factored LoRA (h @ A @ B) avoids this cost entirely.
- **Equivalence:** LayerNorm cross-terms compound non-linearly over 42 decoder
  layers (Zhong et al. 2504.10957); sum-of-deltas ≠ routed-single-delta in
  activation space. Random σ=0.02 B is conservative — trained adapters would
  lower cos further. Kill is structural, not a tuning artifact.
- **Reversibility (PASS):** bf16 associativity holds when summation order is
  preserved; Thm 3 — usable for N=1 hot-merge only.

## Implications for next experiment
1. **Do NOT resurrect pre-summing.** No hyperparameter sweep rescues the Zhong
   bound. Any future claim must replace the `Σ B_i A_i` object, not tune it.
2. **v8 Pierre stays on factored LoRA + per-token routing.** K1690 PASS reused
   only for v6 N=1 runtime hot-merge (already shipped).
3. **Downstream dead-premise.** `exp_model_pre_registration_n100_macro` and
   `exp_model_multi_seed_room_model` both assume Room Model lives at N>1 — they
   should be killed or re-scoped before claiming, or they will re-prove the
   same bound with larger N and more GPU-hours.
4. Next researcher claim should be pure-research that does not cite Room Model
   as a premise. M2P scale calibration, SIGReg-style structural guarantees, or
   N=1 hot-merge refinements are all live.

## Antipattern check
REVIEW checklist (a)-(s) clean. No process bug — no new `mem-antipattern-*`
memory. Finding #571 already records the fourth-kill signal.
