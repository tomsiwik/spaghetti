# LEARNINGS: exp_model_thinking_preservation_training

DB: killed (smoke). PAPER: PROVISIONAL (recipe-level). CLI has no
`provisional`; killed is the conservative slot when all KCs fail at smoke.

## Core Finding

LoRA r=8, scale=1.0, N=20, LR=1e-5 on s1K-math (single domain) trained on
Gemma 4 E4B 4-bit **under-performs base by 33.4pp** on MMLU-Pro+thinking
(base 66.7% → adapt 33.3%, n=6). K1685 FAIL 33×. K1687 FAIL 2/3 cats.
K1686 INCONCLUSIVE — both base and adapter 0 think-chars is the
`mem-antipattern-008` parser/template footprint, not recipe falsification.

## Why

Smoke hits the F#538 trough: drifted off base, far from SFT-residual
equilibrium. MATH §3 (A3) `η·N·√r·max‖∇L‖ ≤ 0.5` holds only at convergence;
N=20 ≈ 1% of trajectory, val loss still descending (1.207→1.169). The
0-think-chars artifact: F#536 measured ≥2900 chars on the same base, so
the parser — not the model — is broken here.

## Implications for Next Experiment

Full-rerun blockers (do not re-run at N=20 — it's provably worst regime):
1. **Parser probe:** dump `response[:400]` raw, find real Gemma 4 E4B 4-bit
   `enable_thinking` delimiter; until fixed, K1686 is unmeasurable.
2. **3-domain A2 data:** s1K + `<think>`-augmented code + medical, ~500/domain.
3. **SFT-residual head (F#403):** `B_applied = B_sft + s·head(z)`, zero-init,
   custom MLX loop. Invoke `/mlx-dev` + `/fast-mlx` BEFORE writing (PLAN rule).
4. Full scale: N=1000, EVAL_PER_CAT=20, full MMLU-Pro.

## Antipattern

`mem-antipattern-008` triggered on K1686. Already catalogued — no new memory.
