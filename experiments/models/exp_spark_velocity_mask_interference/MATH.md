# MATH — exp_spark_velocity_mask_interference

## Disease being prevented
Composing a **thinking** adapter (OpenThoughts-universal, targets `self_attn.v_proj` + `o_proj`, rank 8,
scale 1.0) with a **domain** adapter (`math`, targets `self_attn.q_proj`, rank 6, scale 6.0) on top of the
frozen `gemma-4-e4b-it-4bit` is hypothesized to suffer **destructive interference** that lowers GSM8K
accuracy below the math-solo ceiling. The repo has killed adapter composition treated as a *whole-adapter*
operation. The novel claim: the interference is **localized in the thinking adapter's LATE-moving weights**
— entries of the effective weight delta `ΔW = A@B` that only reach their final magnitude *after* training
step 200. The early-velocity **core** (entries already at ≥80% of final magnitude by step 200, with matching
sign) composes additively and carries the useful reasoning structure; the late residual carries the damage.

## Objects (all real, frozen)
- Base: `mlx-community/gemma-4-e4b-it-4bit` (frozen, 4-bit).
- Thinking trajectory: `data/adapters/thinking-openthoughts-universal-v0/` checkpoints `0000200` and `0001000`.
  Per projection p ∈ {v_proj, o_proj} and layer ℓ, define the effective delta at step t:
  `ΔW_t = A_t @ B_t`  (shape in×out).
- Math endpoint: `data/adapters/math/adapters.safetensors` (q_proj, rank 6).

## Velocity mask (the construction)
Let `m_t = |ΔW_t|` (elementwise). The **early-velocity-core mask** is
```
M = 1[ m_200 ≥ 0.80 · m_1000 ]  ∧  1[ sign(ΔW_200) == sign(ΔW_1000) ]
```
Measured global core fraction over all v_proj+o_proj layers: **0.2335** (≈23%) — the mask is non-trivial
(neither all-pass nor all-fail), so the split has real discriminative power. The masked thinking delta is the
dense `ΔW_core = M ⊙ ΔW_1000`; the late residual is `ΔW_late = (1−M) ⊙ ΔW_1000`.

## Composition (per method.md)
Two adapters on **disjoint** projections (thinking on v/o_proj, math on q_proj), applied as independent
additive deltas `Σ_i (B_i A_i)` — never `(ΣB)(ΣA)`. Math keeps its low-rank form at scale 6.0 (≤8 guard OK).
Thinking is applied either as low-rank `x@A@B` (full) or as the dense masked add `x @ ΔW_core` (core / late).

## Four conditions, GSM8K n=50, greedy, real answer extraction
- **A** math-solo (q_proj only) — the domain ceiling, no thinking.
- **B** math + full-thinking (step-1000 low-rank, scale 1.0) — the interference candidate.
- **C** math + early-velocity-core (dense `ΔW_core`, scale 1.0) — the claim.
- **D** math + late-residual-only (dense `ΔW_late`, scale 1.0) — control: damage should live here.

## Prediction (pre-registered)
1. `acc(C) − acc(B) ≥ +6pp` — the early core recovers the composition lost by the full adapter.
2. `loss_late > loss_early`, operationalized as `acc(C) > acc(D)` — the late residual is the damaging part.

## Numeric refutation threshold (KILL 2298)
KILL if **`acc(C) − acc(B) < +6pp`** OR **`acc(C) ≤ acc(D)`** (i.e. late residual is not the worse half).
If the data crosses this, verdict = `killed`; the goalpost is fixed before the run.

## Verdict mapping
- `supported`: `acc(C)−acc(B) ≥ +6pp` AND `acc(C) > acc(D)`.
- `killed`: either clause fails.
- `is_smoke:false` always (real model, real adapters, real GSM8K execution).
