# MATH — Pierre v6: Precomputed Concatenated Deltas

> Retroactively authored 2026-04-17 from the experiment claim notes. No prior
> MATH.md existed when `run_experiment.py` and `results.json` were produced
> (git history: single commit `f421b73`, chore: research progress). This file
> records the pre-registered theorem + KCs as they were written in the DB
> claim notes on 2026-04-05, so the kill verdict can be read against them.

## 1. Thesis

Three algebraic transforms reduce per-token Metal kernel dispatches on a LoRA
side-path from 420 to 60, with **bit-exact equivalence** (within float
rounding) to the v3/v5.3 bf16 side-path:

1. **Attention-only** — skip MLP adaptation (LoRA paper: sufficient). 420 → 240.
2. **Precompute `ΔW = A @ B` offline** by matrix associativity. 240 → 120.
3. **QKV concatenation** — q_proj, k_proj, v_proj share post-layernorm input
   `x`; stack their ΔW into one matmul. 120 → 60.

## 2. Prior math

- Matrix associativity: `(x @ A) @ B = x @ (A @ B)`. No approximation.
- LoRA (arXiv:2106.09685): attention-only adaptation is parameter-efficient
  and typically matches full-module adaptation in task score.
- SPEED_RESEARCH.md: merge-in-place (v4/v5.1/v5.2) KILLED because ternary
  re-quantization destroys ΔW structure; side-path is the only valid route,
  so dispatch reduction is the only available speed lever.

## 3. Theorem (claimed pre-run)

**Claim.** For input `x ∈ ℝ^{B×T×d_model}`, precomputed ΔW concatenated across
q/k/v projections produces output identical to the v3/v5.3 bf16 side-path
within float rounding, using 60 Metal dispatches per forward pass.

**Proof sketch.** Each transform is an algebraic identity, no approximation:

- Attention-only: drops MLP ΔW terms by design. Choice, not math.
- Precompute: `(x @ A) @ B = x @ (A @ B)`; `ΔW = A @ B` is a fixed matrix once
  the adapter is loaded. One matmul replaces two.
- QKV concat: let `Δ_q, Δ_k, Δ_v ∈ ℝ^{d×d}`. Define `Δ_{qkv} = [Δ_q; Δ_k; Δ_v]
  ∈ ℝ^{d × 3d}`. Then `x @ Δ_{qkv} = concat(x@Δ_q, x@Δ_k, x@Δ_v)`. One fused
  matmul replaces three.

Per-layer dispatches: 2 (for post-layernorm attn output) × 30 layers = 60.

## 4. Predictions (from claim notes)

| Quantity | Predicted |
|---|---|
| Dispatches per forward | ~60 |
| Overhead vs native BitLinear | ~10% |
| Tokens/sec | ~126 |
| Behavioral score (overall) | ~0.41 (identical to v3) |
| Peak memory | <6 GB |

## 5. Pre-registered kill criteria (locked at claim time, DB IDs 742–744)

- **K742 (fail condition):** Speed < 100 tok/s.
- **K743 (fail condition):** Behavioral overall < 0.35 (must not degrade from
  v3's 0.41).
- **K744 (fail condition):** Memory peak > 6 GB.

All three derive from the proof's quantitative predictions (>100 tok/s required
to claim progress on the structural side-path bottleneck; <6 GB required to
fit 5-domain adapter bank plus base in 48 GB unified memory with headroom).

## 6. Antipattern self-check (2026-04-17 retroactive)

- **Tautological routing (Finding #553).** `run_experiment.py:150,:172` call
  `route(model, tok, val[d][0], W, MAX_SEQ)` — a single sample of the
  ground-truth domain `d` decides the adapter for **all** samples of that
  domain. Consequence: `v6_pierre[d] ≡ v6_single[d]` by construction whenever
  the router is ≥ chance, which the results confirm (0% degradation across
  all 5 domains; `ppl.v6_pierre == ppl.v6_single` byte-identically). This
  is the fix-category named in the DB tag `tautological-routing`.
- **LORA_SCALE = 20.0 (antipattern-003).** `run_experiment.py:44`. Inflated
  scale was a systemic claim-inflation lever in the audit; v6 inherits it
  from v5 copy-paste. Even in the bit-exact regime, scale 20 would amplify
  adapter noise and is not paper-grounded.
- **Non-rerunnable dependencies.** `pierre.v6` module (`pierre/v6/`) does not
  exist — `pierre/` contains only the flat `pierre.py` file. The SFT adapter
  directory `micro/models/bitnet_sft_generation_v3/sft_adapters/` is absent.
  The skeleton `micro/models/real_data_domain_experts/adapters/` is absent.
  Cannot apply a routing fix and re-measure; the recorded results are the
  only data available.

## 7. Verdict against KC (from recorded results)

| KC  | Predicted | Measured | Result |
|---|---|---|---|
| K742 | ≥ 100 tok/s | 86.8 tok/s | **FAIL** |
| K743 | ≥ 0.35 | 0.315 | **FAIL** |
| K744 | ≤ 6 GB | 2.23 GB | pass |

Two of three KCs fail on the recorded data. Independent of the routing-
tautology bug (K742 is speed-only; K743 is the overall mean where per-domain
routing happens to agree with ground truth in 4/5 cases), the experiment
falsifies its own claim that three algebraic transforms would push the
side-path past 100 tok/s at v3 behavioral quality.
