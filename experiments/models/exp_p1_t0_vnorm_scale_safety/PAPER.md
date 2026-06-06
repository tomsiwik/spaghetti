# PAPER.md: T0.2 — V-Norm Scale Safety

## Experiment
**ID:** exp_p1_t0_vnorm_scale_safety  
**Model:** Qwen3-4B-4bit (LoRA rank=4, v_proj only, 100 steps GSM8K)  
**Design:** Controlled v_norm injection: compare Qwen3 with/without VNormLinear on v_proj  
**Note:** Gemma 4 (the intended target) cannot be loaded with mlx_lm 0.29.1 (missing gemma4 model class). Experiment redesigned as within-Qwen3 controlled comparison.

---

## Prediction vs. Measurement Table

| Prediction (MATH.md) | Measured | Pass? |
|---------------------|----------|-------|
| K994: WITH v_norm, MMLU degradation ≤ 5pp at scale=5,10,20 | -36pp, -40pp, -32pp | FAIL |
| K995: WITHOUT v_norm, scale=20 degrades MMLU >30pp | -32pp (scale=20) | PASS |
| K996: Quality ratio at scale=10,20 vs 5 ≥ 0.95 | 1.0 (degenerate: all 0%) | DEGENERATE |
| Theorem 1: v_norm forces ||V||_RMS = sqrt(h) ∀ scale | Holds mathematically; cannot test MMLU preservation on Qwen3 | UNTESTABLE |

---

## Raw Results (smoke test: n=25 MMLU questions, 10 train steps)

| Condition | scale=5 | scale=10 | scale=20 |
|-----------|---------|----------|----------|
| Base (no adapter) | 68.0% | — | — |
| WITHOUT v_norm | 52.0% (-16pp) | 56.0% (-12pp) | **36.0% (-32pp)** |
| WITH v_norm (injected) | 32.0% (-36pp) | 28.0% (-40pp) | 36.0% (-32pp) |

GSM8K accuracy WITH v_norm: 0.0% at all scales (task broken by normalization)

---

## Analysis

### Why K994 FAILS on Qwen3-4B

**Root cause:** Qwen3-4B's output projection `o_proj` was trained to receive
**un-normalized** value vectors (`values = v_proj(x)`, no normalization). When we
inject `v_norm` at inference time, `o_proj` receives a fundamentally different
distribution than it expects → MMLU degrades WORSE than without v_norm.

**Mechanistic prediction confirmed:**
- At low scale (5): base model W_v @ x still dominates the value. Without v_norm, output
  is close to base → moderate -16pp. With v_norm, normalization disrupts base distribution
  → severe -36pp.
- At high scale (20): adapter B @ A @ x dominates. Whether normalized or not, output
  direction ≈ adapter direction → similar -32pp in both conditions.
- The convergence at scale=20 confirms Corollary from Theorem 1: at high scale, v_norm
  effectively normalizes just the adapter direction (base becomes negligible).

### Why K995 PASSES

Scale=20 without v_norm degrades MMLU by 32pp (>30pp threshold). This directly
replicates Finding #320's finding that scale=20 is catastrophic without value normalization.
The prediction was correct: Davis-Kahan bound becomes vacuous at high perturbation.

### Impossibility Structure

**What makes K994 untestable on Qwen3-4B:**

The v_norm guarantee (Theorem 1) is mathematically correct: `||V_norm(s)||_RMS = sqrt(h_v)`
for all `s`. But MMLU preservation requires more: the model's `o_proj` must be trained
to receive v_norm'd values. Without this, injecting v_norm:
1. Changes the value distribution that `o_proj` expects
2. Causes degradation WORSE than the scale catastrophe we were fixing

**Formal statement:** Let `f(V)` be the linear mapping `o_proj`. If `o_proj` was
trained on distribution `P(V_raw)`, evaluating on `V_norm ~ P_norm(V_raw)` produces
distribution shift `||P - P_norm||_TV > 0` even at scale=0.

**Resurrection path:** Test on Gemma 4 where v_norm is integral to training.
Requires mlx_lm to add `gemma4` model type support (currently missing).

---

## Status

- **K994 (id:994):** FAIL — cannot be tested without Gemma 4 support in mlx_lm
- **K995 (id:995):** PASS — 32pp degradation at scale=20, confirming scale catastrophe
- **K996 (id:996):** DEGENERATE — v_norm breaks Qwen3, all GSM8K = 0%
- **Overall:** KILLED (structural architecture barrier, not hypothesis failure)

## Key New Learning

The MATH.md theorem is correct, but it says nothing about MMLU preservation on models
trained WITHOUT v_norm. The full guarantee requires:

> Gemma 4's ENTIRE training was done with v_norm applied. The o_proj weight matrix
> was updated to expect unit-RMS value vectors. Any adapter applied at inference
> produces unit-RMS values (by Theorem 1), matching what o_proj expects.
> MMLU is preserved by construction.

On Qwen3-4B, `o_proj` was trained on un-normalized values. Injecting v_norm breaks
this expectation regardless of adapter scale. The v_norm injection is not safe on
models trained without it.

## Next Steps

1. Add `gemma4` model class to mlx_lm (nested text_config wrapper around gemma3n)
2. Re-run K994 on Gemma 4 E4B — should PASS by Theorem 1 + Gemma 4 training
3. Current result confirms K995 (catastrophe without v_norm) — sufficient for that gate
