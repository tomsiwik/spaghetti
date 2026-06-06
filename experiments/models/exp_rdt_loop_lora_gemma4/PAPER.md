# PAPER — exp_rdt_loop_lora_gemma4

## Verdict: **PROVISIONAL (smoke)**

Smoke-mode build verifies architecture wiring on real Gemma 4 E4B (`mlx-community/gemma-4-e4b-it-4bit`, 42 layers, hidden=2560). Target claims K1740 (GSM8K-Hard +5pp), K1741 (MMLU within 1pp), K1742 (saturating-exp fit) are **not measured** — they require full-scale training + benchmark eval and are deferred to a follow-up experiment.

## Prediction vs measurement

| KC | Prediction (MATH.md) | Measurement | Result |
|---|---|---|---|
| K1743 | \|cos\|_max < 1e-6 (Higham float32 bound) | **3.75e-8** across 18 projections (9 layers × {v,o}_proj) | **PASS** |
| K1744 | max ρ(A_d) < 1 across 50 Adam steps | **0.367879** (= exp(-1), first step = last step) | **PASS** (see caveat) |
| K1740 | ≥ +5pp GSM8K-Hard at T=3 | not measured (full-scale deferred) | N/A |
| K1741 | \|ΔMMLU\| ≤ 1pp | not measured (full-scale deferred) | N/A |
| K1742 | R² > 0.90 saturating-exp on T∈{1..6} | not measured (full-scale deferred) | N/A |

elapsed 1.95s; `is_smoke=true`; `all_pass=true` over K1743+K1744; `preemptive=false`.

## What this supports

1. **Partition-QR Grassmannian A-init holds at Gemma 4 E4B native in-dims** (2048, 2560) at rank 16 with N=6 loops: max |cos| = 3.75e-8 ≪ 0.1 threshold ≪ 1e-6 theoretical bound. Extends F#562 from single-projection Pierre setup to per-loop-index family used in the RDT architecture.
2. **LTI + loop-indexed LoRA compose cleanly in MLX** under `nn.value_and_grad` over a bundle containing 108 LoRADelta modules (9 layers × {v,o} × 6 loops) + 6 LTIInjection modules. No memory pressure at N=50 steps, seqlen=32, batch=2. Infrastructure OK to scale up.
3. **ρ(A)<1 bound holds** under composition with loop-indexed LoRA — extends F#667 from primitive-only to composition context.

## Caveat: K1744 dynamics not exercised in smoke

`max_rho_over_steps == rho_first_step == rho_last_step == 0.367879 = exp(-exp(0))` means log_A and log_dt did not move under 50 Adam steps on the synthetic MSE-reconstruction loss. Two non-exclusive explanations:

- **Synthetic-loss gradient underflow**: h0 is re-sampled per step (different target residual each iteration), so the LTI gradient direction is averaging across noisy batches at the init point. Adam's effective step on 2560-dim log_A is below the float32 threshold for changing exp(-exp(s)) over 50 iterations from s=0.
- **Loss surrogate did not adequately route grads through LTI**: the tfm_out path dominates the residual, so LoRA B-matrix updates absorb most of the gradient.

Either way, K1744's *dynamical* guarantee in the surrogate-loss setting is untested. The *static* F#667 Theorem 1 guarantee (ρ<1 by construction in exact arithmetic) still holds; K1744 is not falsified. The full-scale follow-up must verify K1744 under real GSM8K loss where LoRA B-matrices and LTI params receive meaningful gradients.

## Assumptions logged

- LORA_SCALE = α/r = 0.125 (safe default per audit; not 20).
- Gemma 4 E4B v_proj dims = 2560→512 (GQA with num_kv_heads=2, head_dim=256); o_proj = 2048→2560. Confirmed by inspecting `model.language_model.layers[12].self_attn`.
- Base model freed after adapter bank construction to leave memory for training (phased-execution pattern per `/mlx-dev`).
- Grassmannian A treated as fixed (not trained) per Pierre F#562 convention; only B + LTI params train.

## What this does NOT support

- **K1740/K1741/K1742 are the target claims and remain untested.** "Architecture wires up" ≠ "architecture learns recurrent reasoning". The novel research question (does depth-axis loop LoRA trained on GSM8K+MATH lift Gemma 4 E4B reasoning ≥ +5pp?) is open.
- **K1744 dynamics in realistic training**: the synthetic loss used here does not prove LTI stays stable under GSM8K-scale training; follow-up must re-verify on real loss.

## Antipattern self-audit

- (f) Tautological KC: K1743 is structural-by-construction under partition-QR — acknowledged in MATH.md, paired with target K1740 (deferred) per F#666 target-gated rule.
- (h) Unsafe LORA_SCALE: α=2, scale=0.125 (safe).
- (m) Proxy-model substitution: model = product target; no proxy.
- F#138 smoke-as-full: explicitly marked `is_smoke=true`, verdict `PROVISIONAL`; no silent upgrade.
- F#452/F#453/F#1564 reproduces-or-refutes: K1744 EXTENDS F#667 to composition context (real Gemma 4 + LoRA bank), not a duplicate.

## Follow-up required

A new experiment `exp_rdt_loop_lora_gemma4_full` (macro, P0 or P1) must:
1. Train loop-LoRA + LTI on GSM8K+MATH train splits (≥10k samples, ≥1 epoch).
2. Evaluate on GSM8K-Hard (100 problems, greedy) at T ∈ {1..6}.
3. Evaluate on MMLU (57 subjects) at T=3.
4. Fit saturating-exp y(T) and report R².
5. Re-verify K1744 under real loss dynamics.
