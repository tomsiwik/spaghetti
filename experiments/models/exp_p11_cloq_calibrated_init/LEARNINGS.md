# LEARNINGS.md — P11.K0: CLoQ Calibrated LoRA Initialization

**Status**: KILLED (2026-04-17). Finding #556 logged.

## Core Finding
CLoQ's theorem (Eckart-Young optimal low-rank init for W_float − W_4) is mathematically
sound but operationally inapplicable to Gemma 4 4-bit. Top-8 SVs captured **3.3%** of
||E||_F² (prediction: ≥70%). Quantization error is spread across ~1400 singular modes,
so rank-8 CLoQ collapses to near-standard-init. K1537 FAIL (33.8% MMLU-Pro vs 66% target);
K1535 FAIL (absolute accuracy below any plausible s1K baseline); K1536 PASS (28.9s).
2-of-3 → KILLED.

## Why
- **Theorem still holds.** E_r = U_r diag(Σ_r) V_r.T is Eckart-Young optimal. Verified.
- **Precondition fails on this model.** Group-size=64 quantization + Gemma 4's
  post-training decorrelates output rows, so the covariance structure CLoQ's motivating
  analysis assumes is destroyed. ||E||_F² energy is ~uniform across modes, not
  concentrated in a low-rank subspace.
- **Rank doesn't rescue it.** 3.3% at r=8 extrapolates to ~20% at r=64 — still looks
  like standard init. Scaling rank is not a fix.
- **Training made it worse, not just flat.** 33.8% is ~28pp below the ~62% Gemma 4 4-bit
  base (Finding #530). Likely rank-8 SFT on 1000 s1K examples overfits format and
  regresses MMLU-Pro content. Answer-extractor fragility (greedy first A-J after
  thought-strip) contributes but does not rescue the comparison (same extractor applies
  to the s1K companion).

## Implications for Next Experiment
- **Abandon CLoQ on Gemma 4 4-bit.** Not rank-32, not rank-64, not "more data". The
  error structure is wrong for the method, not the budget.
- **Route reasoning-SFT capacity to data-quality + on-policy methods.** LIMO, GRPO,
  ThinkPO. Don't debug initialization when the base is already near-float quality.
- **Before trusting any low-rank init method, measure top-r SV energy capture on the
  target model first** — cheap SVD probe avoids wasting a 1000-step SFT run when the
  precondition is already falsified. Add as pre-flight for future quantization-aware
  adapter experiments.
- **Rank-8 LoRA SFT on Gemma 4 can regress the base by ~28pp when init is bad.** Keep
  this in mind before comparing any two LoRA recipes at this rank: absolute numbers can
  be dominated by init quality, not recipe differences.
- **Open question (non-blocking).** Why non-low-rank error on Gemma 4 4-bit specifically?
  Likely group-quant + output-row decorrelation from post-training. Future finding
  candidate; not on critical path.
