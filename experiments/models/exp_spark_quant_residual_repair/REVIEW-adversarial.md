# REVIEW (adversarial) — exp_spark_quant_residual_repair  →  PROCEED (killed)

Verdict under review: KILLED (K2301). LoRA q_proj delta ΔW=(A@B)ᵀ orthogonal to 4-bit quant
residual R; math cos̄=−4.66e-5 vs 2×p95=1.63e-3 (ratio −0.057). I tried to break it; it holds.

## Cardinal-sin checks — all clear
- verify-experiment.sh exits 0. results.json present, is_smoke:false, verdict killed.
- Real models on disk: gemma-4-e4b-it-4bit AND -8bit both cached. The true fp16 google/gemma-4-e4b-it
  is a 12K metadata stub (no weights) — researcher's "gated/absent" claim is TRUE, proxy is justified.
- Adapters distinct (sha1 d71b428/89fade96/57aa04f3) — no shutil.copy sibling-adapter trick.
- Model in MATH.md == model loaded (MODEL_4BIT/MODEL_8BIT literals match).

## Integrity
- MATH.md is untracked (new experiment) ⇒ kill threshold K2301 cannot have been moved post-run via git.
- Code measures what MATH claims: cos(flatten((A@B)ᵀ), flatten(W8−W4)) per layer, vs norm-matched
  rank-6 null. Shape verified live: lora_a(2560,6)@lora_b(6,2048) → (A@B)ᵀ=(2048,2560)=W orientation;
  R same shape. No transpose/orientation trap that would trivially zero the cosine.
- Not tautological: null is non-degenerate (p95 8.1e-4, p50 2.7e-4) and lands on the predicted O(1/√D)
  floor (1/√5.24M=4.4e-4), proving the cosine machinery is live, not a hardcoded 0.

## The 8-bit-as-fp proxy CANNOT manufacture the kill (the one I was told to scrutinize)
Proxy error biases the result TOWARD the null, i.e. toward a kill — but only by ~6% direction error
(8-bit step 16× finer than 4-bit). The observed gap to threshold is ~100× (ratio 0.057 vs required 2.0).
A perfect fp16 reference would have to convert a −0.057 ratio into ≥2.0 — a 35× lift — which 6% reference
noise physically cannot do. Crucially, real |cos̄| (3.5e-4) sits AT/BELOW the null's own p50 (2.7e-4):
the signal is absent, not a real signal being erased by the proxy. The proxy's worst-case effect is to
make a marginal positive look null; here there is nothing marginal to erase. Kill is conservative & safe.

## Consistency
results.json verdict=killed, all_pass=false, is_smoke=false; PAPER verdict line KILLED; per-adapter
math/medical/python all fail identically (ratios −0.057/−0.075/+0.035). Triple corroboration.

## Evidence quality
Weight-space, not behavioral — but the question IS a weight-space geometric claim ("ΔW repairs R"),
so cosine in flattened weight space is the directly-meaningful signal, not a proxy for behavior. Sound.

CONCLUSION: PROCEED. Hypothesis refuted by 2–3 orders of magnitude on the registered metric.
