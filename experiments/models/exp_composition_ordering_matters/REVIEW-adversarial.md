# REVIEW-adversarial.md — exp_composition_ordering_matters

## Verdict
**PROCEED**

## One-line reason
Both paired KCs pass by ≥100× margin; theorem-backed prediction confirmed at weight-space level and behavioral-PPL-level; verdict lines aligned across results.json / PAPER.md / claim.

## Adversarial checklist

| Check | Finding |
|---|---|
| (a) results.json verdict vs claim | `SUPPORTED` / `supported`. ✅ aligned |
| (b) all_pass vs claim | `all_pass=true`, both KCs pass. ✅ aligned |
| (c) PAPER.md verdict line | "SUPPORTED." No hedging. ✅ aligned |
| (d) is_smoke vs full-run claim | `is_smoke=false`, N=100 full. ✅ |
| (e) KC mutation in git | Experiment dir is untracked single-write; no post-hoc KC edit possible. ✅ |
| (f) Tautology sniff | K1928 = Frobenius(S_π − S_σ) / ‖S_π‖; K1975 = |PPL_π − PPL_σ|/PPL_avg across 6 real forward passes. Non-tautological per F#666 (proxy + target both measured). ✅ |
| (g) K-ID semantic match | K1928 code computes rel Frobenius gap across 6 perm sums per layer; K1975 code computes rel PPL gap across 6 permuted forwards. Matches MATH.md. ✅ |
| (h) Composition math | Uses `Σ_i (s/r) · B_i A_i` correctly per-layer (not `(ΣB)(ΣA)`). Verified at `run_experiment.py:118` and forward-path `:197-202`. ✅ |
| (i) LORA_SCALE | `ADAPTER_SCALE=6.0` (adapter's trained scale). Not inflated. ✅ |
| (j) Per-sample routing | N/A — no routing. ✅ |
| (k) shutil.copy as new adapter | N/A. ✅ |
| (l) Hardcoded `pass=True` | KCs computed from measurements. ✅ |
| (m) Proxy model substitution | Base = `mlx-community/gemma-4-e4b-it-4bit` matches MATH.md. ✅ |
| (m2) MLX skill invocation evidence | Code uses `mx.eval`, `mx.clear_cache`, `mx.random.seed`, proper dtype preservation, `model.eval()`, LoRALinear monkey-patch restored post-run. Idiomatic MLX throughout. ✅ |
| (n) Base-eval truncation | Teacher-forcing PPL, no thought-channel truncation. ✅ |
| (o) N<15 | N=100. ✅ |
| (p) Synthetic padding | All 3 adapters genuine (from `exp_p1_t2_single_domain_training`). ✅ |
| (q) Cited baseline drift | No cited baseline; all measurements in-run. ✅ |
| (t) Target-gated kill | K1928 (proxy) paired with K1975 (target PPL) per F#666; both passed → SUPPORTED. ✅ |
| (u) Scope-changing fixes | Full-scope run, no mid-run downgrade. ✅ |
| (r) Prediction-vs-measurement table | Present in PAPER.md with 4 rows (Frob rel/abs, PPL rel, equivalence class count). ✅ |
| (s) Math errors | Higham FP32 bound `(n-1)u·Σ|x_i|` cited correctly; factor-of-2 worst-case-over-orders is sound; BF16 intermediate-dtype explanation of the 1000× weight-vs-behavior gap is self-consistent (`N·u_bf16·‖term‖ ≈ 2·7.8e-3·‖term‖` matches measured 1.94e-3). ✅ |

## Non-blocking notes

1. **Cosmetic: variable naming drift.** `results.json` uses key `k1929_fire` and the code uses variable `k1929_fire`, but MATH.md defines the KC as **K1975**. The threshold (`0.01`) and semantic content are correct — it's just a variable rename that wasn't applied. Does not affect verdict consistency (PAPER.md table correctly labels K1975). Analyst may want to note this in LEARNINGS.md for future-hat searchability.
2. **Novel finding worth surfacing.** The *pattern* of 3 distinct PPLs for 6 permutations (swap-of-first-two-addends is bit-exact, re-associativity is not) is a **clean empirical fingerprint of left-fold GEMM ordering**. Worth a LEARNINGS.md line item as a diagnostic check for future composition experiments.
3. **Novel finding: 1000× weight-vs-behavior dtype gap.** Predicted FP32 PPL gap `~2e-6`; measured BF16 gap `~2e-3`. The bound *still holds* — it's just at BF16 unit-roundoff, not FP32. This is the first cross-scale measurement in this codebase of how intermediate-GEMM dtype propagates FP summation error into PPL. Forecloses a whole class of "maybe BF16 accumulation is catastrophic" concerns — it's bounded and well below 1% at N=3.
4. **Forecast for N=10 (Pierre macro).** Paper notes `(N-1)u_bf16` scales to `~6e-3` at N=10 — still below 1%. Good for Pierre composition plan. Follow-up experiment proposal P1 (N=5/N=10 replication) is reasonable but low-priority given the theorem.

## Assumptions logged
- Interpreted `k1929_fire` in code/results.json as alias for K1975 (thresholds match, semantics match). No verdict impact.
- No git history for MATH.md → treated pre-reg as first-write (pre-registered by default).
- Did not re-run the experiment; verdict based on artifacts on disk and math verification.
