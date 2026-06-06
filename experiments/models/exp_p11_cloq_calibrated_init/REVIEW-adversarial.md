# REVIEW-adversarial.md — P11.K0: CLoQ Calibrated LoRA Init

**Verdict: PROCEED**

## Math Assessment

### Theorem correctness ✓
The CLoQ initialization proof is algebraically correct. Verified step-by-step:

```
scale · lora_b.T @ lora_a.T
= scale · [U_r · diag(√(Σ_r/scale))] @ [diag(√(Σ_r/scale)) · V_r.T]
= scale · U_r · diag(Σ_r/scale) · V_r.T
= U_r · diag(Σ_r) · V_r.T = E_r  ✓
```

Eckart-Young optimal: E_r minimizes ||E - E_r||_F over all rank-r matrices. Applied correctly.

### Code math alignment ✓
`compute_cloq_init` implements the theorem faithfully:
- `lora_a = Vt[:r,:].T * sqrt_s[None,:]` — shape (n, r) ✓
- `lora_b = sqrt_s[:,None] * U[:,:r].T` — shape (r, m) ✓

### Adapter key format VERIFIED ✓
Checked against actual s1K adapter: mlx_lm uses `language_model.model.layers.{i}.self_attn.{proj}.lora_a`. CLoQ code uses identical format. `--resume-adapter-file` will load correctly.

### 8-bit proxy bound ✓
The (1/16) bound is a first-order estimate (2^4 / 2^8 ratio of quantization step sizes). Informal but directionally correct — 8-bit error is ≪ 4-bit error for the same weight range.

---

## Non-Blocking Issues

**Issue 1: K1535 baseline assumption**
The kill criteria table predicts "≥65% → ≥67%" — this assumes s1K lands at 65%. s1K result is still pending (pueue task 0). The code correctly handles this as a relative `delta >= 0.02` check, so the table entry is just a placeholder. Not a math error, but PAPER.md should report the actual s1K baseline once known.

**Issue 2: SVD energy capture prediction is informal**
The "≥70% energy in top-8 SVs" claim is based on group_size=64 covariance structure. This is plausible but not rigorously proven. The experiment measures it directly — if it falls below 70%, the low-rank approximation is weaker than claimed, but the math doesn't break. Treat as empirical prediction.

**Issue 3: Same-recipe comparison assumes s1K and CLoQ use identical hyperparams**
K1535 compares CLoQ-trained (this experiment) vs s1K (exp_p11_reasoning_sft_s1k). Both use rank=8, scale=1.0, lr=1e-5, 1000 steps, same data. PAPER.md must confirm hyperparameter parity when written.

---

## Failure Mode Analysis

**Primary failure mode (unchanged from design):** Gemma 4's 4-bit model is already near-float quality for reasoning tasks, so ||E||_F is small and the CLoQ correction is negligible in practice. In that case, K1535 fails (no +2pp) but K1536 and K1537 may still pass.

**Secondary failure mode:** s1K training doesn't converge (random init wanders far from E_r by step 1000), making the comparison noisy. Mitigated by same data/hyperparams.

---

## Conclusion

The mathematical foundation is sound. Eckart-Young is classical, the proof is tight, the code matches. Adapter key format verified against s1K output — no silent failure risk. Proceed with current design.

---

## Bug Fix Verification (2026-04-14, task 26 requeue)

**Fix**: `dequantize_layer_weight()` and `get_raw_linear_weight()` both now call `.astype(mx.float32)` before `np.array()`.

**Root cause**: MLX's 4-bit quantized model dequantizes to bfloat16. NumPy's PEP 3118 buffer protocol does not support bfloat16 (item size 2 mapped to 'B' format fails). Converting to float32 first bypasses this.

**Fix correctness**: ✓ The cast is lossless in the sense that bfloat16 → float32 expands mantissa precision; SVD is done in float32 anyway so no information is lost. The math is unchanged.

**PROCEED verdict unchanged.**

---

## Post-Run Review (2026-04-17)

**Verdict: KILL** (confirms researcher's proposed KILLED status)

### Adversarial checklist — all gates pass for KILL

| # | Check | Result |
|---|-------|--------|
| a | results.json (k1535=false, k1537=false) ↔ PAPER verdict KILLED | ✓ consistent |
| b | all_pass / kill_criteria: 2/3 fail | ✓ killed |
| c | PAPER.md verdict line: "**Verdict: KILLED**" | ✓ |
| d | is_smoke=false, n_steps=1000, N=260 MMLU-Pro items | ✓ full run |
| e | MATH.md KCs unchanged post-hoc (no git diff) | ✓ |
| f | No tautological passes | ✓ |
| g | K-IDs measure correct quantities (SVD energy, wall-clock, MMLU-Pro acc) | ✓ |
| h | No composition bug (not applicable) | ✓ |
| i | LORA_SCALE=1.0 (safe) | ✓ |
| j | Routing not applicable | ✓ |
| k | No `shutil.copy` adapter-as-new-domain | ✓ |
| l | Phase-skip hardcodes `k1536_pass=True, 0.033, 28.9s` when adapter exists — not load-bearing for KILL (K1536 is the passing criterion; the failing K1535/K1537 are freshly measured). Flagged, non-blocking. | ⚠ soft |
| m | Target: Gemma 4 4-bit loaded = MATH.md target | ✓ |
| m2 | No explicit `/mlx-dev` / `/fast-mlx` skill-invocation note in MATH/PAPER; non-blocking for KILL (no code being promoted). | ⚠ soft |
| n | Measured 33.8% with avg 1002 thinking chars — not empty-string pathology | ✓ |
| o | N=260 ≫ 15 | ✓ |
| p/q | s1K baseline absent; PAPER is honest (one-sided fail on absolute acc, not a bogus delta) | ✓ |
| r | Prediction-vs-measurement table present | ✓ |
| s | Math unchanged; Eckart-Young application still correct | ✓ |

### Load-bearing finding
Eckart-Young theorem is sound; its **operational precondition fails on Gemma 4 4-bit**:
top-8 SV capture = 3.3% of ||E||_F² vs predicted ≥70%. Error is spread across ~1400 modes.
Even rank-64 would capture only ~20%, so CLoQ is not rescuable on this model by increasing rank.

### Implications
- Do NOT resurrect CLoQ on Gemma 4 with higher rank.
- Reasoning-SFT capacity should route to data-quality / on-policy methods (LIMO, GRPO, ThinkPO).
- Open question (non-blocking, for future finding): why is Gemma 4 4-bit group-quant error non-low-rank? Likely group_size=64 + output-row decorrelation from post-training destroys the covariance structure assumed in CLoQ's motivating analysis.

### Route
→ `review.killed` — experiment complete, finding logged, handoff to Analyst.

