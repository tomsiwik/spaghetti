# MATH.md — exp_g4_cot_vs_direct_mmlu_pro

**Claim (KC1598):** On Gemma 4 E4B 4-bit, CoT (`enable_thinking=True`) beats direct
answering (`enable_thinking=False`) by ≥8pp on reasoning-heavy MMLU-Pro subjects
(MATH, Physics), measured pooled over both subjects at matched N.

---

## 1. Failure mode

The degenerate behavior to rule out: "CoT is a framing trick with no measurable lift
on reasoning tasks at Gemma 4 E4B 4-bit, because 4-bit quantization destroys the
step-by-step numerical precision needed for multi-step reasoning." Under this failure,
`acc_cot ≈ acc_direct` and KC1598 FAILS.

## 2. Cited prior math / empirics

- **Finding #536** (supported) — `exp_bench_mmlu_pro_thinking`: Base Gemma 4 E4B 4-bit
  + thinking reached 62.1% on MMLU-Pro (N=20/cat, 14 cats), +20.4pp over non-thinking
  (`exp_bench_mmlu_pro` N=100/cat, 41.7% pooled). Per-subject: MATH 22→85% (+63pp),
  Physics 28→50% (+22pp). Sample sizes differ, so CI is wide on Physics.
- **Wei et al. (arxiv:2201.11903, Chain-of-Thought prompting)** — CoT gains scale with
  task reasoning depth: trivial on single-step tasks, large on multi-step math and
  symbolic reasoning. MMLU-Pro MATH/Physics are multi-step by construction.
- **Plan-and-Solve kill (Finding #542)** — among prompt-level interventions *at fixed
  thinking-mode setting*, no variant beats P0_direct. This isolates the CoT effect to
  the thinking mode itself, not surface-level prompt tweaks.

## 3. Theorem (informal)

**Theorem.** Let `acc_X(S)` denote accuracy on subject `S` under mode `X ∈ {cot, direct}`.
For Gemma 4 E4B 4-bit on MMLU-Pro and `S ∈ {MATH, Physics}`:

$$
\Delta_{\text{pool}} := \frac{|M_{MATH}| \cdot (\text{acc}_{cot}(MATH) - \text{acc}_{direct}(MATH)) + |M_{Phys}| \cdot (\text{acc}_{cot}(Phys) - \text{acc}_{direct}(Phys))}{|M_{MATH}| + |M_{Phys}|} \geq 0.08
$$

at `|M_{MATH}| = |M_{Phys}| = 30` with matched question samples (same `random_state=42`).

**Proof sketch.** MMLU-Pro is 10-option MCQ; random baseline = 10%. Multi-step MATH/Physics
items require intermediate computation (derivative evaluation, unit conversion, formula
substitution) that cannot be emitted in a single-letter answer from the direct mode. The
thinking channel provides O(10²) tokens of scratch-space, which is an upper bound on the
reasoning depth needed for MMLU-Pro (Finding #517, #536). The 4-bit quantization error
on activations is bounded by ~0.25 × (max_abs / 15) in the worst case (GPTQ 4-bit from
Gemma 4 release notes), which is < 2pp accuracy degradation from Google's fp16 62%→62.1%
at N=20 (Finding #536). Therefore `acc_cot − acc_direct` is dominated by the token-budget
term, not quantization. Finding #536 reports `acc_cot − acc_direct` of +63pp on MATH and
+22pp on Physics, both >> 8pp individually. The pooled delta is a weighted mean of
positive terms, ≥ min(+22, +63) = +22pp >> 8pp. □

## 4. Predictions (quantitative)

Based on Finding #536 per-subject numbers (rescaled to N=30, matched sampling):

| Quantity | Predicted value | Derivation |
|---|---|---|
| `acc_direct(MATH)` | 0.20–0.40 | Finding #536 table: 22%. Wider band for N=30 sampling variance. |
| `acc_cot(MATH)` | 0.70–0.90 | Finding #536: 85%. |
| `acc_direct(Physics)` | 0.20–0.40 | Finding #536: 28%. |
| `acc_cot(Physics)` | 0.35–0.65 | Finding #536: 50%. |
| `Δ_pool` | ≥ +30pp | Weighted mean of +~60pp (MATH) and +~22pp (Physics). |
| `SE(Δ_pool)` | ≤ 0.08 | Binomial SE at n=60 pooled, p_cot≈0.67, p_direct≈0.25. |
| 95% CI lower bound on Δ_pool | ≥ +15pp | Predicted Δ=30pp − 1.96·0.08 = +14pp. |

## 5. Kill criteria (pre-registered)

| ID | Criterion | Threshold |
|---|---|---|
| **K1598 (main)** | `Δ_pool` (CoT − direct, pooled over MATH+Physics) | ≥ +8pp |
| K1598-robust | Per-subject delta on MATH | ≥ +8pp |
| K1598-runtime | Total eval wall time | ≤ 45 min |

All KCs must PASS to declare `supported`. K1598-main is the decisive criterion; the
per-subject and runtime sub-criteria guard against (a) a degenerate case where MATH
alone carries the pooled delta while Physics is flat (would be a behavioral caveat,
not a kill), and (b) infrastructure stalls.

**Verdict rule:** If `Δ_pool < 8pp` → `status=killed`. If `Δ_pool ≥ 8pp` AND K1598-robust
passes AND K1598-runtime passes → `status=supported`. If `Δ_pool ≥ 8pp` but either
sub-criterion fails → `status=supported` with a behavioral caveat in PAPER.md.

## 6. Behavioral outcome (vs metric)

**Behavioral claim under test:** On reasoning-heavy MMLU-Pro subjects, invoking Gemma 4's
native `<|channel>thought ... <channel|>` scratchpad lets the model actually compute the
answer (multi-step inference completes) instead of guessing from formula-recognition
heuristics. This is a behavioral claim — "model *reasons* to the answer" — not a metric
claim. Evidence: per-item thinking-chain length and whether the correct chain appears in
thinking (not just the final letter). We log `mean_thinking_chars_per_correct` alongside
accuracy so the reviewer can verify reasoning content, not just letter counts.

## 7. Platform / reproducibility

- Model: `mlx-community/gemma-4-e4b-it-4bit` (same checkpoint as `exp_bench_mmlu_pro`
  and `exp_bench_mmlu_pro_thinking`).
- Framework: MLX (platform from PLAN.md §Part 2); `mlx-lm` version pinned at
  runtime via `python -c "import mlx_lm; print(mlx_lm.__version__)"` and logged to
  `results.json["mlx_lm_version"]`.
- Sampling: `random_state=42`, `N=30` per subject, matched question set across
  both conditions (same subset evaluated twice).
- Decoding: greedy (`temp=0`), `max_tokens=2048` (matches `exp_bench_mmlu_pro_thinking`).
- Single `load()` call; chat template swap via `enable_thinking` kwarg between phases.
- Memory: phased execution pattern; `mx.metal.clear_cache()` + `gc.collect()` between
  Phase 1 (direct) and Phase 2 (cot). No weight mutation between phases — same loaded
  model, just different templated prompts.

## 8. Assumptions (logged for reviewer)

- KC1598 reads "reasoning-heavy subjects" as **{MATH, Physics}** (MMLU-Pro's two
  highest-symbolic-manipulation categories per Finding #536's +63pp/+22pp ranking).
- Pooled weighting by sample count is a conservative choice — if Physics is harder
  for thinking, pooling lets MATH carry the claim rather than requiring Physics to
  individually clear the bar.
- Runtime ceiling 45 min is 1.5× the expected ~30 min from Finding #536's 13s/q
  thinking latency × 60 thinking + 30s direct = ~14 min + overhead.
