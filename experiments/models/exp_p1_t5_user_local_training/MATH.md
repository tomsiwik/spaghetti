# MATH.md — T5.1: User Trains Personal Adapter (< 100 Examples)

## Experiment Type: Guided Exploration
Proven framework: LoRA low-rank fine-tuning (Hu et al. 2021).
Unknown: minimum iteration/sample count on M5 Pro for personal preference injection.

---

## Theorem 1: Low-Rank Sufficiency for Style Injection

**Claim:** LoRA with rank r=4 on q_proj is sufficient to inject a personal stylistic
preference into Gemma 4 E4B from n=50 training examples.

**Proof:**

Let W₀ ∈ ℝ^{d×d} be a frozen pre-trained weight matrix (d=2048 for Gemma 4 E4B q_proj).
LoRA restricts the weight update to:

    ΔW = B·A,   B ∈ ℝ^{d×r},  A ∈ ℝ^{r×d}

**Step 1 — Intrinsic dimensionality argument.**
Aghajanyan et al. (2020, arxiv 2012.13255) showed that fine-tuning tasks have intrinsic
dimension d_int << num_params. For simple stylistic tasks (e.g., adding a signature phrase
to every response), the preference is a rank-1 intervention in the output distribution:
the model learns to append a fixed token sequence regardless of context. This is captured
by LoRA with r=1; rank r=4 provides 4× capacity for richer preferences.

**Step 2 — Convergence with n=50.**
For a stylistic preference P that appears in every training example at the same position
(response suffix), the negative log-likelihood loss is:

    L(θ) = -Σᵢ log p_θ(y_i | x_i),   y_i = [content_i + suffix]

The suffix tokens are deterministic given the user's preference. With n=50 examples and
I=300 gradient steps, each example is seen on average 12 times (300 steps × batch_size=2
/ 50 examples = 12 passes). Standard SGD convergence (Bottou 2010) guarantees:

    E[L(θ_T)] ≤ L(θ*) + C/(η·T)

With η=1e-4 and T=300, the loss converges to near-optimum. The model assigns high
probability to the suffix after 12 passes of consistent signal.

**Step 3 — Routing independence (from T3.7 + T4.5).**
T4.5 (Finding #433) confirmed trained adapter A-matrices drift 0.579 from Grassmannian
init. However, T3.7 (Finding #430) proved hot-remove/add is bit-exact under exclusive
routing: once the personal adapter is loaded for a given user, no other adapter is active
concurrently. Zero interference by routing construction, not geometry.

**QED**

**Quantitative predictions:**
- Base compliance (no adapter): ~0% (phrase never naturally produced)
- Adapter compliance after training: ≥ 60% (prediction; ≥ 5pp = K1097)

---

## Theorem 2: Adapter Size Bound (Arithmetic)

**Claim:** A rank-4 LoRA adapter on q_proj across all Gemma 4 E4B attention layers
fits in < 10MB.

**Proof:**

Gemma 4 E4B dimensions (measured from existing T2.1 adapters, rank=6, 4.8MB):
- d_model ≈ 2560 (q_proj input/output dimension)
- num_attention_layers = 42

For rank r=4, targeting q_proj only:
- A matrix per layer: r × d = 4 × 2560 = 10,240 params
- B matrix per layer: d × r = 2560 × 4 = 10,240 params
- Per layer total: 20,480 params × 4 bytes (float32) = 81,920 bytes

Total adapter size:
  S = 42 layers × 81,920 bytes = 3,440,640 bytes ≈ 3.28MB

3.28MB << 10MB → K1098 PASS by arithmetic. □

**Using only 16 LoRA layers (last 16 for speed):**
  S₁₆ = 16 × 81,920 = 1,310,720 bytes ≈ 1.25MB << 10MB

Both configurations pass K1098.

---

## Theorem 3: Training Time Bound

**Claim:** Training 300 gradient steps (batch_size=2) on M5 Pro 48GB completes
in < 10 minutes including Gemma 4 E4B model loading.

**Proof sketch (empirical bounds from prior experiments):**

From T2.1 (1000 iters, all 42 layers, batch_size=2, seq_len≤512):
  - Observed runtime: ~60-90 min → ~3.6-5.4 s/iter at 42 layers, seq_len=512

For T5.1 (300 iters, 16 layers, batch_size=2, seq_len≤256):
  - Layer scaling: 16/42 ≈ 0.38× → ~1.4-2.0 s/iter
  - Sequence scaling: 256/512 = 0.5× for forward pass → ~0.7-1.0 s/iter
  - Predicted: 300 × 1.0s = 300s ≈ 5 min training
  - Model loading: ~60s (4-bit quantized 4B model)
  - Total: ~360-420s ≈ 6-7 min << 10 min

**QED** (subject to empirical verification)

---

## Kill Criteria Predictions

| ID | Criterion | Prediction | Basis |
|----|-----------|-----------|-------|
| K1096 | Training < 10 min on consumer hardware | PASS (~6-7 min) | Theorem 3 |
| K1097 | Personal adapter improves user task ≥ 5pp | PASS (~60-80pp gain, 0%→60%+) | Theorem 1 |
| K1098 | Adapter size < 10MB | PASS (~1.25-3.28MB) | Theorem 2 |
| K1099 | Script single-file, < 200 lines | PASS (~127 lines, measured) | By construction |

## References
- LoRA: Hu et al. 2021, arxiv 2106.09685 — rank-4 captures task structure
- Intrinsic dimensionality: Aghajanyan et al. 2020, arxiv 2012.13255
- T2.1 training rates, T3.7 routing invariance, T4.5 adapter format compatibility
- mlx_lm.lora: QLoRA by default when base model is 4-bit quantized

## Failure Modes
- K1097 FAIL: if 300 iters insufficient → increase to 500 (still < 10 min)
- K1096 FAIL: if model loading > 10 min (unlikely; measured ~60s in T4.3)
- Style injection harder than rank-1 → increase rank to 8 (adapter still < 10MB)

---

## V2 Revision (2026-04-18, `thinking-mode` audit fix)

Round-1 run (2026-04-10, 76pp gain) is retained as provenance but triggers
antipattern `mem-antipattern-008` (`type: fix`): `MAX_TOKENS=120` truncated
Gemma 4 E4B mid-`<|channel>thought...`, so base never reached any answer and
scored 0% for the wrong reason. The 76pp gain then conflates style injection
with thinking-suppression.

**Surgical fix (does not change the theorem, only the measurement):**
- `MAX_TOKENS` 120 → 4096 (antipattern-008 minimum: ≥4096).
- `split_thinking()` added to record `avg_thinking_chars` on the base run.
- K1097 now requires **both** `improvement_pp ≥ 5.0` **and**
  `base_avg_thinking_chars > 0` — if base never entered thinking mode the
  comparison is uninformative and K1097 fails regardless of pp-gain.

**Predicted V2 numbers (after fix):**
- Base compliance: still ~0% — the phrase "Hope that helps, friend!" remains
  vanishingly rare in the pre-training distribution regardless of token budget.
  Base will now emit a full answer (post-thought), so any 0% result is
  pure absence-of-marker, not truncation.
- Adapter compliance: ≥60% (training objective still applies; training data
  contained no `<|channel>` tokens so adapter should continue to emit the
  direct `answer + marker` format).
- Improvement: ≥50pp expected; K1097 threshold 5pp is easily cleared if the
  style-injection effect is real (as Theorem 1 predicts) rather than an
  artifact of truncation.

If after the fix `improvement_pp < 5pp` OR `base_avg_thinking_chars == 0`,
K1097 is FAIL — Theorem 1's low-rank style-injection claim is not supported
on this benchmark and a v2 experiment with a different preference marker is
required. This keeps the KC falsifiable under the corrected measurement.

Other KCs (K1096 training time, K1098 size, K1099 line count) are unaffected
by the MAX_TOKENS fix (fix is eval-only).
