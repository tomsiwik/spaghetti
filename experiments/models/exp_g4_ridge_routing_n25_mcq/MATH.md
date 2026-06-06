# MATH.md — exp_g4_ridge_routing_n25_mcq

## Claim
At N=25 MMLU subjects with disjoint train/test splits and hard-negative confusable pairs, a ridge classifier trained on mean-pooled Gemma 4 E2B last-hidden-state features attains per-sample test accuracy ≥ 90%.

## Notation
- Domain set: `D = {d_1, ..., d_25}` with 10 hard-negative pairs (see run_experiment.py).
- Per-domain splits: `X_d^train ∈ R^{n×d}`, `X_d^test ∈ R^{m×d}`, strictly disjoint (0% sample overlap).
- Feature extractor: `φ(x) = mean_t(text_model(tok(x)).norm_h_t)` where `text_model = model.language_model.model`, `norm_h_t` is the RMSNorm-normalized last-layer hidden state at token t. Dim `d = h_E2B` (Gemma 4 E2B hidden size from config).
- Ridge classifier: one-vs-rest ridge regression `W = argmin ‖Wφ − Y‖_F² + α‖W‖_F²`, closed form `W* = Y^T Φ (Φ^T Φ + αI)^{-1}`, predict `argmax_d W*φ(x)`.

## Theorem 1 (Ridge existence and optimality)
**Statement:** For any `α > 0` and Φ full or rank-deficient, the ridge solution exists uniquely and is the linear classifier minimizing Tikhonov-regularized squared error.
**Proof:** `Φ^T Φ + αI` is PD for `α>0` (eigenvalues ≥ α > 0), hence invertible. Optimality is standard KKT (see e.g. Hastie-Tibshirani-Friedman §3.4). QED.

## Theorem 2 (Structural upper bound on confusion)
**Statement:** Let `C_i = mean_x φ(x) | domain=i` (class centroid in feature space). Let `s_ij = cos(C_i, C_j)` be between-class cosine. Then ridge test accuracy satisfies the empirical upper bound:
```
P(argmax W*φ_test = y_test) ≥ 1 − sup_{ij} s_ij − O(1/√n)
```
when `α` is tuned to balance bias-variance.
**Proof sketch:** Ridge with small α approaches least-squares; as `α→∞`, W → class-centroid dot-product. For moderate α, the decision boundary between i,j is `(C_i − C_j)^T φ + b_ij = 0`, which misclassifies `φ_test^j` only if `(C_i − C_j)^T φ_test^j > b_ij`. Probability bounded by within-class variance and `s_ij`; see [Bishop PRML §4.1.5]. The `O(1/√n)` is Hoeffding on centroid estimation. QED (sketch).

## Theorem 3 (Gemma 4 hidden states separate N=25 MMLU subjects)
**Statement (empirical, conditioned on priors):** `sup_{ij} s_ij` on Gemma 4 E2B hidden-state centroids for the 25-subject set is bounded above by the TF-IDF centroid upper bound (which produced 84.2% at N=25 per F#502), because transformer hidden states are richer than lexical features. Specifically, F#310 reports 98.3% token accuracy at N=5 with hidden states vs 96.6% for TF-IDF — a +1.7pp gain that, extrapolated under F#502's N=25/N=5 scaling law (84.2%/96.0% = 0.877 ratio), predicts N=25 hidden-state accuracy `≈ 0.877 × 98.3% ≈ 86.2%` at minimum, with likely upward bias since hidden-state features encode semantic similarity more tightly than lexical overlap.
**Upper-side prediction:** Accuracy in [86%, 98%] range. K1616 threshold (≥90%) is the median of this predicted interval.

## Predictions (bind to code)
1. **P1 (K1616, primary):** Test accuracy ≥ 90% at N=25 with disjoint splits + hard negatives.
2. **P2:** Worst-domain accuracy ≥ 60% (no catastrophic failure).
3. **P3:** Train time ≤ 60 s for ridge fit at N=25, hidden_dim ≤ 4096.
4. **P4:** Per-sample inference latency ≤ 10 ms (excluding Gemma 4 forward pass).
5. **P5:** Hidden-state features outperform TF-IDF (if measured) by ≥ 3 pp (a-posteriori validation, not kill condition).

## Kill criteria (pre-registered, DO NOT EDIT POST-HOC)
- **K1616 (primary):** ridge test acc < 90% → FAIL.
- No secondary KC; P2-P5 are quality checks, not kill conditions.

## Antipattern self-check
- **ap-017 stub-cascade:** N/A — experiment does not load any LoRA adapter. Only Gemma 4 base + ridge classifier.
- **ap-020 cascade-dependent:** N/A — `depends_on: []`; no sibling/parent to cascade from.
- **ap-tautological-routing (F#502 closure):** Addressed by (a) disjoint train/test splits per-subject, (b) including 10 hard-negative subject pairs (medical↔{clinical_knowledge,anatomy,virology}, etc.), (c) no synthetic centroid shortcut — features come from real Gemma 4 forward passes on MMLU questions.
- **ap-no-knowledge-gap (F#478 closure):** N/A — this is a routing experiment, not a LoRA training experiment. F#478 closure applies to LoRA quality gain on MMLU-Pro, not to hidden-state classifier accuracy.
- **ap-smoke-reported-as-full:** Run is at SMOKE scale (smaller than F#502 which used N_TRAIN=200/N_TEST=80). This run uses N_TRAIN=100/N_TEST=40 per domain. PAPER.md declares this explicitly; verdict downgraded to `provisional` if thresholds passed (IS_SMOKE=true). Use `experiment complete --status provisional` unless IS_SMOKE=false.
- **ap-hardcoded-pass:** K1616 gated on actual measured accuracy; no hardcoded booleans.

## References
- **F#310:** Ridge on hidden states = 98.3% at N=5 (linear separability confirmed).
- **F#458:** TF-IDF ridge = 98.8% at N=25 — SUPPORTED but tautological (synthetic centroids, no disjoint test). This experiment is the non-tautological counterpart.
- **F#502:** TF-IDF ridge = 84.2% at N=25 with disjoint + hard-negatives. Below 90% — this experiment tests whether hidden-state features close the gap.
- MixLoRA (arxiv:2312.09979): learned routing over LoRA experts; ridge is the closed-form baseline.
- Hastie-Tibshirani-Friedman, *Elements of Statistical Learning* §3.4 (ridge regression).

## Assumptions (logged for transparency)
- **A1 (dataset):** "MMLU-Pro subjects" in DB title is interpreted as "MMLU subjects" per F#502's methodology (MMLU-Pro has 14 disciplines; cannot yield N=25). Logged so future readers can contest.
- **A2 (model size):** Gemma 4 E2B 4-bit chosen for speed; E4B results would likely be similar or higher.
- **A3 (feature):** mean-pool over all tokens (no attention mask weighting). Standard baseline.
- **A4 (α):** swept over `[0.01, 0.1, 1.0, 10.0]`; best on validation (held out from train).

## What falsifies the theorem
Theorem 3 falsifies if:
- Measured N=25 test accuracy < 86% (below the lower bound of the predicted interval).
- Worst-domain accuracy < 30% (indicates systematic feature collapse on that domain).
- Accuracy does not improve vs TF-IDF baseline at same N=25 methodology (would invalidate the "hidden states > lexical" prior).
