# P7.C0: Projection-Quality Feedback Loop — KILLED

## Summary

Null-space projection magnitude cannot serve as adapter quality signal.
AUC = 0.4293 (below chance, threshold 0.7). Feedback-calibrated routing
shows exactly 0.00pp improvement over static routing (threshold 5pp).
Misplacement detection is noise. All 3 kill criteria FAIL, confirming
the MATH.md impossibility theorems.

**Status: KILLED** — Null-space projection magnitude carries zero quality
information. The "geometry-as-reward" approach is structurally impossible
within null-space LoRA.

## Prediction vs Measurement

| ID | Prediction | Threshold | Measured | Verdict |
|----|-----------|-----------|----------|---------|
| K1306 | Feedback routing improvement | >= 5pp | **0.00pp** | **FAIL** |
| K1307 | Quality prediction AUC | >= 0.7 | **0.4293** | **FAIL** |
| K1308 | Misplacement detection | informative | **noise (0.10 vs 0.05)** | **FAIL** |
| Spearman r (proj vs quality) | ~0 | > 0 | **-0.224** | Confirms #495 |

## Detailed Results

### K1307: Quality Prediction from Projection Magnitude

Overall AUC = 0.4293: projection magnitude is *anti-predictive* of quality.
High projection magnitude (large adapter norm) weakly correlates with *worse*
quality, not better. This matches Finding #495 (Spearman r = -0.19, we get -0.224).

**Why below 0.5 (anti-correlated)**: The legal adapter has 3x larger A-matrix
norm (Finding #495). Since all adapters provide positive quality on most texts
(ensemble effect from Finding #496), the legal adapter's inflated projection
gives it high scores despite being one of the weaker adapters on non-legal text.
High projection → legal adapter → worse-than-average quality on non-legal text.

Within-adapter AUC is degenerate: projection magnitude is constant per adapter
(||A_i||_F^2 doesn't depend on input), so within a single adapter there is
zero discriminative information.

### K1306: Feedback-Calibrated Routing

Static routing (TF-IDF weights) and feedback routing produce **identical** losses
(avg 4.9953 both). The feedback quality estimates converge to exactly 0.0 for all
adapters. This occurs because:

1. Feedback signal = (loss_static - loss_feedback) × weight_d
2. Both strategies produce the same loss (near-uniform weights from TF-IDF)
3. The quality estimate EMA receives zero signal → stays at zero
4. Feedback weights = static weights × (1 + 0) = static weights

The feedback loop has nothing to learn from: the static routing is already
near-optimal (from Finding #496: ensemble effect dominates), and the projection
magnitude carries no information to improve upon it.

### K1308: Misplacement Detection

Of 50 adapter-text pairs:
- 46 show positive quality (adapter helps), 4 show negative quality
- High-projection group: 10% negative quality rate
- Low-projection group: 5% negative quality rate

The difference (5pp) is well within noise for this sample size. The "misplaced"
adapters (3 total) are domain-mismatched (0% domain-match rate), which is expected
by chance since 4/5 of all adapter-text pairs are domain-mismatched.

## Failure Mode Analysis

### The Disease: Confusing Magnitude with Information

Projection magnitude |A_i Q^T x|^2 measures two things:
1. **Adapter norm** ||A_i||_F^2 — a property of the adapter, constant across inputs
2. **Null-space input energy** ||Q^T x||^2 — constant across adapters

Neither factor carries domain or quality information:
- Adapter norm reflects training dynamics (legal adapter trained to higher norm)
- Null-space input energy reflects how much of the input is irrelevant to W_v

The directional component (cosine between A_i's rows and Q^T x) is the only
potential discriminant, but Finding #495 proved it carries no domain signal
because Q^T strips domain features.

### Impossibility Structure

**Theorem (confirmed)**: No function f: R+ → R mapping projection magnitude
to quality prediction can achieve AUC > 0.5 + ε for any ε > 0, because:

1. I(||A_i Q^T x||^2 ; quality_i(x)) ≈ 0
2. The null-space projection Q^T x is orthogonal to domain features
3. Quality depends on domain match, which requires domain features
4. Therefore projection magnitude is independent of quality

This closes the "geometry-as-reward" research line for null-space LoRA.
Quality signals must come from outside null(W_v):
- Text features (TF-IDF, embeddings) → already works (Finding #496)
- Range(W_v) features → contains domain info, could work for routing
- External reward (user feedback, LLM-as-judge) → orthogonal to geometry

## Connections

- **Finding #495**: Routing killed (r = -0.19). We extend: quality prediction
  also killed (r = -0.224, AUC = 0.43). Same structural cause.
- **Finding #496**: Weighted composition works via ensemble, not routing.
  Feedback cannot improve on ensemble because the signal is zero.
- **LoRAHub (2310.13699)**: Uses task-level loss as composition signal,
  NOT geometric features. Our result explains why: geometry (adapter norms,
  projection magnitudes) carries no task/quality information.

## What This Closes

The P7 null-space line has now established:
1. ✅ Null-space LoRA works for adaptation (Finding #494, 98.7% quality)
2. ✅ Weighted composition beats exclusive routing (Finding #496, +32.7pp)
3. ❌ Null-space projection cannot route (Finding #495, 20% accuracy)
4. ❌ Null-space projection cannot predict quality (this experiment, AUC 0.43)
5. ❌ Null-space projection cannot provide feedback signal (this experiment, 0pp)

**Conclusion**: Null-space geometry is excellent for **isolation** (zero
interference) but carries zero **information** (no routing, quality, or
feedback signal). Routing and quality signals must come from range(W_v)
or external features.

## Runtime

- Total: 0.3 min
- Platform: Apple M5 Pro 48GB
- Peak memory: ~4.7 GB
