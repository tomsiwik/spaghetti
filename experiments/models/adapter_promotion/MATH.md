# MATH: Adapter Promotion via NRE Composition

## Experiment Type: Guided Exploration

**Proven framework:** NRE Norm Preservation (Finding #275)
**Unknown:** Does a "universal" adapter (medical) retain ≥70% of its standalone benefit after NRE composition with 4 other domain adapters?

---

## I. Prior Results

**Finding #275 (conclusive):** NRE (Norm-Rescaled Euclidean averaging) matches Fisher-Rao Karcher mean. The mechanism: `result = mean(ΔW_i) * mean(‖ΔW_i‖_F) / ‖mean(ΔW_i)‖_F`. This prevents 1/√N norm shrinkage from naive averaging.

**Finding #126 (conclusive):** Structural orthogonality — adapters trained on different domains have cosine similarity 17–69× below the Welch bound √(r/d). For our d=896 architecture, cos ≪ 0.03.

**Finding #330 (supported):** At scale=13, N=5 NRE composition achieves −4pp MMLU degradation (near-lossless). At scale=20, domain-specific PPL improves but OOD degrades (Finding #328).

---

## II. NRE Retention Lemma

**Setup:** N adapters with B-matrices {ΔW_i ∈ ℝ^{r×d}}_{i=1}^N, Frobenius norms {σ_i}.

**NRE composition formula:**
```
ΔW_composed = mean(ΔW_i) · (σ_mean / ‖mean(ΔW_i)‖_F)
```
where σ_mean = (1/N) Σ σ_i.

**Retention factor derivation for near-orthogonal adapters:**

For unit vectors with cos(ΔW_i, ΔW_j) ≈ 0 (Finding #126):
```
‖(1/N) Σ ΔW_i‖_F ≈ σ_mean / √N      (by orthogonality)
```

Therefore the NRE result magnitude is:
```
‖ΔW_composed‖_F = σ_mean · (σ_mean / ‖mean‖) ≈ σ_mean · √N
```

The projection of ΔW_composed onto adapter i's direction (ê_i = ΔW_i/‖ΔW_i‖):
```
⟨ΔW_composed, ê_i⟩ ≈ (σ_i / N) · (σ_mean / ‖mean‖) ≈ σ_i / √N
```

**Standalone contribution:** ⟨ΔW_i, ê_i⟩ = σ_i

**Directional retention ratio:**
```
η_i = σ_i/√N / σ_i = 1/√N
```

For N=5: η = 1/√5 ≈ 0.447 (45% directional retention under ideal orthogonality).

**Norm retention:** NRE rescales the total to σ_mean·√N ≈ σ_medical (if σ_medical ≈ σ_mean), so the overall magnitude is preserved, but spread across N directions.

---

## III. Predictions

The PPL benefit of an adapter is approximately (first-order):
```
ΔL_i ≈ -⟨∇_W L_i(W_base), ΔW⟩
```

Under NRE composition, each adapter's directional contribution is η_i ≈ 1/√N = 0.447 at N=5, **if adapters are perfectly orthogonal and equal-norm.**

However, the medical adapter is described as "most universal" — it likely has:
1. Higher norm than average (promotes its contribution above 1/√N)
2. Non-zero projections onto other adapter directions (shared "universal" component)

**Unknown parameter:** The actual retention η_medical ∈ (0, 1). The pure orthogonality lower bound is 1/√5 ≈ 0.45.

### Prediction Table

| Prediction | Theoretical Basis | Expected Value | Kill Threshold |
|------------|-------------------|----------------|----------------|
| P1: Medical PPL retention under N=5 composition | NRE orthogonality bound | η ≥ 0.45 | K828: ≥70% |
| P2: No other domain catastrophic degradation | NRE norm preservation | All domains ≤ 1.5× base | K829: all domains ≤ 1.5× |
| P3: Medical is highest-norm adapter | "Most universal" hypothesis | σ_medical / σ_mean > 1.0 | No threshold; informative |

**Critical uncertainty:** K828 requires 70% retention. The theoretical lower bound (pure orthogonality) predicts only 45%. The gap (45% → 70%) must come from:
- (a) Medical adapter having higher norm (σ_medical > σ_mean), or
- (b) Non-orthogonal overlap: medical shares subspace with other adapters

The experiment measures which of these holds.

---

## IV. Kill Criteria (Derived)

**K828 (PASS threshold: 70%):**
```
retained_benefit = (base_ppl - composed_ppl) / (base_ppl - solo_ppl) ≥ 0.70
```
This threshold is set empirically from product requirements (≥70% benefit preserved = useful promotion). Note the orthogonality bound predicts only ≈45%; K828 PASS would indicate significant structure beyond pure orthogonality.

**K829 (no catastrophic interference):**
```
∀ domain d ≠ medical: composed_ppl_d ≤ 1.5 × base_ppl_d
```
NRE norm preservation (Finding #275) predicts this holds; the 1.5× slack is generous.

---

## V. Self-Test

**What would falsify P1 (K828)?**
If η_medical < 0.70, then either:
- The adapters ARE near-orthogonal (η ≈ 1/√N ≈ 0.45), and the promotion concept requires a different architecture (e.g., explicitly promoting by merging ΔW_medical into W_base before further composition), OR
- scale=20 is too high, causing large ΔW magnitudes that dominate and interfere (per Finding #328)

**What would falsify P2 (K829)?**
If K829 fails, NRE's interference bound is violated for our adapters — implies significant shared subspace structure that creates cross-domain interference.

**What makes the failure mode impossible in future work?**
If K828 fails: true promotion would require explicitly setting W_base' = W_base + ΔW_medical, then training the other 4 adapters from the new base. This removes the composition interference by construction.
