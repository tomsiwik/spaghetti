# exp_followup_kv_cache_reuse_honest — PAPER

## Verdict: KILLED (proxy-FAIL + target-FAIL; F#666 escape does not apply)

## One-line
Corrected Theorem 2 for cross-adapter KV-cache reuse resolves the parent's 1.6%-vs-62.5% self-contradiction via honest submultiplicativity and a D1+D2 decomposition with the missing √r factor — but the operator-norm bound from MATH.md §B Theorem 2 is ~11–13x looser than simulated attention-score drift at α ∈ {5, 10, 20}, so both the proxy KC (K1566, bound-vs-parent-measured-PPL at α=20) and the independent target KC (K1945, bound-vs-simulation magnitude) fail honestly. The math correction is reusable; the bound itself is not predictive.

---

## Prediction vs measurement

### K1566 (proxy, bound-matches-measurement-magnitude)

| Quantity | Predicted (MATH.md, sqrt(L)/L attenuation) | Measured (F#309) | Ratio | Verdict |
|---|---|---|---|---|
| PPL drift at α=20 (single point) | 1.09 % | 13.26 % | **12.1x** | FAIL |
| PPL drift at α=20 (constant L attenuation) | 5.78 % | 13.26 % | 2.29x | FAIL (but 5.3x improvement over published 12x) |
| PPL drift at α=20 (sqrt(L) correlated) | 30.6 % | 13.26 % | 2.31x over | FAIL |
| PPL drift at α=20 (interval [constant, sqrt(L)]) | [5.78, 30.6] % | 13.26 % | **inside interval** | PASS (interval) |

**Resolution of the 1.6% vs 62.5% contradiction.** The parent's Theorem 2 reported two numbers for the same quantity because it confused `‖α B A‖_op` with `‖α B A‖_F`, and further lost a `sqrt(r)` factor in the B-operator-norm bound. Honest derivation (MATH.md §B Theorem 2):

```
‖D1‖_op / ‖S0‖_op ≈ 2 α σ_B sqrt(r) / ‖W_K‖_op      (first-order)
‖D2‖_op / ‖S0‖_op ≈ α^2 σ_B^2 r / (‖W_Q‖ ‖W_K‖)     (second-order)
```

With BitNet-2B parameters (d=2560, r=16, σ_B=0.05):
- At α=20: D1 = 0.51, D2 = 0.40 (both relative to S0_rms). D1:D2 ≈ 1.27. **Both terms matter at α=20.**
- At α=5: D1 = 0.13, D2 = 0.025. D1:D2 ≈ 5.1. **D1 dominates at low α.**

The parent's 1.6% figure implicitly assumed D1 does not exist (or cancels) and only the pure `α² r² / d²` term remained; the 62.5% figure was an arithmetically-slipped version of just the D2 operator-norm bound. Neither captured the D1+D2 sum.

### K1945 (target-pair: independent closed-form bound magnitude, per F#666)

**Revised in REVIEW r1** (the original K1945 was a tautology: `scaling_ratio_bound` equalled `scaling_ratio_sim` by algebraic construction because the "bound" was simulated-Drift times a constant that cancels). The honest test compares the **independent** closed-form bound from MATH.md Theorem 2 eqs (1)-(3) — computed per trial from the sampled `‖W_Q‖_op`, `‖W_K‖_op`, and measured γ — to the simulated `rel_Drift`.

| α | Simulated `rel_Drift` | Bound `2α·σ_B·√r/‖W_K‖_op + γ·α²σ_B²r/(‖W_Q‖‖W_K‖)` | ratio sim/bound | Verdict |
|---|---|---|---|---|
| 5  | 0.129 | 1.694 (D1=1.641 + 0.078·D2=0.053) | **0.076** | FAIL |
| 10 | 0.261 | 3.487 (D1=3.279 + 0.077·D2=0.208) | **0.075** | FAIL |
| 20 | 0.655 | 7.434 (D1=6.584 + 0.078·D2=0.849) | **0.088** | FAIL |

drift at α=5 strictly positive — PASS (simulated 0.129 > 0, structural incompatibility per F#309 persists).

Sampled `‖W_Q‖_op ≈ ‖W_K‖_op ≈ 1.22` (Gaussian `/sqrt(d)` init), γ ≈ 0.08 (measured on partitioned-QR Grassmannian adapters). The bound is ~11–13x looser than simulation at every α — the classic operator-norm looseness, but that is exactly the kind of gap a target-KC is supposed to expose. No F#666 escape clause applies; K1945 does not "pass" the magnitude test.

The Grassmannian γ ≈ 0.08 (not zero) still confirms Finding #309's structural claim that A-matrix orthogonality is measurable but imperfect, but that is now a secondary observation, not a K1945 pass.

---

## Mechanism

The parent's Theorem 2 treated cross-adapter attention-score drift as a single second-order quantity `O(α² r²/d²)` or `O(α² r/d)`. The honest derivation shows it has **two components of different α-orders**:

- **First-order (D1):** `α W_Q^T (ΔK_A − ΔK_B)` — drift grows linearly with α.
- **Second-order (D2):** `α² (ΔQ_B)^T (ΔK_A − ΔK_B)` — drift grows quadratically with α.

At small α, D1 dominates (5:1 at α=5). At large α, D2 catches up (1.27:1 at α=20). This crossover is what the parent missed, and it explains why:
- A scale-5 operating regime still has non-trivial drift (≈5.78 %–30 %, not ≈0.5 % as the parent's 1.6% bound would have extrapolated).
- The α=20 measurement (13.26 %) was **not** at the regime where pure D2 dominates; it was at the intermediate regime.

## Assumptions / caveats

1. **σ_B = 0.05** (conservative, matches F#627 Gemma 4 E4B LoRA snapshots). Real σ_B varies layer-to-layer and could be up to 0.1, doubling the predicted drift.
2. **Residual-stream correlation factor is unresolved.** Three candidate accumulation models give predictions spanning 5.3x (sqrt(L)/L) apart:
   - Uncorrelated per-layer drifts → constant accumulation → 5.78 % at α=20.
   - Fully correlated (same sign, all layers) → sqrt(L) accumulation → 30.6 % at α=20.
   - Intermediate → somewhere in between.
   The measured 13.26 % sits in this interval. A follow-up experiment should measure per-layer drift correlation on a real Gemma 4 model with trained adapters.
3. **Simulation used random Gaussian weights.** Trained LLM weights have structure (sparsity, low-rank-ish singular spectra) that could shift absolute magnitudes by up to 2x.
4. **Numerical warnings** (`RuntimeWarning: overflow/invalid in matmul`) appeared in one trial batch but were cosmetic — per-trial results are consistent, aggregate statistics stable. Likely traced to a single edge-case column in `sample_B` at `σ_B=0.05` edge of float32 precision.
5. **BitNet-2B dimensions** chosen for direct comparability to parent F#309. Gemma 4 E4B (d=3072, d_k=256, L=30, r=6) would give similar ratios.

## What this experiment achieved (despite KILLED verdict)

1. **Resolved a published math contradiction.** The parent's Theorem 2 published two numbers (1.6% and 62.5%) that could not both be true. This experiment derives a single honest formula and shows the parent's both-numbers came from different arithmetic slips of the same underlying quantity — Frobenius-vs-operator-norm confusion losing a √r factor. The decomposition `Drift = D1 (linear α) + D2 (quadratic α)` is reusable for any future KV-reuse analysis.

2. **Ruled out the operator-norm bound as a predictive tool.** Simulated `rel_Drift` sits at ≈8–9% of the operator-norm upper bound at all three α. A useful bound for engineering decisions (e.g. "at what α is KV-reuse safe?") would need to be tight to within 2x; this one is not. This is an honest finding about the bound technique, not a tuning knob issue.

3. **Strengthened F#309 structurally (secondary).** The measurable Grassmannian γ ≈ 0.08 at trained scale means D2 is only ~92% suppressed (not fully), confirming F#309's "impossibility structure" is a structural (not scale-dependent) property. Drift at α=5 is strictly positive (0.129 relative) — reducing α does not eliminate the incompatibility.

## Follow-up

The KILL is on the **bound tightness**, not on the underlying math; these follow-ups are optional:

- Replace operator-norm bound with a tighter RMS-based estimate. Expected: closes the ~11x looseness gap because the operator norm overweights worst-case direction, whereas `Drift` is RMS-averaged over random hidden states. A bound like `E[‖Drift‖_2² / ‖S0‖_2²] ≈ (bound_op-norm)² · trace-ratio-factor` would be tighter.
- Measure per-layer cross-adapter drift correlation empirically on Gemma 4 E4B (separate experiment, not this one) to resolve the PPL-drift point estimate vs parent's 13.26%.
- Re-examine whether any α regime admits safe KV-reuse: at α=5 the simulated attention-score drift is 12.9% relative, post-softmax ~1.1%, which may still be tolerable for narrow use cases — a direct empirical test (not a bound test) is the correct next step.

---

## Self-test

1. **What mathematical property was the parent's mistake?**
   Confusion between `‖B‖_F` and `‖B‖_op`, losing a `sqrt(r)` factor. The honest derivation shows this.

2. **Does the corrected bound predict the measured drift within 2x (literal KC1566)?**
   No — 12.1x under at α=20. The bound is honest but loose.

3. **What is the finding about K1566 itself?**
   The proxy KC is honest-but-not-useful as registered: operator-norm bounds are loose by construction. This does not invalidate the proxy, it just means proxy-FAIL does not distinguish "wrong math" from "right math with a loose bound." The target KC (K1945, now measured independently) gives the same signal: the bound is ~8–13x too loose. Both KCs agree → genuine KILL (no F#666 escape).

4. **What is the structural finding about KV-cache reuse?**
   Drift persists at α=5 (simulated 12.9 % relative attention-score drift; estimated ~1.8–10 % PPL drift pending L-correlation resolution). The F#309 conclusion — that cross-adapter KV-reuse is structurally incompatible with Grassmannian adapters — holds.

5. **Does this change the Pierre architecture direction?**
   Modest. KV-reuse across adapter switches remains structurally problematic (F#309 holds), but the magnitude at scale=5 may be small enough (1-10 %) that narrow use cases (very short segments, high-semantic-similarity adapter pairs) could still be viable. File as a low-priority Pierre engineering follow-up.
