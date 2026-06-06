# Peer Review: Adaptive Rank Selection (v2)

## NotebookLM Findings

Manual deep review conducted. The experiment is pure linear algebra simulation
(no training), making the mathematical claims directly verifiable from code and
results.json.

---

## Mathematical Soundness

### What holds

**The Eckart-Young-Mirsky framework is correct.** Reconstruction error formula,
domain generation, SNR control, and all five complexity metrics (effective rank,
stable rank, energy rank at 90/95/99%) are standard definitions implemented
without error. The test suite validates edge cases.

**The Kneedle implementation is correct.** The algorithm normalizes both axes
to [0,1], computes perpendicular distance to the diagonal line, and selects
the maximum-distance point. This is a faithful implementation of Satopaa et al.
(2011). It operates on a uniform rank grid (1, 2, ..., d), which eliminates
the non-uniform grid bias that broke v1. Verified: for exact_rank=8 at d=64
SNR=5, Kneedle correctly detects knee=8 while the threshold detector reports
38 (noise-inflated). This is the key fix that makes v2 credible.

**The null baseline is properly implemented.** Always-predict-16 is evaluated
on per-domain medians with the same 2x ratio criterion, providing a fair
comparison point.

**Per-domain median analysis is appropriate.** Seeds within a domain share
the same spectral structure (different random rotations U, V), so treating
them as independent observations would inflate degrees of freedom. Taking
per-domain medians first (N=15) is the correct approach.

**The spectral decay analysis in MATH.md Section 4.1-4.2 is sound.** The
derivation that both r_99 and the Kneedle knee scale as 1/|log(gamma)| for
geometric decay is correct, providing a principled explanation for why
energy_rank_99 correlates with Kneedle-optimal rank.

### What does not hold

**The Kneedle algorithm has a subtle edge case for nearly-flat spectra.**
The normalization `(errors - errors.min()) / max(errors.max() - errors.min(), 1e-10)`
maps the minimum error to 0 and maximum to 1. For a nearly-flat spectrum
(decay=0.98), the error curve barely decreases, making `errors.max() - errors.min()`
small. The Kneedle knee in this regime is numerically unstable. The paper
acknowledges this in Limitation 4 and MATH.md Section 4.3.1, so this is
documented rather than hidden.

**No formal justification for the 2x ratio tolerance.** The choice of "within
2x" as the prediction accuracy threshold is conventional but arbitrary. A factor
of 2 in rank means a factor of 2 in adapter parameters and 4x in N_max capacity.
This tolerance is generous enough that the practical impact of "correct within
2x" versus "exact" could be substantial for SOLE capacity planning. The paper
does not quantify this.

### Hidden assumptions verified

1. Assumption that LoRA converges to Eckart-Young optimum -- acknowledged in
   Limitation 1. This is the experiment's fundamental gap between simulation
   and reality.

2. Single (d,d) matrix per domain -- acknowledged in Limitation 2. Real LoRA
   involves multiple weight matrices per layer.

3. The v1 post-hoc 1.5x multiplier on r_95 has been ELIMINATED in v2. The
   paper now recommends energy_rank_99 directly without any ad-hoc correction
   factor. This addresses the v1 review's concern about overfitting.

---

## Novelty Assessment

**Low but sufficient for a micro-experiment.** The relationship between spectral
structure and LoRA rank is implicit in AdaLoRA (per-layer) and Aghajanyan et al.
(per-task). The specific contribution here is:

1. Identifying energy_rank_99 as the best predictor (over r_95, r_90,
   effective_rank, stable_rank) -- this is an empirical finding, not a
   theoretical advance.

2. Showing the Kneedle knee (signal-noise boundary) is better correlated
   with energy_rank_99 than with energy_rank_95, with a principled
   explanation (the spectral tail between 95% and 99% energy contributes
   disproportionately to reconstruction error).

3. Positioning as per-domain complement to AdaLoRA's per-layer approach,
   explicitly noted as composable.

For a micro-experiment within SOLE, this level of novelty is acceptable.
The goal is to establish a practical heuristic, not to publish a standalone
paper.

---

## Experimental Design

### Does this test what it claims?

**Yes, within the simulation framework.** The experiment claims that spectral
complexity of a target transformation predicts the optimal LoRA rank. Using
Kneedle on a uniform rank grid as ground truth, the correlations are strong
(rho 0.94-0.99) and the prediction accuracy beats the null baseline by
13-47pp across all 9 conditions.

### Assessment of v2 fixes

**Fix 1 (Kneedle): ADEQUATE.** The v1 curvature detector was definitively
broken (knee=48 for exact_rank=8). Kneedle on a uniform grid correctly
identifies the signal-noise boundary. Threshold method retained as cross-check
with appropriate caveat (conflates noise removal at low SNR).

**Fix 2 (null baseline): ADEQUATE.** The null baseline achieves 53-67% within
2x, confirming that K2's original 50% threshold was too generous on its own.
The strengthened requirement (beat null by >10pp) is appropriate.

**Fix 3 (multi-SNR): ADEQUATE.** Testing at SNR={5, 20, 100} covers low-noise,
medium-noise, and high-noise regimes. Results degrade gracefully at SNR=5
as expected.

**Fix 4 (per-domain medians): ADEQUATE.** Primary analysis uses 15 data points
per condition. Pooled analysis retained as secondary. This is methodologically
correct.

### Critical discrepancy: the PAPER.md narrative vs actual results

The PAPER.md summary table (lines 69-79) reports "Best 2x%" and "K1 rho" per
condition. The K1 rho column reports the best metric's correlation, which is
indeed energy_rank_99 in 7 of 9 conditions. However, inspection of
results.json reveals:

**The K2 best predictor is NOT always energy_rank_99:**

| Condition | K2 Best Predictor | energy_rank_99 within 2x |
|-----------|-------------------|--------------------------|
| d=64, SNR=5 | **effective_rank** (80.0%) | 53.3% |
| d=128, SNR=5 | **energy_rank_95** (86.7%) | not best |
| d=256, SNR=5 | **energy_rank_95** (86.7%) | not best |

At SNR=5, energy_rank_99 is actually WORSE than the null baseline at d=64
(53.3% vs 66.7%, delta = -13.3pp). This is because noise inflates the 99%
energy threshold, causing systematic overprediction. The paper acknowledges
this phenomenon in the text ("At SNR=5, performance degrades slightly") but
the summary table obscures it by always showing the BEST predictor's accuracy.

**This is not a mathematical error but it is misleading presentation.** The
paper's Correlation Analysis section (lines 83-98) correctly shows
energy_rank_99 as the best correlator at d=128/SNR=20, but the implication
that energy_rank_99 is universally best is contradicted by the low-SNR
conditions. The practical heuristic (Section "Practical Heuristic for SOLE",
lines 161-176) recommends energy_rank_99 unconditionally, which would fail
at SNR=5.

### Controls

The null baseline is appropriate. The multi-SNR sweep provides robustness
evidence. The threshold method as cross-check adds validation. No critical
missing controls.

### Statistical concerns

The per-domain analysis uses N=15 (15 domains), which is adequate for
Spearman correlations at the observed rho values (0.94+, all p < 1e-6).
The best-metric selection across 5 candidates introduces a mild multiple
comparisons issue, but the margins are large enough (all rho > 0.89) that
no correction would change the K1 verdict.

---

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml kill criteria:
- K1: Spearman rho < 0.5 -- SURVIVED (min rho = 0.94)
- K2: >2x off for >50% AND must beat null by >10pp -- SURVIVED

The kill criteria are correctly evaluated. The status "proven" is appropriate
given that all 9 conditions pass both criteria.

The evidence entries in HYPOTHESES.yml correctly reference the v2 revision.

---

## Macro-Scale Risks (advisory)

1. **The Eckart-Young assumption breaks for real LoRA.** Trained LoRA deltas
   A @ B have rank at most r by construction, but real training may not find
   the optimal rank-r subspace. The practical heuristic requires training a
   pilot adapter at r_init, computing SVD of the learned delta, then measuring
   r_99. This adds one pilot training run per domain.

2. **energy_rank_99 fails at low SNR.** Real fine-tuning may have noisy
   gradients (small learning rates, early stopping), producing low effective
   SNR. The heuristic should fall back to energy_rank_95 or effective_rank
   when the spectral tail is noise-dominated. A simple diagnostic: if
   r_99 > 2 * r_95, the tail is likely noise, and r_95 should be used.

3. **Mixed ranks break Grassmannian capacity.** N_max = d^2/r^2 assumes
   uniform rank. With adaptive ranks, the capacity analysis requires the
   mixed-rank Grassmannian, which is more complex. This is flagged in the
   paper (line 228) and MATH.md (Assumption 6).

4. **Per-layer rank variation.** AdaLoRA shows 3-4x rank variation across
   layers. A single per-domain rank may be suboptimal. The paper correctly
   positions this as complementary to per-layer allocation.

---

## Verdict

**PROCEED**

The v2 revision adequately addresses all four required fixes from the v1
review. The Kneedle detector produces sensible ground truth, the null baseline
contextualizes the predictions, multi-SNR robustness is demonstrated, and
per-domain medians are methodologically correct.

The core finding -- that energy_rank_99 of a target transformation's spectrum
predicts the Kneedle-optimal LoRA rank with rho > 0.94 across 9 conditions --
is sound within the simulation framework.

### Non-blocking issues (do not require re-review)

1. **Clarify that energy_rank_99 is not universally best.** The summary table
   in PAPER.md should note which metric is the K2 best predictor per condition,
   or add a column. At SNR=5, energy_rank_99 is WORSE than null (d=64: -13.3pp).
   The practical heuristic should include a fallback: "if r_99 > 2*r_95, use
   r_95 instead" (indicates noise-dominated tail).

2. **The "within 2x" tolerance is generous for SOLE capacity planning.** A 2x
   rank error translates to 4x error in N_max capacity. Future work should
   quantify the downstream impact of rank misprediction on composition quality
   and capacity utilization.

3. **The per-domain detail table (PAPER.md lines 113-131) uses energy_rank_95
   predictions**, not energy_rank_99. This is inconsistent with the recommendation
   of energy_rank_99 as the preferred predictor. The table should show r_99
   predictions to match the heuristic.
