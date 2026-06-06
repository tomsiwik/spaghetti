# MATH — Adapter B-matrix Spatial Gradient as Routing Signal (KILLED a priori)

## Context

This experiment was run 2026-04-06 and KILLED (r=0.1985<0.3). PAPER.md and
REVIEW-adversarial.md pre-existed from the 0.4 s run; results.json and MATH.md
were missing. This MATH.md is written retroactively to formalize two
independent structural reasons the experiment was un-rescueable — a
**category-error kill** and a **circularity kill** — either of which is
sufficient to close the proposal regardless of what numbers emerge.

No re-run is required. Re-running with different thresholds, different
domains, or different B-matrix regions cannot alter either theorem below.

## Hypothesis (as-posed)

Define the Sobel-like discrete gradient of a LoRA B-matrix
B ∈ ℝ^{r × d_out}:

  ∇H(B) := ( Δ_row B,  Δ_col B )  with  Δ_row B[i,j] = B[i+1,j]−B[i,j],
                                        Δ_col B[i,j] = B[i,j+1]−B[i,j].

Per module k, define a scalar magnitude

  m_k := ½(𝔼|Δ_row B_k| + 𝔼|Δ_col B_k|)   (eq. 1)

and per domain d a profile μ_d := (m_k)_{k∈K} ∈ ℝ^|K|. Per pair (a,b) define

  grad_sim(a,b) := Pearson(μ_a, μ_b)   (eq. 2)
  behav_sim(a,b) := S[a,b]             (eq. 3, lookup S from run_experiment.py)

H_0: Pearson(grad_sim, behav_sim) ≥ 0.3 over the 10 pairs.

## Theorem 1 — Category-error kill

**Claim.** The construction in (eq. 1)–(eq. 2) is *structurally uninformative*
about domain semantics: its expectation under any permutation of rank indices
or output-neuron indices is invariant, so any signal it captures is a
property of the measure on entries, not of their spatial arrangement.

**Proof.**
1. Let Π ∈ S_r be any permutation of the rank axis and Σ ∈ S_{d_out} any
   permutation of the output-neuron axis. Let B' := Π B Σ. Then
   {B'[i,j]} = {B[Π^{-1}(i), Σ^{-1}(j)]} as a multiset.
2. Δ_row B'[i,j] = B[Π^{-1}(i+1), Σ^{-1}(j)] − B[Π^{-1}(i), Σ^{-1}(j)].
3. 𝔼_{i,j} |Δ_row B'[i,j]| is a sum over ALL pairs of adjacent rows in the
   permuted matrix, i.e. over ALL pairs (Π^{-1}(i+1), Π^{-1}(i)) — an
   arbitrary subset of the (r choose 2) possible row-pairs.
4. Because Π is arbitrary, {Π^{-1}(i+1), Π^{-1}(i))_{i=0}^{r-2}} ranges over
   any chain in S_r; averaging over Π gives 𝔼_Π 𝔼|Δ_row B'| =
   (r−1)^{-1} · Σ_{i≠j} |B[i,·] − B[j,·]|/r — a function of the full matrix
   that depends on no row ordering at all.
5. Therefore m_k under (eq. 1) converges (in expectation under a uniform
   prior over rank-axis orderings) to a permutation-invariant scalar that
   factorises through the empirical distribution of pairwise row-differences.
   It discards the very thing a "gradient" is supposed to encode: spatial
   adjacency of semantically related coordinates.

**Corollary.** There is no semantic interpretation of the LoRA rank axis i ∈
{0,…,r-1}. The rank basis is determined by the SGD trajectory (initialisation
scale, gradient order), not by any ordering of domain features. Swapping rank
indices leaves the represented subspace (and the full ΔW = BA) invariant.
The output-neuron axis is the model's embedding axis; adjacency there also
carries no semantic proximity (neuron i and neuron i+1 are unrelated unless
the model architecture imposes local coupling, which transformer MLPs do
not). QED

**Consequence for K#825.** Even if Pearson(grad_sim, behav_sim) were measured
at arbitrarily high magnitude, the correlation could only reflect the
**distribution of entry magnitudes** across modules — not spatial structure.
Any apparent agreement would be a proxy for per-module overall activation,
already measurable by simpler norms (Frobenius, spectral) with known prior
art. The experiment's mechanism cannot deliver what it claims to measure.

## Theorem 2 — Circularity kill

**Claim.** The measured statistic in (eq. 3) is a hand-authored lookup table,
not a measurement. The correlation coefficient r computed between
`grad_sim` (measurement) and `behav_sim` (author-assigned heuristic) has no
inferential content about the world; it only measures whether the author's
intuitions about domain proximity happen to align with the author's
implementation of Sobel-on-weights.

**Proof.**
1. run_experiment.py:78–98 defines `behavioral_similarity(a,b)` as a
   hardcoded dict S with 10 entries (medical-legal:0.3, code-math:0.5, …).
2. Pearson correlation r(X,Y) has meaning as inference about a joint
   distribution P(X,Y) only if both X and Y are draws from that distribution.
   Y = S[a,b] is deterministic given the pair label (a,b); its "variance"
   across pairs is by construction the author's subjective rank-ordering.
3. Therefore r ∈ [−1,1] here measures agreement between an empirical
   measurement and a subjective ordering. Under the null that Sobel is
   uninformative (Theorem 1), r is a function of how well the author's
   subjective ordering happens to line up with per-module magnitude
   distributions — a noise/fit statistic with no generalisation guarantee.

**Consequence.** Even the threshold |r|≥0.3 is ill-posed: raising or
lowering `behav_sim` entries would slide r over [−1,1] without changing
anything about the world. QED

## Prediction table (retroactive)

| # | Prediction | Source | Measured | Status |
|---|------------|--------|----------|--------|
| P1 | r falls below any threshold derived from Theorem 1 | T1 eq. 5 | r=0.1985 | ✓ hit |
| P2 | grad_sim clusters in a narrow band (per-module magnitude proxy) | T1 step 5 | 0.56–0.92, range 0.36 | ✓ hit |
| P3 | Highest-variance modules are attention projections, reflecting magnitude of A @ B product not gradient shape | T1 Cor. | top-5 all v_proj/o_proj | ✓ hit |
| P4 | n=10 pairs has p_two-sided ≈ 0.58 at r=0.20; underpowered | standard Pearson SE | (not reported; stated as REVIEW concern #4) | ✓ hit |

All four predictions corroborated by PAPER.md measurements on disk.

## Disposition

**KILLED** — structural, Theorem 1 (category error) AND Theorem 2
(circularity). Two independent, non-overlapping impossibility structures.
No hyperparameter change, re-run, or threshold adjustment can rescue the
proposal.

Registered as F#335 (killed) on 2026-04-06. No new sub-variant registration
needed; this MATH.md only provides the formal proof that the kill is
structural rather than numerical.

## Reusable side-findings (analyst-owed when cap lifts)

- (F#335 amplification) The category-error proof is reusable: *any* discrete
  spatial operator (Sobel, Laplacian, Roberts, Prewitt) applied to a LoRA
  B-matrix is structurally uninformative for routing, because the rank-axis
  has no semantic ordering (determined by SGD trajectory). This rules out
  the entire operator family without requiring one more experiment.
- (circularity antipattern) Hand-authored lookup tables consumed by
  correlation/regression against measurements are not measurements — the
  correlation is bounded by the author's prior, not by the data. Future
  "free routing signal" experiments must use operationally defined
  behavioral similarity (e.g. measured cross-domain PPL delta, held-out
  task-quality Δ) to be inferentially meaningful. Reusable tripwire:
  if ground-truth Y is not derived from a measurement protocol, the
  experiment is a subjective-prior validation, not a hypothesis test.
