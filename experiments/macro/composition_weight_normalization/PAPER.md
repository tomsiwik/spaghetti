# Weight-Normalized LoRA Composition: Research Digest

## Hypothesis

Weight-normalized composition using $1/\sqrt{N}$ scaling prevents PPL explosion
at high expert count ($N \leq 50$) and follows a power-law $N^{-\beta}$ with
$\beta \approx 0.5$ (random subspace regime).

## What This Experiment Is

An inference-only experiment testing 4 scaling strategies for composing N
independently-trained LoRA adapters via weight addition on Qwen2.5-7B:

1. **Unit weight** ($w=1.0$): known catastrophic baseline (PPL in trillions at N=5)
2. **Mean** ($w=1/N$): averaging regime, validated at N=5 (PPL=2.36)
3. **Sqrt** ($w=1/\sqrt{N}$): random subspace prediction from MATH.md
4. **Grid search** over $w \in \{0.01, 0.05, 0.1, 0.2, 0.5, 1.0\}$

Tested at $N \in \{5, 10, 25, 50\}$ using pilot-50 adapters (rank-16, all-modules).

## Key References

- **Task Arithmetic** (Ilharco et al., 2023): Task vector composition with scalar $\lambda$
- **TIES-Merging** (Yadav et al., NeurIPS 2023): Sign-aware merging for interference resolution
- **DARE** (Yu et al., 2023): Drop-and-rescale exploiting parameter redundancy
- **LoRA-Flow** (Wang et al., 2024): Dynamic per-token fusion gates (upper bound on expressivity)
- **sole_critical_path** (this project): 1/N scaling at N=5 gives PPL=2.36

## Empirical Results

**Note:** Only 5 adapters were available on the RunPod workspace, so N=10/25/50
were not tested. The experiment ran only at N=5. Power-law fitting was not possible
with a single data point. Despite this limitation, K2 produced a clear kill signal.

### Phase 1: Single-Expert Baselines

| Metric | Value |
|--------|-------|
| Number of adapters evaluated | 5 |
| Average single-expert PPL | 2.2461 |

### Phase 2: Scaling Strategies

| Strategy | N=5 |
|----------|-----|
| Unit ($w=1.0$) | 26,009,523,412,340 |
| Mean ($w=1/N=0.2$) | 5.7813 |
| Sqrt ($w=1/\sqrt{5}=0.447$) | 5.6382 |

Key observations:
- Unit weight confirms catastrophic PPL explosion (~26 trillion), consistent with sole_critical_path
- **Sqrt is marginally better than mean** (5.64 vs 5.78, ~2.5% improvement), suggesting adapters
  are between Regime B (random) and Regime C (correlated)
- Both normalized strategies produce PPL close to base model (5.70), indicating the adapters
  effectively cancel each other out — composition recovers base model quality, not better

### Phase 3: Grid Search at N=5

Grid search found $w=0.2$ (= $1/N$) as optimal among the tested grid points.
This is consistent with the mean strategy result. The sqrt weight $0.447$ slightly
overshoots but still produces reasonable PPL.

**Best grid weight:** $w=0.2$ → PPL = 5.7813
**Transfer check:** Not testable (only N=5 available)

### Power-Law Fit

Not possible with a single N value. Would require N=10,25,50 data points to
fit $\alpha^*(N) = N^{-\beta}$ and estimate $\beta$.

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: sqrt reduces PPL >50% vs unit | >50% | 100.0% reduction | **PASS** |
| K2: best PPL < 2x single avg | ratio < 2.0 | 2.57x (5.78 / 2.25) | **FAIL** |
| K3: weight transfers across N | ratio < 2.0 | Not testable (only N=5) | N/A |

**Verdict: KILLED by K2**

The best achievable composed PPL (5.78) is 2.57x worse than the average
single-expert PPL (2.25). This means **uniform static scaling alone cannot
make multi-expert composition competitive with individual experts**.

### Interpretation

The K2 kill has a subtle but important nuance: the comparison is between
a specialist evaluated on its own domain (PPL=2.25) versus 5 experts
composed and evaluated on each domain (PPL=5.78). The composed model
is essentially recovering **base model quality** (PPL≈5.70 from prior work),
which means:

1. The adapters' contributions cancel out under uniform scaling — each expert
   helps on its domain but hurts equally on others
2. Static uniform $\alpha$ cannot selectively weight experts by relevance
3. This motivates **per-input routing** (hash ring, PPL-probe) rather than
   static composition, which is already the SOLE architecture direction

## Limitations

1. **Contaminated eval data.** Per-domain eval falls back to train.jsonl tails.
   Relative comparisons between strategies are valid (same data); absolute PPL
   numbers are optimistic.

2. **Uniform scaling only.** This experiment tests a single scalar $\alpha$ for
   all adapters and all layers. Per-adapter or per-layer weights (cf. LoRA-Flow,
   PPL-probe) are not tested and could improve results significantly.

3. **NF4 quantization confound.** Base model is 4-bit quantized. Quantization
   noise may interact with the scaling factor in ways not captured by our
   spectral analysis.

4. **Adapter selection bias.** First-N alphabetical selection does not control
   for domain diversity or adapter quality. Different subsets could produce
   different optimal scaling factors.

5. **Single seed.** No randomization over adapter selection order. Results may
   not generalize to different compositions of the same size.

## What Would Kill This

- **K1 FAIL**: $1/\sqrt{N}$ does not help vs unit weight. Implies adapters are
  in fully-correlated regime (Regime C), only $1/N$ works. SOLE has inherent
  $O(1/N)$ dilution penalty.

- **K2 FAIL**: Even optimal scaling produces bad PPL. Implies interference is
  destructive (sign conflicts), not just constructive (magnitude). Motivates
  TIES-like sign-aware merging.

- **K3 FAIL**: Optimal weight is not transferable. Implies no stable power law
  in N; per-composition tuning required. Kills the "set it and forget it"
  scaling story.
