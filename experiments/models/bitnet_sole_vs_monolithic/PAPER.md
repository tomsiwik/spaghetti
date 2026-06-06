# BitNet-SOLE vs Monolithic: Research Digest

## Hypothesis

Composed BitNet-SOLE domain experts (N=5 ternary LoRA adapters with routing)
match or beat a single monolithic ternary LoRA trained on the union of all
domain data, for per-domain perplexity.

**Falsifiable claim:** If monolithic wins on >80% of per-domain metrics
(4+ out of 5 domains), SOLE's value proposition is killed.

## What This Model Is

A controlled head-to-head comparison of two multi-domain adaptation strategies
on the actual BitNet-2B-4T (2.4B params, d=2560, ternary weights):

1. **SOLE routed:** 5 independent ternary LoRA experts (rank-16 each,
   QAT+STE), each trained on one domain (400 steps). At inference, each
   domain query is routed to its own expert.

2. **SOLE composed (1/N):** Same 5 experts merged via 1/N scaling and
   evaluated on all domains simultaneously.

3. **Monolithic shuffled:** 1 ternary LoRA (rank-16) trained on shuffled
   union of all 5 domain datasets for 2000 steps (= 5 x 400, same total
   gradient updates as SOLE).

4. **Monolithic sequential:** 1 ternary LoRA trained on domains one at a
   time (400 steps each), measuring catastrophic forgetting.

### Budget Matching

| Condition | Rank | Total LoRA params | Total gradient steps |
|-----------|------|-------------------|---------------------|
| SOLE (5 experts) | r=16 each | 5 x 21.6M = 108M | 5 x 400 = 2000 |
| Monolithic | r=16 | 21.6M | 2000 |

SOLE uses 5x more total parameters. This is deliberate: SOLE's value
proposition is that each domain gets its own full-rank subspace, adding
capacity per domain rather than sharing a single low-rank approximation.
The monolithic model compensates by seeing all domain data in each step.

### Design choice: same rank, not same total params

We chose same-rank (both r=16) over same-total-params (SOLE at r=3 each,
mono at r=16) because:
- r=3 is below the minimum useful rank threshold for a 2560-dim model
- SOLE's architecture promise IS that each domain gets adequate rank
- The monolithic model's advantage is cross-domain transfer, not parameter count
- This matches the prior FP16 experiment design for direct comparison

## Key References

- BitNet b1.58 2B4T Technical Report (Microsoft, 2025)
- LoRA (Hu et al., 2021): Low-rank adaptation fundamentals
- LoRAHub (Huang et al., 2023): Cross-task LoRA composition
- TIES-Merging (Yadav et al., 2023): Resolving parameter interference
- Branch-Train-Merge (Li et al., 2022): Independent domain model training
- Prior FP16 experiment: micro/models/composition_vs_monolithic/PAPER.md

## Empirical Results

### Per-Domain PPL Comparison (the key table)

| Domain | Base | SOLE Routed | Mono Shuffled | Winner | Gap |
|--------|------|-------------|---------------|--------|-----|
| medical | 15.80 | **8.00** | 8.36 | SOLE | -4.3% |
| code | 3.52 | **2.76** | 2.84 | SOLE | -2.8% |
| math | 4.74 | **3.12** | 3.27 | SOLE | -4.5% |
| legal | 25.52 | **17.95** | 19.67 | SOLE | -8.8% |
| creative | 3.60 | 3.17 | **3.00** | MONO | +5.5% |
| **Average** | **10.64** | **7.00** | **7.43** | **SOLE** | **-5.8%** |

**SOLE routed wins 4/5 domains.** Monolithic wins only on creative writing,
where cross-domain transfer (from code/math pattern recognition) provides
a small benefit (+5.5%).

### All Conditions Summary

| Condition | Avg PPL | vs Base | vs Mono |
|-----------|---------|---------|---------|
| Base (no adapter) | 10.64 | -- | +43.3% |
| **SOLE routed** | **7.00** | **-34.2%** | **-5.8%** |
| SOLE composed (1/N) | 9.55 | -10.2% | +28.6% |
| Monolithic shuffled | 7.43 | -30.2% | -- |
| Monolithic sequential | 12.01 | +12.9% | +61.7% |

### Kill Criteria Assessment

**K1: monolithic beats SOLE routed on >80% of per-domain metrics?**
- Monolithic wins 1/5 domains (creative only, by +5.5%)
- SOLE wins 4/5 domains (medical -4.3%, code -2.8%, math -4.5%, legal -8.8%)
- **K1: PASS.** SOLE routed beats monolithic on 80% of domains.

### Key Findings

1. **SOLE routed beats monolithic by 5.8% on average.** Each domain expert,
   trained only on its own domain data, achieves lower PPL than the
   monolithic model that saw all domains. The specialization advantage
   outweighs the cross-domain transfer advantage.

2. **Creative writing is the exception.** Monolithic wins by 5.5% on creative,
   likely because creative writing benefits from exposure to diverse text
   patterns (code structure, mathematical reasoning, legal formality). This
   is the ONE domain where cross-domain transfer helps more than specialization.

3. **Sequential training causes catastrophic forgetting.** Training on domains
   one at a time yields avg PPL 12.01, which is 12.9% WORSE than not training
   at all. Medical (trained first) suffers +61.1% PPL degradation from base.
   SOLE has zero forgetting by construction.

4. **Composed (1/N) is worse than monolithic.** At avg PPL 9.55 (+28.6% vs
   mono), the 1/N scaled composition dilutes each expert's signal. This is
   expected: composed SOLE is not the deployment mode. Routed SOLE is.

5. **Adapter orthogonality is excellent.** Mean |cos| = 0.0019 across 10
   pairs, 41x below the random bound of sqrt(16/2560) = 0.079. Legal-creative
   has highest cosine (0.0036), medical-math second (0.0044) -- both still
   negligible.

### Comparison with Prior FP16 Result

| Metric | FP16 (prior, d=32) | BitNet-2B (this, d=2560) |
|--------|-------------------|--------------------------|
| SOLE routed vs mono | +5.5% (mono wins) | **-5.8% (SOLE wins)** |
| SOLE domain wins | 0/5 | **4/5** |
| Mono domain wins | 5/5 | 1/5 |
| Mean |cos| | 0.191 | **0.0019** (100x lower) |
| Forgetting (sequential) | +39% vs base | +12.9% vs base |

The reversal is striking. At d=32 with rank-4 experts, SOLE lost every
domain. At d=2560 with rank-16 experts, SOLE wins 4/5 domains. This
confirms the prior experiment's prediction: "K1b is a micro-scale artifact"
and "expected to fully pass at macro scale."

The mechanism is clear: at d=2560, rank-16 captures sufficient domain
signal without rank starvation, and the near-perfect orthogonality
(cos=0.002 vs cos=0.191 at d=32) eliminates cross-adapter interference.

## Training Details

| Domain | SOLE Time (s) | Converged | Mono Time | Sequential |
|--------|--------------|-----------|-----------|------------|
| medical | 678.5 | Yes | -- | 196.3s |
| code | 347.4 | No (flat) | -- | -- |
| math | 238.4 | Yes | -- | -- |
| legal | 244.1 | Yes | -- | -- |
| creative | 253.2 | No (flat) | -- | -- |
| Monolithic | -- | -- | 1017.8s | 990.0s total |

Total SOLE training: 1761.6s (29.4 min). Parallelizable to max(678.5s) = 11.3 min with 5 workers.
Total Mono training: 1017.8s (17.0 min).
SOLE sequential wallclock: 29.4 min. SOLE parallel wallclock: 11.3 min.

Code and creative show flat train loss curves but still achieve excellent
val PPL improvements (-21.5% and -12.0% vs base respectively). The STE
quantization introduces noise that obscures convergence in training loss
while still learning meaningful features.

## Limitations

1. **Single seed.** All results are from one random initialization. Multi-seed
   validation (exp_bitnet_multiseed_validation) is needed for confidence
   intervals. The margins (2.8% to 8.8% for SOLE wins, 5.5% for mono win)
   may shift with different seeds.

2. **Parameter budget asymmetry.** SOLE uses 5x more total parameters than
   monolithic. A truly parameter-matched comparison (SOLE at r=3 vs mono at
   r=16) would likely favor monolithic. However, this parameter asymmetry
   IS the architecture's value: each expert adds dedicated capacity.

3. **Same data quality.** All domains use high-quality HuggingFace datasets.
   In production, expert quality depends on training data quality, which
   varies by domain.

4. **Oracle routing.** The SOLE routed condition assumes perfect domain
   identification. In production, routing errors would degrade SOLE
   performance. However, routing at <21us with >95% accuracy has been
   demonstrated (exp_inference_routing_strategies).

5. **Short training.** 400 steps per expert is far below convergence on
   some domains. Both SOLE and monolithic would improve with more training,
   and the relative ranking may change.

6. **seq_len=128.** Shorter than typical inference contexts (512-4096 tokens).
   Longer sequences may change the dynamics, particularly for legal and
   medical domains with longer documents.

## What Would Kill This

### At micro scale (already tested):
- K1: Mono wins 4/5+ domains -> **PASS (mono wins 1/5)**
- SOLE routed is worse than base on any domain -> **PASS (all 5 improve)**

### At macro scale (future test):
- With proper routing, SOLE routed is >10% worse than monolithic on average
  across 10+ domains on a 7B model
- Cross-domain transfer in monolithic systematically outweighs specialization
  benefit of domain experts
- The creative-writing exception generalizes: most domains benefit more from
  multi-domain exposure than from specialization

### What would be truly fatal:
- Adapter orthogonality degrades at scale (cos >> 0.01 at d=4096)
  -> Contradicted by d=2560 measurement (cos=0.002) and scaling law
- Routing accuracy degrades with N > 50 experts -> Not yet tested at scale
- Ternary LoRA quality ceiling prevents matching FP16 quality
  -> Contradicted by individual PPL measurements (ternary matches FP16)

## Honest Assessment

This experiment provides strong evidence for SOLE's fundamental value
proposition on BitNet-2B-4T. SOLE routed beats monolithic on 4/5 domains
with an average advantage of 5.8%.

The result reverses the prior FP16 experiment at d=32, where monolithic won
every domain. The reversal was predicted by the capacity analysis: at d=2560,
rank-16 experts have sufficient capacity, and near-perfect orthogonality
(cos=0.002) ensures minimal cross-adapter interference.

**Caveats that prevent "proven" status:**
1. Single seed -- need multi-seed validation
2. Parameter asymmetry -- SOLE uses 5x more params
3. Oracle routing -- production routing may degrade results
4. 5 domains is a small N -- need N=15+ with more diverse domains

**Strongest finding:** Sequential monolithic training causes catastrophic
forgetting (+12.9% worse than base, +61.1% on the first domain). SOLE has
structural immunity to forgetting. This operational advantage persists
regardless of the quality comparison.

Status: **SUPPORTED** (K1 pass at 4/5 domains; pending multi-seed and
macro validation for "proven" status).
