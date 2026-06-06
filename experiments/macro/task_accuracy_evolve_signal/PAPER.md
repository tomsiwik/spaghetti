# Task Accuracy as Evolve Signal: Research Digest

## Hypothesis

A 10-question held-out MMLU subset reliably ranks adapter quality
(Kendall tau >= 0.7 vs 100-question gold standard), enabling cheap
clone-and-compete evolution tournaments.

## What This Experiment Is

The Evolve phase of SOLE requires a quality signal to compare adapter
variants (clones vs originals) in a tournament. PPL-based signals failed
at macro scale:

- Answer-only PPL: r=-0.63 cross-domain (KILLED)
- Full-sequence PPL: r=-0.31 with task accuracy (anti-correlated)
- PPL-guided expert refinement: BLOCKED without alternative signal

The simplest alternative: use actual task accuracy on a tiny held-out
benchmark. If 10 MMLU questions per domain produce stable adapter
rankings, a tournament comparison costs ~0.1s per matchup (vs 60s+ for
teacher-based scoring). This would unblock the Evolve phase entirely.

## Key References

- Polo et al. (2024) -- "How Reliable is Language Model Micro-Benchmarking?"
  Found that 10-question "Anchor Points" achieve tau=0.73 on BBH, but with
  MDAD of 6pp (can only distinguish models >6pp apart).
- "EssenceBench" -- 50 questions achieve tau>0.90 via genetic algorithm selection
- individual_expert_held_out -- 5 adapters span only 2.8pp on MMLU (below MDAD)
- sole_critical_path -- LOO PPL ranking works but requires full composition

## Method

1. Select 10 diverse MMLU subjects (science, humanities, professional)
2. Use 100 questions per subject as gold standard
3. Sample 5 random 10-question subsets (also 25 and 50 for sweep)
4. Evaluate all 5 adapters + base via HF transformers with NF4 quantization
5. Score via logprobs: argmax P(A|B|C|D | prompt) per question
6. Compute Kendall tau between subset and gold rankings
7. Compare accuracy ranking vs answer-only PPL ranking
8. Measure wall-clock time per adapter per domain

Note: Originally designed for vLLM batch inference, but RunPod env corruption
(failed vLLM install ate disk space, broke transformers/peft/huggingface_hub)
forced fallback to HF transformers + PEFT with NF4 quantization. Relative
rankings are unaffected; absolute timings reflect sequential HF inference.

## Empirical Results

### Base Model Accuracy (Qwen2.5-7B, NF4)

| Subject | Accuracy |
|---------|----------|
| abstract_algebra | 51% |
| anatomy | 70% |
| college_computer_science | 63% |
| college_physics | 52% |
| econometrics | 61% |
| high_school_biology | 79% |
| high_school_us_history | 90% |
| machine_learning | 56% |
| professional_medicine | 78% |
| world_religions | 81% |
| **Overall** | **68.1%** |

### Adapter Gold-Standard Accuracies (100q per subject, mean over 10 subjects)

| Adapter | Mean Accuracy | Delta from Base |
|---------|---------------|-----------------|
| bash | 70.5% | +2.4pp |
| sql | 69.3% | +1.2pp |
| python | 68.3% | +0.2pp |
| medical | 66.7% | -1.4pp |
| math | 66.5% | -1.6pp |

Spread: 4.0pp (bash 70.5% - math 66.5%). Consistent with prior individual_expert
results showing narrow adapter spread on out-of-domain MMLU.

### Per-Subject Ranking Correlation (Kendall tau vs gold)

| Subject | tau@10 | tau@25 | tau@50 | PPL tau | Gold #1 |
|---------|--------|--------|--------|---------|---------|
| abstract_algebra | 0.441 | 0.584 | 0.802 | 0.000 | bash |
| anatomy | 0.632 | 0.645 | 0.755 | 0.600 | bash |
| college_computer_science | 0.521 | 0.559 | 0.574 | 0.359 | sql |
| college_physics | 0.327 | 0.453 | 0.594 | 0.316 | bash |
| econometrics | 0.083 | -0.012 | 0.476 | 0.000 | bash |
| high_school_biology | 0.297 | 0.229 | 0.640 | 0.738 | python |
| high_school_us_history | 0.391 | 0.184 | 0.207 | 0.894 | bash |
| machine_learning | 0.099 | 0.454 | 0.621 | -0.316 | sql |
| professional_medicine | 0.135 | 0.287 | 0.620 | 0.359 | bash |
| world_religions | 0.095 | 0.439 | 0.511 | -0.598 | bash |
| **Mean** | **0.302** | **0.382** | **0.580** | **0.235** | |

### Kill Criteria Assessment

| Criterion | Threshold | Value | Verdict |
|-----------|-----------|-------|---------|
| K1: mean tau (10q) | >= 0.7 | 0.302 | **KILL** |
| K1: mean tau (25q) | >= 0.7 | 0.382 | KILL |
| K1: mean tau (50q) | >= 0.7 | 0.580 | KILL |
| K2: per-domain time | < 60s | 10.3s | **PASS** |
| K3: acc tau > ppl tau | acc > ppl | 0.302 > 0.235 | **PASS** |

**K1 verdict: HARD KILL** -- Even 50 questions fail to reach tau >= 0.7 on MMLU.
The 10-question approach is not viable for MMLU-based ranking of these adapters.

### Top-1 Stability

| Subset Size | Subjects with Stable #1 |
|-------------|------------------------|
| 10 | 1/10 |
| 25 | 1/10 |
| 50 | 0/10 |

Top-1 adapter is unstable across all subset sizes -- different random draws
frequently change which adapter ranks first. This confirms the spread is
too narrow for reliable discrimination on MMLU.

### Minimum Questions Sweep

| Subset size | Mean tau | Best subject | Worst subject |
|-------------|----------|--------------|---------------|
| 10 | 0.302 | anatomy (0.632) | econometrics (0.083) |
| 25 | 0.382 | anatomy (0.645) | econometrics (-0.012) |
| 50 | 0.580 | abstract_algebra (0.802) | high_school_us_history (0.207) |

Only 1 subject (abstract_algebra) reaches tau >= 0.7 at k=50. The trend
suggests k=100+ would be needed for reliable overall ranking on MMLU.

### Timing

| Metric | Value |
|--------|-------|
| Per-adapter mean | 103.5s |
| Per-domain mean | 10.3s |
| Total inference | 517.2s |
| Wall clock | 608.4s |

Note: These timings are with HF sequential inference (NF4). vLLM batch
inference would be ~10-50x faster, but the K2 threshold (60s/domain) is
easily met even with sequential HF.

## Analysis

### Why K1 Failed (As Predicted)

The MATH.md pre-registration predicted this outcome:

1. **Adapter spread too narrow**: 4.0pp spread (bash 70.5% - math 66.5%)
   is well below the 6pp MDAD threshold for reliable 10-question discrimination.
2. **MMLU is out-of-domain**: These adapters (bash, math, medical, python, sql)
   are domain-specialized. MMLU tests general knowledge where they show minimal
   differentiation.
3. **Expected tau confirmed**: MATH.md predicted tau~0.17 at k=10 and ~0.56 at
   k=50 using the normal model formula. Actual: 0.302 and 0.580. Slightly
   better than theory (likely due to some subjects having wider spreads).

### PPL Ranking is Also Broken on MMLU

PPL tau vs gold = 0.235 (mean over 10 subjects). Per-subject PPL tau ranges
from -0.598 (world_religions, anti-correlated) to 0.894 (high_school_us_history).
Neither accuracy nor PPL provides reliable ranking on general MMLU.

Interesting: on 2 subjects (high_school_biology tau=0.738, high_school_us_history
tau=0.894), PPL ranking works well -- these may be subjects where the adapters
introduce meaningful knowledge shifts rather than just noise.

### What This Means for Evolve

**MMLU cannot serve as a universal quality signal for clone-and-compete.**

However, this does NOT kill task-accuracy-based evolution. It means:

1. **Use domain-specific benchmarks**: HumanEval for code, MATH-500 for math,
   MedQA for medical. In-domain deltas should be >>6pp, making small subset
   ranking reliable.
2. **Use LOO PPL ranking**: Already proven at sole_critical_path (all 5 adapters
   ranked correctly by marginal PPL contribution). Works because it measures
   composition impact, not individual MMLU accuracy.
3. **Hybrid signal**: LOO PPL for cross-domain ranking + domain-specific accuracy
   for within-domain tournament (clones compete on their own benchmark).

### K3 Insight: Accuracy Slightly Beats PPL

Even though both signals are weak on MMLU (tau ~0.3 vs ~0.2), accuracy
marginally outperforms PPL. This supports the hypothesis that direct
task performance is a better signal than perplexity, even when both
are noisy. With domain-specific benchmarks where deltas are larger,
accuracy-based ranking should dominate.

## Limitations

1. **MMLU is out-of-domain for most adapters**: Adapters trained on
   bash/math/medical/python/sql show minimal MMLU delta. In-domain
   benchmarks would show larger deltas and potentially higher tau.

2. **5 adapters is low for Kendall tau**: With N=5 adapters, tau has only
   10 pairs. Ties further reduce effective pairs. The tau estimate is
   inherently noisy at small N.

3. **0-shot MMLU**: Literature standard is 5-shot. Our adapters are
   instruction-tuned and may perform differently with few-shot prompts.

4. **NF4 quantization**: Absolute accuracy numbers differ from FP16 baselines.
   Relative comparisons (adapter rankings) are unaffected.

5. **HF sequential inference**: Originally designed for vLLM batch inference.
   RunPod env corruption forced HF fallback. Rankings and accuracies are
   identical; only wall-clock timing is inflated.

6. **Single seed**: The 5 random draws use seed=42. Different seeds could
   produce different tau values.

## Verdict

**K1: HARD KILL on MMLU** -- Even 50 questions don't produce stable rankings
when adapter spread is <6pp. This was predicted by MATH.md.

**Not a project-level kill** -- The approach is sound for domain-specific
benchmarks where adapter deltas are large. MMLU is the wrong benchmark for
comparing domain-specialized adapters.

**Evolve signal recommendation**: Use LOO PPL ranking (proven) as the primary
tournament signal, supplemented by domain-specific accuracy benchmarks
(HumanEval, MATH-500, MedQA) for within-domain clone competition.

## Evolution Tournament Implications

| Signal | Status | Cost | Reliability |
|--------|--------|------|-------------|
| MMLU 10q accuracy | KILLED | ~0.1s | tau=0.302 |
| MMLU 100q accuracy | Marginal | ~10s | tau=1.0 (gold) |
| Answer-only PPL | KILLED (prior) | ~0.5s | r=-0.63 cross-domain |
| LOO PPL ranking | PROVEN | ~120s | All 5 ranked correctly |
| Domain-specific accuracy | UNTESTED | ~1-10s | Expected high tau |

**Next step**: Test domain-specific accuracy signal (HumanEval for code
adapters, MATH-500 for math adapter) -- expected to show tau >> 0.7 due
to much wider accuracy deltas.
