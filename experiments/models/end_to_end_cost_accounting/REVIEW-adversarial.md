# Peer Review: End-to-End Cost Accounting

## NotebookLM Findings

Skipped -- this is an accounting exercise, not a mechanism experiment. The review focuses on numerical verification and hidden assumptions.

## Mathematical Soundness

### Arithmetic Verification

The core decomposition adds up correctly:

```
$0.354 + $0.071 + $0.023 + $0.014 + $0.014 + $0.002 + $0.000 = $0.477
```

Verified against results.json: `total_per_expert = 0.4771`. Consistent.

### K1 Assessment: Correct

$0.477 < $1.00. Pass with 52% margin. Straightforward.

### K2 Assessment: Definitional Problem

The K2 criterion as stated in HYPOTHESES.yml is:

> "overhead (data gen, eval, merge, benchmark) >3x training cost"

The paper defines "training cost" as `C_train + C_load = $0.085` and "overhead" as `C_total - training = $0.392`, giving ratio 4.61x.

**Issue 1: Data generation is not "overhead" in the usual sense.** The K2 criterion was designed to catch hidden engineering costs that silently inflate per-expert cost. Teacher API cost is the primary input cost -- it is the raw material, not overhead. Calling it "overhead" and then arguing the kill is "mechanistically informative but not economically threatening" is post-hoc reframing. The paper acknowledges this clearly (Section "K2 Interpretation"), which is good intellectual honesty, but it means K2 was poorly designed from the start. A better K2 would have been: "non-teacher, non-training overhead > 1.0x training cost." Under that formulation, non-teacher overhead is $0.038, ratio = 0.45x, a clear pass.

**Verdict on K2:** The kill criterion was poorly scoped. The paper's interpretation is reasonable but should have been caught at hypothesis design time. The "supported" status is defensible given the K2 reinterpretation.

### K2 Numerical Check

```
overhead = $0.477 - $0.085 = $0.392
ratio = 0.392 / 0.085 = 4.61x
```

Arithmetic is correct.

### Scaling Projections: One Error Found

**Scenario C (A5000 GPU):** The paper claims $0.400/expert but the code uses `train['total_time_min']` (15 min, which includes model loading) at A5000 rate. This is valid -- same total training time, cheaper GPU. But it implicitly assumes the same training time on A5000 as on 4090, which is wrong. A5000 has roughly 60% of 4090's throughput for QLoRA workloads (FP16 TFLOPS: 27.8 vs 82.6, but QLoRA is memory-bound so the gap is smaller, probably 1.3-1.5x). The 15 min on 4090 would be approximately 20-22 min on A5000. At $0.16/hr this is $0.053-$0.059 vs the claimed $0.040. This makes Scenario C approximately $0.41-$0.42 and Scenario D approximately $0.07-$0.08.

**Impact:** Minor. The directional conclusion (teacher cost dominates) is unchanged. The "optimal" $0.061 figure is modestly underestimated, probably $0.07-$0.08. Not enough to threaten K1.

### CPU Operations at $0.00: Technically Correct, Slightly Misleading

The paper claims CPU operations cost $0.00 because they run on the local machine. This is true for marginal cost accounting. However:

1. GS projection at N=50 takes 5 minutes total. At N=10,000, this is O(N^2 * r * d) total if done pairwise, which is 40,000x more -- roughly 3.3 days on CPU. The paper's O(N*r*d) per expert claim in MATH.md is correct for incremental GS (project against existing basis), but does not account for the growing basis size. The true cost per expert is O(k*r*d) where k is the current number of experts, making total cost O(N^2*r*d/2).

2. At SOLE cosines of 0.0002, GS projection is indeed a near-no-op in practice (the projection magnitude is negligible). So the O(N^2) scaling is theoretical, not practical. The paper should state this more precisely.

**Impact:** Does not affect the cost accounting at N=50 or even N=500. At N=10,000+, GS may need GPU acceleration or could be skipped entirely given SOLE's structural orthogonality. Not a cost accounting error, but an incomplete scaling analysis.

## Novelty Assessment

This is a cost accounting exercise, not a novel mechanism. Novelty is not the goal. The contribution is honest forensic accounting of the SOLE pipeline.

**Prior art check:** No published end-to-end cost breakdowns for LoRA distillation pipelines were found in the references. This fills a genuine gap -- most papers report only training cost or API cost, not both together with all pipeline overhead.

## Experimental Design

### Strengths

1. **Uses actual data.** The Groq generation log is parsed directly. Training times come from the pilot-50 run. The $22 total spend is a real number.

2. **Honest about estimates.** The Limitations section clearly flags which numbers are measured vs estimated (training time, model loading time, benchmark time).

3. **Correct identification of dominant cost.** The 74.1% teacher API finding is the key insight and it is well-supported.

### Weaknesses

1. **No variance or confidence intervals.** Single pilot run, no repeated measurements. The paper acknowledges this in Limitations but it means the $0.477 figure could easily be $0.40-$0.55. At this margin above K1, that is acceptable.

2. **"From pilot PAPER.md" circular sourcing.** Training time of ~15 min/expert is cited from the pilot paper, which itself may be an estimate. There are no train_meta.json files available locally to verify. The code explicitly notes this. Acceptable given constraints but fragile.

3. **Idle cost calculation is residual-based.** `idle = $22 - accounted`. This catches everything unexplained, which is good, but means any error in the "accounted" numbers shows up as idle time rather than being attributed correctly. The $1.16 idle figure could include misattributed training time, forgotten SSH sessions, or pricing rounding.

4. **8B teacher quality assumed equivalent.** The $0.119/expert with 8B teacher is conditional on quality holding up. The paper flags this in Limitations and "What Would Kill This," which is correct. But the scaling projections present $0.119 without a quality asterisk in the table. The "Optimal" $0.061 scenario compounds this assumption.

## Hypothesis Graph Consistency

The HYPOTHESES.yml entry has status "supported" with K1 PASS and K2 KILL explained as teacher-dominated. The kill criteria are:

- K1: true cost per expert > $1.00 -- tested, PASS
- K2: overhead > 3x training cost -- tested, KILL (4.61x)

The "supported" status with a K2 kill requires justification. The evidence entry provides this justification (teacher cost is a known lever, not hidden overhead). This is a reasonable interpretation. Strictly, one kill criterion failing should make the status "supported" at best, not "proven." The current "supported" status is appropriate.

## Macro-Scale Risks (advisory)

1. **8B teacher quality degradation.** If 8B teacher produces measurably worse experts, the quality-adjusted cost per useful expert could exceed the 70B teacher cost. This needs empirical validation before claiming the $0.119 scenario.

2. **GS projection at N>1000.** Even if near-no-op at SOLE cosines, the quadratic pairwise computation may become a latency bottleneck for the pipeline (not a cost bottleneck, since CPU is "free," but a wall-clock bottleneck). Incremental projection against a growing orthonormal basis is O(k*r*d) per new expert, which is linear in the number of existing experts. At N=10,000 and d=4096, r=16: each new expert requires ~655M FLOPs of projection, completing in <1s on CPU. This is fine.

3. **Groq pricing volatility.** The cost model is tightly coupled to Groq's batch pricing. A 3x price increase would still leave K1 passing ($0.477 + 2*$0.354 = $1.19, which would kill K1). The paper should note the pricing sensitivity more explicitly.

## Verdict

**PROCEED**

This is a well-executed accounting exercise. The core finding -- that SOLE is teacher-bound (74%) and SOLE-specific operations cost $0.00 -- is correct and actionable. K1 passes with comfortable margin. The K2 kill is a criterion design flaw, not a pipeline problem, and the paper's reinterpretation is honest and reasonable.

Minor issues to note but not blocking:

1. A5000 training time assumption underestimates Scenario C/D costs by ~15-30%, but directional conclusions hold.
2. The "supported" status is appropriate given one kill criterion technically fails. Do not upgrade to "proven."
3. The $0.061 "optimal" scenario stacks two unvalidated assumptions (8B quality + A5000 training parity). Present it as aspirational, not projected.

None of these are blocking. The experiment achieves its purpose: forensic cost transparency for the SOLE pipeline with clear identification of the dominant cost lever.
