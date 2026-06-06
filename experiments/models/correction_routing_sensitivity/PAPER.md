# Correction Routing Sensitivity: Research Digest

## Hypothesis

The correction signal quality decision tree is robust to parameter perturbation:
routing decisions (which correction source to use per domain) do not flip under
plausible variation in teacher accuracy and problem difficulty.

**Falsifiable:**
- K1: Decision tree flips for >50% of domains under +/-10% parameter perturbation.
- K2: No robust region exists where execution > teacher ordering holds.

## What This Experiment Is

A systematic sensitivity analysis of the parent experiment (correction_signal_quality),
which built a decision tree routing corrections to human, teacher (70B), or execution
feedback sources. The adversarial review identified that:

1. The K1 margin (teacher error 19.6% vs 20% threshold) is 0.4pp -- within a 95% CI
   that spans both sides of the threshold.
2. Difficulty distributions are hand-specified, not derived from data.
3. The error-only K1 metric ignores degenerate corrections (6.5% additional harmful rate).

This experiment addresses all three concerns by:

- Sweeping teacher_hard_accuracy over [0.60, 0.80] (21 points)
- Sweeping difficulty_mean perturbation over [-0.10, +0.10] (11 points)
- Running a full 2D sweep (231 grid points, 10 seeds each)
- Computing closed-form analytical breakpoints via Brent's method
- Introducing harmful rate (wrong + degenerate) as alternative K1 metric

## Lineage in the Arena

```
micro/correction_signal_quality (supported: decision tree for evolve)
  |
  +-- THIS: micro/correction_routing_sensitivity (sensitivity analysis)
  |
  +-- exp_teacher_correction_empirical_validation (macro: real teacher accuracy)
```

## Key References

- Parent experiment: micro/models/correction_signal_quality/
- Adversarial review: micro/models/correction_signal_quality/REVIEW-adversarial.md
- Lee et al. 2023, RLAIF -- calibration source for teacher accuracy endpoints
- Brent 1973, "Algorithms for Minimization without Derivatives" -- root-finding for breakpoints

## Empirical Results

### K1: Decision Tree Flip Rate

| Perturbation | Teacher Error | Teacher Harmful | Flips | Flip Rate |
|:---:|:---:|:---:|:---:|:---:|
| baseline (h=0.70, dp=0) | 19.6% | 26.0% | 0/6 | 0% |
| teacher -10% (h=0.63) | 22.6% | 28.9% | 0/6 | 0% |
| teacher +10% (h=0.77) | 16.1% | 22.9% | 0/6 | 0% |
| difficulty -10% (dp=-0.10) | 17.2% | 23.8% | 0/6 | 0% |
| difficulty +10% (dp=+0.10) | 21.9% | 28.2% | 0/6 | 0% |
| worst case (h=0.63, dp=+0.10) | 25.2% | 31.3% | 0/6 | 0% |

**K1 verdict: SURVIVES decisively.** Zero flips across the entire parameter space,
including the worst-case corner (teacher 10% worse, problems 10% harder). The full
2D sweep (231 points) produces zero flips.

**Why:** The decision tree is driven by the 10x cost gap between execution ($0.0001)
and teacher ($0.001), and the categorical applicability constraint (execution only
works for code). These structural features dominate any quality-rate perturbation.
A flip would require execution to become ~25x worse in quality improvement per
correction, which is far outside any plausible parameter range.

### K2: Execution > Teacher Region

| Metric | Value |
|--------|:---:|
| Grid points where execution wins all code domains | 231/231 (100%) |
| Execution wins at worst corner (h=0.60, dp=+0.10) | Yes |
| Minimum h for execution dominance | 0.60 (lowest tested) |

**K2 verdict: SURVIVES decisively.** Execution dominates teacher for all code
domains across 100% of the swept parameter space. The ordering is robust because
execution's cost advantage (10x) overwhelms any accuracy disadvantage.

### Key Finding: Harmful Rate Redefines K1 Boundary

The most important result is not about flip rates but about the parent
experiment's K1 threshold interpretation.

| Metric | Aggregate | Threshold | Verdict |
|--------|:---:|:---:|:---:|
| Error rate (wrong only) | 19.6% | 20% | PASSES (margin 0.4pp) |
| Harmful rate (wrong + degenerate) | 26.0% | 20% | **FAILS by 6.0pp** |

The original K1 asked "are teacher corrections wrong >20% of the time?" and
found 19.6% -- barely passing. But the adversarial review correctly noted that
degenerate corrections (technically correct but harmful) should also count.
Adding the 6.5% degeneracy rate gives 26.0% harmful rate, which exceeds 20%
by a wide margin across ALL domains.

### Analytical Breakpoints (Closed-Form)

| Breakpoint | Value | Interpretation |
|:---:|:---:|:---|
| h*_error (aggregate) | 0.6644 | Teacher hard acc where avg error = 20% |
| h*_harmful (aggregate) | 0.8206 | Teacher hard acc where avg harmful = 20% |
| K2 coverage threshold | 0.6667 | Test coverage below which execution degeneracy > 10% |

**The harmful breakpoint (0.8206) is the actionable threshold.** To achieve
< 20% harmful rate from teacher corrections, teacher hard accuracy must exceed
0.82 -- substantially above the 0.70 baseline and above published RLAIF
agreement rates (~0.75 on hard tasks).

Per-domain harmful breakpoints:

| Domain | h* (error < 20%) | h* (harmful < 20%) |
|--------|:---:|:---:|
| python_basics | n/a (always passes) | 0.6788 |
| algorithm_design | 0.6685 | 0.8226 |
| systems_programming | 0.7341 | 0.8459 |
| creative_writing | 0.6087 | 0.8007 |
| logical_reasoning | 0.6919 | 0.8309 |
| medical_qa | 0.7175 | 0.8401 |

Systems_programming requires the highest teacher accuracy (h > 0.846) due to
its high baseline difficulty (mu = 0.75).

## Implications for SOLE Evolve Phase

1. **The routing tree is stable.** The code/non-code split and the execution-first
   cascade are robust features, not artifacts of parameter tuning. This is good
   news for production: the routing logic does not need continuous recalibration.

2. **The harmful rate is the real concern.** Teacher corrections at baseline
   produce 26% harmful outputs (wrong + degenerate). The Evolve phase must either:
   - (a) Use a confidence filter on teacher corrections (reject low-confidence ones)
   - (b) Accept that ~1 in 4 teacher corrections degrades the expert
   - (c) Increase teacher hard accuracy above 0.82 (e.g., use 70B with chain-of-thought)

3. **Execution feedback is unconditionally preferred for code.** Even at teacher
   hard accuracy = 0.80 (much better than literature suggests), execution's cost
   advantage makes it dominant. The only caveat is systems_programming with low
   test coverage (K2 killed at coverage < 0.667).

## Micro-Scale Limitations

1. **Still a simulation.** The 0% flip rate is a consequence of the structural
   cost gap (10x) and categorical applicability constraint. If real-world costs
   converge (e.g., teacher becomes as cheap as execution), the flip rate would
   increase. This experiment does not test cost perturbation.

2. **No new correction sources tested.** The tree cannot flip to a source that
   does not exist. Real production may introduce new sources (e.g., self-play,
   crowdsourcing) that could change the routing.

3. **Difficulty distributions remain hand-specified.** While we swept perturbations,
   the base distributions are not data-derived. The sensitivity analysis shows
   robustness to mean shifts but does not test distributional shape changes
   (e.g., bimodal, heavy-tailed).

4. **Degeneracy rate modeled as constant.** Real degeneracy may correlate with
   difficulty (harder problems have more subtle failure modes). This would
   increase the harmful rate further.

## What Would Kill This

**At micro scale (this experiment):**
- K1 (>50% flip rate): SURVIVES with 0% flips. Would require a fundamentally
  different cost structure to kill (e.g., teacher becomes 100x cheaper).
- K2 (no robust exec>teacher region): SURVIVES with 100% of parameter space.
  Would require execution accuracy to drop below ~0.50 to kill.

**At macro scale (what needs validation):**
- If actual teacher hard accuracy is measured at < 0.66 on pilot-50 outputs,
  even the error-only K1 is killed. The exp_teacher_correction_empirical_validation
  experiment will resolve this.
- If the real harmful rate exceeds 30% (plausible given the 26% simulation
  estimate plus unmodeled difficulty-degeneracy correlation), the Evolve phase
  needs mandatory confidence filtering, not optional escalation.

**What this experiment DOES establish:**
1. The routing tree is structurally stable (cost-driven, not accuracy-driven).
2. Closed-form breakpoints: h*_error = 0.6644, h*_harmful = 0.8206, c*_k2 = 0.667.
3. The harmful rate metric changes the K1 verdict from "barely passes" to "clearly fails."
4. The parent experiment's decision tree is correct in routing, but its K1 assessment
   was misleadingly optimistic due to excluding degenerate corrections from the error count.
