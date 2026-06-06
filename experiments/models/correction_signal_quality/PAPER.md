# Correction Signal Quality: Research Digest

## Hypothesis

Different correction sources (human, teacher model, execution feedback) have
quantifiably different accuracy/cost/degeneracy profiles, enabling an optimal
routing decision tree for the SOLE Evolve phase.

**Falsifiable:**
- K1: If teacher corrections are wrong >20% of the time (average), the Evolve
  loop cannot rely on automated teacher feedback.
- K2: If execution feedback produces degenerate solutions (>10% of accepted
  solutions are technically correct but useless), execution-based self-learning
  is unsafe without additional filtering.

## What This Experiment Is

A Monte Carlo simulation study comparing three correction sources for expert
evolution in the SOLE architecture. For each source, we model:

1. **Correction accuracy** as a function of problem difficulty, calibrated from
   published RLAIF/feedback literature
2. **Degeneracy rates** (corrections that pass validation but are harmful)
3. **Cost per correction** at production scale
4. **Expert quality trajectories** under 200 corrections per source per domain

Six domains spanning code (3 types) and non-code (3 types) are tested across
10 Monte Carlo seeds. This is explicitly a simulation study -- no actual model
training occurs. The value is in establishing calibrated expectations and
identifying which correction source to use where, before building the actual
self-learning pipeline.

### Correction Source Parameters (from literature)

| Source | Base Accuracy | Hard Accuracy | Degeneracy | Cost/Correction |
|--------|:---:|:---:|:---:|:---:|
| Human | 97% | 90% | 2% | $2.00 |
| Teacher (70B) | 92% | 70% | 8% | $0.001 |
| Execution | 99% | 85% | coverage-dependent | $0.0001 |

Calibration sources:
- **Human:** Inter-annotator agreement in NLP benchmarks (Cohen's kappa ~0.8-0.9)
- **Teacher:** RLAIF (Lee et al., 2023): ~88% agreement with human preferences.
  Self-Refine (Madaan et al., 2023): saturation at ~85%. Positional bias (Wang
  et al., 2023): 10-15% error from ordering effects.
- **Execution:** Test suite pass rates from SWE-bench (~30-40% at solving, but
  100% when passing tests). Degeneracy modeled as (1-coverage) * gamma.

## Lineage in the Arena

```
micro/answer_conditioned_scoring (proven: PPL-based quality signal)
  |
  +-- THIS: micro/correction_signal_quality (which signals to GENERATE corrections)
  |
  +-- exp_execution_based_self_learning (BLOCKED: needs this result)
  |
  +-- exp_model_collapse_detection (BLOCKED: needs self-learning first)
```

## Key References

- Lee et al. 2023, "RLAIF: Scaling RL from Human Feedback with AI Feedback" --
  RLAIF achieves comparable performance to RLHF; ~88% agreement with human labels
- Bai et al. 2022, "Constitutional AI" -- AI self-correction via principles;
  demonstrates teacher-to-student feedback quality
- Wang et al. 2023, "Large Language Models are not Fair Evaluators" -- positional
  bias in LLM-as-judge; 10-15% error from ordering effects
- Madaan et al. 2023, "Self-Refine" -- iterative self-improvement saturates at
  ~85% on many tasks

## Empirical Results

### K1: Teacher Correction Error Rate

| Domain | Teacher Error Rate | vs 20% Threshold |
|--------|:---:|:---:|
| python_basics | 13.0% | PASS |
| algorithm_design | 19.8% | PASS (marginal) |
| creative_writing | 16.6% | PASS |
| logical_reasoning | 22.1% | **EXCEEDS** |
| medical_qa | 22.1% | **EXCEEDS** |
| systems_programming | 24.1% | **EXCEEDS** |
| **Average** | **19.6%** | **PASS (barely)** |

**K1 verdict: SURVIVES on average (19.6% < 20%) but EXCEEDS on 3 of 6 domains.**

The aggregate kill criterion narrowly survives, but individual high-difficulty
domains (systems_programming, logical_reasoning, medical_qa) all exceed 20%.
This means teacher corrections should NOT be used as the sole correction source
for hard domains without a confidence filter or escalation mechanism.

### K2: Execution Feedback Degeneracy

| Domain | Degeneracy Rate | vs 10% Threshold |
|--------|:---:|:---:|
| python_basics | 1.3% | PASS |
| algorithm_design | 5.4% | PASS |
| systems_programming | 11.3% | **EXCEEDS** |

**K2 verdict: KILLED for systems_programming (11.3% > 10%).**

Systems programming with test coverage ~60% produces degenerate solutions at
11.3% -- exceeding the 10% threshold. This confirms the hypothesis notes'
concern: execution feedback is safe for well-tested code but dangerous for
complex systems with integration gaps.

### Quality Improvement per Source

| Domain | Human | Teacher | Execution | Best Source |
|--------|:---:|:---:|:---:|:---:|
| python_basics | +0.285 | +0.230 | **+0.292** | execution |
| algorithm_design | **+0.425** | +0.300 | +0.406 | human |
| systems_programming | **+0.475** | +0.298 | +0.424 | human |
| creative_writing | **+0.379** | +0.287 | N/A | human |
| logical_reasoning | **+0.426** | +0.271 | N/A | human |
| medical_qa | **+0.474** | +0.330 | N/A | human |

Human corrections produce the highest absolute quality improvement in 5/6 domains.
Execution feedback slightly outperforms human on python_basics (0.292 vs 0.285)
due to higher accuracy on easy problems and near-zero cost enabling more iterations.

### Cost-Effectiveness (EIR: quality improvement per dollar)

| Domain | Human | Teacher | Execution |
|--------|:---:|:---:|:---:|
| python_basics | 0.001 | 1.15 | **14.6** |
| algorithm_design | 0.001 | 1.50 | **20.3** |
| systems_programming | 0.001 | 1.49 | **21.2** |
| creative_writing | 0.001 | **1.43** | N/A |
| logical_reasoning | 0.001 | **1.35** | N/A |
| medical_qa | 0.001 | **1.65** | N/A |

Execution feedback is 10-15x more cost-effective than teacher corrections for
code domains. Teacher corrections are 1000x more cost-effective than human
corrections for all domains. Human corrections are only justified when accuracy
requirements exceed automated capabilities.

### Optimal Routing Decision Tree

```
Query arrives for domain D
  |
  +-- Is D a code domain?
  |     |
  |     +-- test_coverage >= 0.8?
  |     |     |
  |     |     YES -> execution feedback (EIR 14-21x)
  |     |     NO  -> teacher_70b + execution as filter
  |     |            (run tests first, escalate fails to teacher)
  |     |
  |     +-- Is difficulty > 0.7?
  |           |
  |           YES -> teacher + human escalation for low-confidence
  |           NO  -> teacher alone (error < 15%)
  |
  +-- Is D a non-code domain?
        |
        +-- Is accuracy requirement > 95%?
        |     |
        |     YES -> human (only source meeting threshold)
        |     NO  -> teacher_70b (EIR 1.3-1.7 quality/$)
        |
        +-- Is difficulty > 0.7?
              |
              YES -> teacher + human escalation (teacher error > 20%)
              NO  -> teacher alone
```

## Micro-Scale Limitations

1. **Simulation, not empirical.** Error rates and degeneracy are modeled from
   literature, not measured from actual SOLE expert corrections. Real teacher
   error patterns may differ from sigmoid difficulty curves.

2. **No interaction effects.** The model assumes corrections are independent.
   In practice, a wrong correction followed by a correct one may not cancel
   out -- the expert may have learned a bad pattern that persists.

3. **Degeneracy model is simple.** Real degenerate solutions (e.g., code that
   passes tests by hardcoding outputs) have complex structure. Our
   coverage-proportional model may underestimate degeneracy for certain problem
   types.

4. **Teacher calibration is from general RLAIF.** Error rates for specialized
   code correction may differ from general helpfulness/harmlessness evaluation.
   70B models may be better at code (higher base accuracy) or worse at edge
   cases (lower hard accuracy) than our calibration assumes.

5. **No temporal dynamics.** Expert quality is assumed to update instantaneously.
   In practice, fine-tuning on corrections takes 50-100 steps and may not fully
   absorb the correction signal.

6. **Cost model ignores latency.** Human corrections take hours-days; teacher
   corrections take seconds; execution is instant. Time-to-correction matters
   for the evolution loop speed.

## What Would Kill This

**At micro scale (this experiment):**
- K1 (teacher error >20% average): SURVIVES at 19.6%, but margin is razor-thin.
  A more realistic difficulty distribution (biased toward hard problems) would
  likely kill K1.
- K2 (execution degeneracy >10%): KILLED for systems_programming (11.3%).
  This is a genuine finding: low test coverage makes execution feedback unsafe.

**At macro scale (what would need validation):**
- Measure actual teacher correction accuracy on SOLE pilot-50 expert outputs.
  If real error rate exceeds 20% on hard domains, the cascade to human
  escalation becomes mandatory (not optional).
- Measure actual execution feedback degeneracy on HumanEval/MBPP with
  production test suites. If real degeneracy exceeds 10% on well-tested
  problems, execution feedback needs a secondary quality check.
- Test the interaction hypothesis: do 3 consecutive wrong corrections degrade
  the expert more than 3x a single wrong correction? (compound error effects)
- Test correction latency impact: does the evolution loop converge differently
  with instant (execution) vs delayed (teacher) vs slow (human) feedback?

**What this experiment DOES establish:**
1. A calibrated decision tree for correction source routing
2. The cost-effectiveness ordering: execution >> teacher >> human
3. A concrete failure mode (systems programming degeneracy) that needs mitigation
4. The K1 margin is dangerously thin -- teacher corrections are at the boundary
   of acceptable error rates for hard domains
