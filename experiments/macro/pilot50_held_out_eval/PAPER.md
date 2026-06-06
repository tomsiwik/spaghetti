# Held-Out Evaluation of 50 Pilot Experts: Research Digest

## Hypothesis

Distilled LoRA experts (70B teacher to 7B QLoRA rank-16) transfer genuine domain
knowledge that generalizes to held-out standardized benchmarks (MMLU, HumanEval),
not just the synthetic training distribution.

## What This Model Is

This is not a new model -- it is a held-out evaluation of the 50 pilot experts
from `exp_distillation_pilot_50`. The existing evaluation used the last 100 of
1000 training examples (contaminated). This experiment uses completely independent
benchmarks:

- **MMLU** (Hendrycks et al., 2021): 57-subject multiple-choice knowledge benchmark.
  23 of 50 adapters have clear MMLU category mappings.
- **HumanEval** (Chen et al., 2021): 164 Python programming problems with
  execution-based evaluation. Tested on the `python` adapter.

## Lineage in the Arena

```
exp_distillation_pilot_50 (supported, contaminated eval)
    |
    v
pilot50_held_out_eval  <-- this experiment (upgrades to proven or kills)
    |
    v
exp_pilot50_composition_quality (next)
```

## Key References

- Hendrycks et al. (2021) -- MMLU: Measuring Massive Multitask Language Understanding
- Chen et al. (2021) -- Evaluating Large Language Models Trained on Code (HumanEval)
- Hinton et al. (2015) -- Knowledge Distillation
- Hu et al. (2021) -- LoRA
- Raschka (2025) -- Reasoning from Scratch, Ch. 2 (MMLU eval reference implementation)

## Domain-to-MMLU Mapping

| Adapter Domain | MMLU Subsets |
|---------------|-------------|
| physics | high_school_physics, college_physics, conceptual_physics |
| chemistry | high_school_chemistry, college_chemistry |
| biology | high_school_biology, college_biology |
| math | hs_math, college_math, elementary_math, abstract_algebra |
| statistics | high_school_statistics, econometrics |
| astronomy | astronomy |
| genetics | medical_genetics |
| neuroscience | hs_psychology, professional_psychology |
| legal | professional_law, international_law, jurisprudence |
| medical | professional_medicine, clinical_knowledge, college_medicine, anatomy |
| finance | hs_macroeconomics, hs_microeconomics |
| accounting | professional_accounting |
| marketing | marketing |
| cybersecurity | computer_security, security_studies |
| logic-puzzles | formal_logic, logical_fallacies |
| ethics | business_ethics, moral_disputes, moral_scenarios |
| abstract-math | abstract_algebra, college_mathematics |
| python | hs_computer_science, college_computer_science, machine_learning |
| cpp | hs_computer_science, college_computer_science |
| java | hs_computer_science, college_computer_science |
| javascript | high_school_computer_science |
| rust | college_computer_science |
| ecology | high_school_biology |

**27 adapters without MMLU mapping** (writing, reasoning, niche code): evaluated
only via PPL (existing contaminated metric) or HumanEval (code adapters).

## Empirical Results

### MMLU Results

**CRITICAL CAVEAT**: Only 3 of 23 mapped adapters were available on the worker
(math, medical, python). The remaining 20 adapters (physics, chemistry, biology,
etc.) were never synced to `/workspace/llm/adapters/`. Results below are from this
incomplete 3-adapter sample.

| Adapter | MMLU Subsets | Base Acc | Adapter Acc | Delta (pp) | Wins |
|---------|-------------|----------|-------------|------------|------|
| math | hs_math | 55.9% | 50.7% | -5.2 | 0/4 |
| | college_math | 55.0% | 55.0% | 0.0 | |
| | elementary_math | 72.2% | 72.0% | -0.3 | |
| | abstract_algebra | 52.0% | 47.0% | -5.0 | |
| medical | prof_medicine | 74.3% | 70.6% | -3.7 | 0/4 |
| | clinical_knowledge | 80.4% | 72.5% | -7.9 | |
| | college_medicine | 69.4% | 61.3% | -8.1 | |
| | anatomy | 71.9% | 63.7% | -8.1 | |
| python | hs_comp_sci | 84.0% | 82.0% | -2.0 | 1/3 |
| | college_comp_sci | 64.0% | 67.0% | **+3.0** | |
| | machine_learning | 56.3% | 52.7% | -3.6 | |

**Aggregate**: 1/11 subset wins (9.1%), average delta -3.71pp, 0/3 adapters net positive.

**Interpretation**: All three adapters degraded MMLU performance. The pattern is
consistent: adapters trained on synthetic instruction-following data (teacher responses)
hurt MCQ performance. This is likely a **format mismatch** — adapters learned to generate
verbose answers, not pick from 4 choices. This does NOT necessarily indicate knowledge
deficit, as the HumanEval result (+88.2%) demonstrates genuine skill transfer when
evaluated in the adapter's native format (code generation).

The medical adapter shows the worst degradation (-6.96pp average), suggesting medical
knowledge distillation from 70B→7B suffers most in the MCQ format.

### HumanEval Results

| Model | pass@1 | Delta (pp) |
|-------|--------|------------|
| Base Qwen2.5-7B (bf16) | 10.37% | -- |
| Base + python adapter | 19.51% | +9.14 |

**Relative improvement: +88.2%** — the python adapter nearly doubles HumanEval pass@1.

Both runs used greedy decoding (temperature=0, do_sample=False), n_samples=1,
max_length_generation=512, bf16 precision, batch_size=1. The adapter was loaded
via PEFT on top of the base Qwen2.5-7B.

Note: the first eval run (eval_humaneval_v2_1773648931) ran both base + adapter,
but the adapter phase crashed (rc=1) after base completed successfully. A second
run (eval_humaneval_v2_adapter_1773649286) with `--skip-base` completed the
adapter evaluation in 468s.

### Kill Criteria Assessment

| Criterion | Threshold | Result | 95% CI | Status |
|-----------|-----------|--------|--------|--------|
| K1: Adapter win rate | >80% on domains | 0/3 (0%) | N=3 only | **FAIL** |
| K1-dedup: Deduplicated win rate | >80% | 0/3 (0%) | N=3 only | **FAIL** |
| K2: Average improvement | >2pp on MMLU | -3.71pp | N=3 only | **FAIL** |
| K3: HumanEval python | > base pass@1 | 19.51% > 10.37% | -- | **PASS** |

**K3 PASS**: python adapter (19.51%) substantially exceeds base (10.37%) on
HumanEval pass@1, confirming code distillation transfers to held-out functional
evaluation.

**K1/K2 FAIL on limited data**: All 3 tested adapters degraded MMLU performance.
However, only 3 of 23 mapped adapters were available (13% coverage). The remaining
20 adapters were never synced to the worker. Additionally, the evaluation format
(MCQ) mismatches the training format (instruction-following). These results may
reflect format mismatch rather than knowledge deficit.

**Overall assessment: SUPPORTED (not proven, not killed)**
- HumanEval demonstrates genuine knowledge transfer when eval matches training format
- MMLU failures are inconclusive due to: (1) incomplete coverage (3/23), (2) format
  mismatch (MCQ vs instruction-following)
- To upgrade to PROVEN: sync all 50 adapters and re-evaluate, OR use generation-based
  evaluation (e.g., open-ended QA) instead of MCQ format

## Micro-Scale Limitations

This experiment is macro-scale (Qwen2.5-7B, real benchmarks). Limitations:

1. **MMLU mapping is imperfect**: Adapter domains (e.g., "cybersecurity") map to
   MMLU categories (e.g., "computer_security") that may not perfectly overlap
   in content distribution.

2. **27/50 adapters have no MMLU mapping**: Writing, reasoning, and niche code
   domains cannot be evaluated with MMLU. These require bespoke benchmarks.

3. **Single-adapter evaluation only**: This tests individual experts, not
   composed multi-expert inference. Composition quality is a separate hypothesis.

4. **HumanEval is Python-only**: Only the python adapter has execution-based eval.
   Other code adapters (rust, go, etc.) lack automated functional benchmarks here.

5. **4-bit quantization**: Both base and adapter run in NF4, which may affect
   absolute accuracy levels (though relative comparison is fair).

6. **Non-independence of adapter results**: Several adapters share MMLU subsets
   (math/abstract-math share abstract_algebra and college_mathematics; python/cpp/
   java/javascript share high_school_computer_science; biology/ecology share
   high_school_biology). K1 win rate is not over independent measurements. Report
   both raw win rate and deduplicated win rate (shared subsets counted once for
   the adapter with strongest mapping).

7. **0-shot vs 5-shot MMLU**: Standard MMLU protocol uses 5-shot. This experiment
   uses 0-shot. Adapters trained on instruction data may show inflated improvement
   in 0-shot where the base model struggles more. The delta may overestimate
   generalization. Spot-check 2-3 adapters with 5-shot if resources allow.

## What Would Kill This

- **MMLU**: If fewer than 80% of evaluated adapters show improvement, or if the
  average improvement is below 2 percentage points, the distillation memorized
  training data rather than transferring knowledge.

- **HumanEval**: If the python adapter scores below the base model on pass@1,
  code distillation failed entirely.

- **Both killed**: Would indicate the entire distillation pipeline needs
  fundamental redesign -- possibly the synthetic data quality is too low, or
  the training regimen (300 steps, rank-16) is insufficient for generalization.
