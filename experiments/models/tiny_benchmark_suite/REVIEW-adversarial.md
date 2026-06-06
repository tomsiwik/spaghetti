# Adversarial Review: exp_tiny_benchmark_suite

## Verdict: PROCEED

This is a clean kill with honest reporting. The negative result is well-characterized.

## Data Verification

Numbers in PAPER.md match results.json:
- Base MMLU: 29/50 = 58.0% (matches)
- Base GSM8K: 16/30 = 53.3% (matches)
- Base HumanEval: 9/15 = 60.0% (matches)
- All adapter configurations match cell-by-cell

## Kill Status: Correct

K820 FAIL is the honest assessment. No adapter configuration improves any benchmark.
The PAPER.md correctly notes this is a category mismatch (domain adapters tested on
general benchmarks) rather than an architectural failure.

## Blocking Fixes: None

## Non-Blocking

1. **Finding #262 contradiction needs investigation.** NTP math adapter showed +10pp GSM8K
   in F#262 but -17pp here. The paper lists 3 plausible explanations but doesn't determine
   which is causal. This matters for the project: if RuntimeLoRA is worse than TernaryLoRA
   for benchmark tasks, the serving architecture choice has consequences.

2. **Sample sizes acknowledged but should be MORE prominent.** With 95% CI width of 26-44pp,
   essentially NO comparison in this experiment reaches significance. The table should note
   this more prominently rather than reporting deltas as if they're meaningful.

3. **MMLU base score discrepancy.** Finding #213 reported base MMLU at 38%. This experiment
   measures 58%. This 20pp difference needs explanation (likely different MMLU subset or
   evaluation method).

## Advisory

1. The "category mismatch" framing is fair but could be stronger. These benchmarks WERE
   listed in the experiment notes as targets. The honest conclusion is: the adapters don't
   do what we hoped on general benchmarks, and that's OK because their purpose is domain
   generation.
2. Total runtime 1010s (17min) is efficient for a benchmark suite.
