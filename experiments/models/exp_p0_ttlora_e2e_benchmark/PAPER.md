# PAPER: TT-LoRA Drop-In E2E Benchmark

## Summary

Drop-in replacement of standard LoRA (21.8 MB/adapter) with TT-LoRA (0.33 MB/adapter)
in the proven E2E pipeline from Finding #508. TT-LoRA rank-6 on v_proj+o_proj, 500 steps
per domain, 3 domains (math/code/medical).

**Result**: 3/4 kill criteria PASS. TT-LoRA retains 93% of LoRA quality on GSM8K and
87% on HumanEval, with 65x compression. Routing is perfectly independent (Theorem 2
confirmed). MedMCQA collapses to 21% (below random chance for 4-choice MCQ), revealing
a task-type sensitivity not captured by the uniform retention model.

## Prediction vs Measurement

| Metric | Predicted | Measured | Delta | Verdict |
|--------|-----------|----------|-------|---------|
| GSM8K | 60-65% | 68.0% | +3-8pp above prediction | BETTER than predicted |
| HumanEval | 50-55% | 55.0% | At prediction upper bound | MATCH |
| MedMCQA | 40-45% | 21.0% | -19 to -24pp | MISS (catastrophic) |
| Routing | 98.3% | 98.3% | 0.0pp | EXACT MATCH |
| Adapter size (each) | ~0.15 MB | 0.33 MB | +0.18 MB | 2x larger than predicted |
| 3-domain total | ~0.45 MB | 1.00 MB | +0.55 MB | safetensors overhead |
| Compression ratio | ~140x | 65x | ~2x less | Still massive compression |
| GSM8K retention | 84% (F#516) | 93% | +9pp | BETTER (o_proj helps) |
| HumanEval retention | ~84% | 87% | +3pp | BETTER |
| MedMCQA retention | ~84% | 42% | -42pp | CATASTROPHIC MISS |

## Kill Criteria Results

| ID | Criterion | Result | Status |
|----|-----------|--------|--------|
| K1426 | GSM8K >= 60% | 68.0% | **PASS** |
| K1427 | HumanEval >= 50% | 55.0% | **PASS** |
| K1428 | 3-domain total < 1 MB | 1.000 MB | **FAIL** (marginal) |
| K1429 | TF-IDF routing >= 95% | 98.3% | **PASS** |

## Training Details

| Domain | Steps | Final Loss | Time (s) | Adapter Size | Converged |
|--------|-------|------------|----------|-------------|-----------|
| Math | 500 | 0.465 | 875 | 325.6 KB | Yes |
| Code | 500 | 0.612 | 581 | 325.6 KB | Yes |
| Medical | 500 | 0.179 | 438 | 325.6 KB | Yes |

135,492 trainable params per adapter. TT-LoRA applied to v_proj+o_proj across all 42 layers
(35 standard layers: 2560->512/2048->2560, 7 wider layers: 2560->1024/4096->2560).

Total training time: ~31 min for 3 domains.

## Analysis

### What Worked

1. **Reasoning tasks retain well** (GSM8K 93%, HumanEval 87%). The TT-SVD error bound
   (Theorem 1) predicted ~84% retention based on Finding #516 (v_proj-only). Adding o_proj
   provided additional capacity that improved retention beyond prediction.

2. **Routing independence confirmed** (Theorem 2). TF-IDF routing at 98.3% — identical to
   baseline. This is a mathematical certainty (routing operates on input text, not weights).

3. **Compression is real**: 65x compression (65.4 MB -> 1.0 MB for 3 domains). Even accounting
   for safetensors overhead, the parameter data is 0.81 MB.

### What Failed

**MedMCQA catastrophic collapse (21% vs 50% baseline, vs 25% random chance).**

The medical adapter training loss converged to 0.179 (lowest of all 3 domains), yet the
adapter actively hurts MCQ performance. This reveals a fundamental task-type sensitivity:

- **Reasoning tasks** (GSM8K, HumanEval): require generating a chain of thought. The adapter
  modifies value/output projections to steer generation. Low-rank TT approximation preserves
  the dominant subspace needed for reasoning.

- **MCQ tasks** (MedMCQA): require selecting from fixed options via a single token. The adapter
  overfits to the training distribution's answer patterns. The TT compression amplifies this
  overfitting by collapsing the representation to rank-6, losing the fine-grained discrimination
  needed for 4-way classification.

This is consistent with Finding #516's observation that TT-LoRA retains quality proportionally
to task complexity: simple pattern matching (MCQ) is more sensitive to rank truncation than
multi-step reasoning.

### K1428 Marginal Failure

Total adapter size: 1,000,152 bytes = 1.000 MB. The criterion is "< 1 MB" (strict).
Actual parameter data: 135,492 params x 2 bytes x 3 = 812,952 bytes = 0.813 MB.
The remaining 187 KB is safetensors header/alignment overhead (~62 KB per file).
Using a single combined file or raw binary format would pass easily.

## Behavioral Assessment

**Does TT-LoRA produce useful text for reasoning tasks?** Yes. GSM8K 68% and HumanEval 55%
demonstrate that the compressed adapters steer generation effectively for math and code.

**Does it work as a drop-in replacement?** Partially. For reasoning tasks, yes. For MCQ/
classification tasks, no — the rank truncation destroys discriminative capacity.

**Does this advance toward the vision (25 domains at $2 each)?** Strongly yes for compression
economics. 325 KB per adapter means 25 domains = 8.1 MB total (vs 545 MB with standard LoRA).
But the MCQ failure means not all task types can use TT-LoRA — some domains may need
full-rank adapters.

## Conclusion

TT-LoRA is a viable drop-in replacement for reasoning-oriented adapters (93% GSM8K retention,
87% HumanEval retention, 65x compression). For classification/MCQ tasks, it fails catastrophically.
The uniform retention model (84% across tasks) is wrong — retention is task-type dependent.
Future work should either (a) use higher TT-rank for classification adapters, or (b) train
MCQ adapters with a different target projection set (q_proj/k_proj may preserve discriminative
features better than v_proj/o_proj).
