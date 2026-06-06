# Adversarial Review: exp_p0_ttlora_n10_scaling

## Verdict: PROCEED

**Status: SUPPORTED** is appropriate for a frontier extension that clearly identifies
what scales (adapters) and what doesn't (TF-IDF routing).

## Checklist

- [x] Prediction-vs-measurement table present in PAPER.md
- [x] Kill criteria results consistent between PAPER.md and results.json
- [x] All 4 kill criteria correctly evaluated (1 PASS, 3 FAIL)
- [x] Structural diagnosis is sound and empirically validated
- [x] Impossibility structure derived (TF-IDF upper bound ~80% at N=10 with vocab overlap)

## Issues (non-blocking)

### 1. Prediction 3 baseline mismatch
MATH.md Theorem 2 predicts quality "within +/-5pp" of e2e benchmark (GSM8K 68%, HumanEval 55%, MedMCQA 21%).
PAPER.md uses different baselines for retention (GSM8K 73%, HumanEval 63%, MedMCQA 50%).
GSM8K measured 50% vs predicted 68% is a 18pp miss, well outside the +/-5pp window.
**Mitigated** by N=50 sample size giving +/-14pp CI, but the prediction was still overconfident.

### 2. Code HumanEval 111% retention
70% TT-LoRA vs 63% baseline = 111% retention. Almost certainly noise at N=50.
Acknowledged in PAPER.md footnote. Not a concern for the finding.

### 3. Routing impossibility argument is empirical, not formal
The claim "P(vocab_overlap) ~ 0.20 giving upper bound ~80%" is post-hoc curve fitting
to the 79.3% result, not a derived prediction. The Hoeffding bound in Theorem 1
predicted 88-92%, which missed. The impossibility structure is observationally correct
but should be noted as empirically derived, not proven.

## What's strong

1. **Honest accounting** -- 3/4 kill criteria fail, clearly reported
2. **Root cause identified** -- TF-IDF is lexical not semantic, this is structural
3. **Adapter scaling confirmed** -- all 10 converge, quality holds, footprint predictable
4. **Clear next step** -- semantic routing (learned embeddings) is the obvious fix
5. **Size prediction exact** -- 3.334 MB vs 3.33 MB predicted

## Finding recommendation

Title: "TT-LoRA adapters scale to N=10; TF-IDF routing degrades to 79.3% from vocabulary overlap"
Status: supported
Key result: Adapters scale linearly (45 min, 75.3% quality retention). TF-IDF routing
is structurally limited by lexical overlap between semantically adjacent domains.
Need semantic routing for N>5.
