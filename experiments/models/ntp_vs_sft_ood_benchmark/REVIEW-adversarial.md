# Peer Review: NTP vs SFT Adapter OOD Benchmark (Re-Review)

**Previous verdict:** REVISE (5 required fixes)
**This review:** Re-review after revision

## Experiment Type
Guided exploration (Type 2)

## Fix Verification

| Fix | Required | Applied? | Notes |
|-----|----------|----------|-------|
| 1. K1 re-evaluated with all 7 OOD benchmarks | Remove mmlu_legal exclusion | YES | PAPER.md now counts 3/7 benchmarks degraded >=5pp, K1 marked FAIL. Exclusion explicitly called out as unjustified. |
| 2. Downgrade to PROVISIONAL | 2/3 predictions missed | YES | Status is PROVISIONAL throughout. "Empirical observation awaiting formal proof." |
| 3. Rename "Theorem 1 (Informal)" to "Hypothesis 1" | No theorem exists | YES | MATH.md line 52 now "Hypothesis 1." Caveat added (line 58) that Gunasekar conditions do not apply. |
| 4. Address MMLU math anomaly | NTP -20pp vs SFT -10pp contradicts theory | YES | PAPER.md Section 4 discusses the anomaly, provides Fisher p=0.74 (not significant), scopes hypothesis to reasoning tasks only. |
| 5. Add Fisher exact p-values | Most differences are noise | YES | Prediction table now includes p-values. Only NTP vs SFT GSM8K gap (p=0.003) is significant. |

All 5 fixes properly applied.

## Hack Detector
- Fix count: 0 (diagnostic experiment, no mechanisms added)
- Is MATH.md a proof or a description? Description with a clearly labeled hypothesis. Appropriate for Type 2 guided exploration.
- Metric used as evidence: OOD accuracy delta (pp) with Fisher exact p-values. The 30pp GSM8K gap at p=0.003 is the only statistically robust finding.
- Kill criteria source: Derived from MATH.md predictions (P1-P3 map to K1). K1 now evaluated honestly on all benchmarks.

## Self-Test Audit

1. **Impossibility property:** "NTP training regularizes the adapter to produce small perturbations on instruction-like inputs." Still a mechanism description rather than an impossibility property, but Self-Test answer 1 now honestly says "this was only confirmed for reasoning tasks, not universally. The hypothesis is partially supported." Acceptable for provisional status.

2. **Cited theorems:** Gunasekar et al. (2017). MATH.md now explicitly states (line 58): "this theorem applies to linear models with squared loss, not to multi-layer transformers with cross-entropy loss. It provides motivation, not proof." This is honest. PASS.

3. **Predicted numbers:** Specific and falsifiable. PASS.

4. **Falsification condition:** "NTP degrades >=5pp on 3+ domains." This condition was MET (K1 FAILS), and the paper honestly reports this. PASS.

5. **Hyperparameter count:** 0. PASS.

6. **Hack check:** No fixes stacked. PASS.

## Mathematical Soundness

MATH.md correctly identifies this as a guided exploration within the proven framework of adapter composition at optimal scales (Finding #249). The unknown (training objective effect on OOD behavior) is precisely stated. The mathematical framework is a plausible perturbation analysis, not a proof, and the revised MATH.md is honest about this distinction.

The key equation (||Delta_W_NTP * h_ood|| / ||Delta_W_NTP|| <= ||Delta_W_SFT * h_ood|| / ||Delta_W_SFT||) is labeled as a hypothesis, not a theorem. The conditions under which the cited Gunasekar result would apply are explicitly flagged as violated. This is the correct framing for Type 2.

No mathematical errors in the framework itself -- the perturbation analysis is standard and the gradient flow argument (SFT gradient only through response tokens, NTP through all tokens) is correct as a description of the training dynamics.

## Prediction vs Measurement

PAPER.md contains a properly formatted prediction-vs-measurement table with Fisher p-values.

| Prediction | Predicted | Measured | p-value | Status |
|-----------|-----------|----------|---------|--------|
| P1: GSM8K | <= 2pp degrad | +10pp improvement | p=0.003 (NTP vs SFT) | EXCEEDED |
| P2: Code gen | <= 2pp degrad | -10pp degrad | p=1.00 | MISS |
| P3: MMLU | <= 3pp degrad | -6pp degrad | p=0.47 | MISS |
| P4: In-dist math | >= 60% | 80% | -- | PASS |
| P5: In-dist code | >= 40% | 75% | -- | PASS |

2/3 OOD predictions missed. K1 FAILS. The paper is honest about all of this. The single robust finding is the 30pp NTP-SFT gap on GSM8K (p=0.003).

## Remaining Concerns (Non-Blocking)

1. **results.json still shows old K1 evaluation** (2/5 benchmarks, "pass"). The paper corrects this, but the raw results file is inconsistent. Minor bookkeeping issue.

2. **The three-mechanism decomposition** (reasoning/format/knowledge in PAPER.md Section 3) is post-hoc rationalization, not a predicted taxonomy. The paper presents it as a "key finding" but it was not predicted by MATH.md. This is acceptable for a provisional finding but should not be cited as if it were theory-derived.

3. **Grassmannian skeleton confound** remains: both adapter types use A matrices computed from NTP training. SFT adapters operate in an NTP-optimized subspace. This is acknowledged in Limitations (point 2) but worth emphasizing -- the comparison is not fully symmetric.

4. **Code gen at n=10** is uninformative. One sample difference. The paper acknowledges this (Limitation 3) but still reports it in the prediction table. Fair enough for completeness.

## Novelty Assessment

The NTP vs SFT OOD comparison at the adapter composition level is a reasonable micro-scale diagnostic. The scoped conclusion (NTP helps reasoning, not knowledge or formatting) is more nuanced and honest than the original claim. The three-way task decomposition, while post-hoc, provides a useful framework for future experiments.

Prior art: LoRA Land (arXiv:2405.00732) documents poor cross-task transfer but does not compare NTP vs SFT training objectives on OOD benchmarks specifically. The delta here is the controlled comparison isolating training objective as a variable.

## Macro-Scale Risks (advisory)

1. GSM8K improvement may be data contamination (NTP math data resembling GSM8K format). Paper flags this.
2. Grassmannian skeleton asymmetry may disappear with SFT-derived skeletons.
3. At larger parameter count, the perturbation analysis may not hold -- nonlinear interactions between adapter and base model increase.

## Verdict

**PROCEED**

All 5 required fixes from the original review have been properly applied. The experiment is now honest about its limitations:

- Status correctly PROVISIONAL (not supported or conclusive)
- K1 correctly evaluated as FAIL with all benchmarks included
- Hypothesis correctly labeled as hypothesis, not theorem
- Statistical significance properly reported with p-values
- MMLU math anomaly discussed and the hypothesis scoped to reasoning tasks

The single robust finding -- a 30pp NTP-SFT gap on GSM8K at p=0.003 -- is a legitimate empirical observation worth recording as provisional. The paper does not overclaim. The three-mechanism decomposition (reasoning/format/knowledge) provides useful direction for future experiments, acknowledged as post-hoc.

This is what a well-executed Type 2 guided exploration with mixed results should look like: honest about what worked, what failed, and why the status is provisional.
