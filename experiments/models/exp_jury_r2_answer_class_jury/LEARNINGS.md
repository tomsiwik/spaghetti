# LEARNINGS — exp_jury_r2_answer_class_jury

**Core finding.** Jury-weighted self-consistency (3 LoRA jurors on one frozen gemma-4-e4b base)
beat SC(8) by +2.5pp (0.845 vs 0.820, n=200 GSM8K), but the pre-registered decorrelation clause
killed it: mean pairwise juror error-kappa 0.106 >= single-juror bootstrap self-kappa 0.064
(K2333, OR-kill). The jury is one verifier in a trenchcoat.

**Why.** All jurors share the same frozen base, so their misrankings are more correlated with
each other than one juror is with itself across split halves (score correlations 0.75-0.79).
The +2.5pp is a single-verifier *calibration/weighting* effect (any single juror's standardized
softmax weighting recovers +1.0-1.5pp), not multi-juror error suppression — and the SUPPORTED
bar (best single + 2pp) would have failed regardless (0.845 < 0.855).

**Implication for the next experiment.** Adapter diversity on a shared frozen base does NOT buy
verifier independence; the jury-decode ladder's load-bearing assumption is refuted, and this is
the third consecutive dead rung (R1 killed, R1v2 killed, R2 killed) with no v2 mechanism left —
this is the bet's obituary. Decorrelation, if pursued at all, would need a different *base*
(or external checkers like code execution), not more adapters.

**PIERRE-IMPACT:** shelved — killed finding; no code change to bet/jury-decode. The decode-time
jury (R4 ship target) is off: residual gains are single-verifier weighting that doesn't clear
support, and the decorrelation premise is dead on a shared frozen base.
