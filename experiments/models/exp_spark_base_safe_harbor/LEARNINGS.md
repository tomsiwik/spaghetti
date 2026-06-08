# LEARNINGS — exp_spark_base_safe_harbor

## Core Finding
The F#627 math-adapter interference signature (+22pp GSM8K / -12pp HumanEval) did not
reproduce on this harness: the adapter mildly helped code (+5pp HE) and hurt math (-7.5pp
GSM), inverting the assumed on/off-domain roles entirely.

## Why
The pre-registered KC was conjunctive and keyed to a specific drop/lift that did not exist;
with both denominators ≤0 the metrics are undefined by construction, making the KC
unmeetable regardless of gate behavior.

## Implication for the Next Experiment
A v2 targeting the gate mechanism must re-derive its KC around "net GSM8K+HE improvement
over FIXED" rather than drop-recovery/lift-retention framing; the discrete-Jaccard
base-fallback gate is real and effective (tau=0.20 raised GSM8K to 0.85 vs fixed 0.65 at
only 6.3% conflict rate) and warrants a fresh pre-registration with the correct premise.
