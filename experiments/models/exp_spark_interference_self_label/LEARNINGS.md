# LEARNINGS — exp_spark_interference_self_label

## Core finding
The top-1 logit-shift sign (adapter vs frozen base, averaged over base-greedy tokens) is an inverted
domain signal: mean AUROC 0.278 (threshold 0.70, killed), with code and medical AUROC at 0.047 and
0.049 — adapters suppress the base's own greedy margin *most* on their own training domain.

## Why
An instruction-tuned adapter learns to redirect probability mass away from the frozen base's greedy
continuation precisely where it has domain-specific knowledge; the margin shift therefore measures
adapter-vs-base disagreement, not domain ownership. The premise "base greedy token ≈ correct
on-domain continuation" is false for a 4-bit IT base teacher-forced on its own trajectory.

## Implication for the next experiment
Training-free routing via logit-shift sign is a dead approach for this adapter class; any domain
detector must either (a) use a learned lightweight head on adapter activations, or (b) exploit a
signal that is not anchored to the frozen base's greedy path (e.g., adapter output norm or
task-conditioned confidence under the adapter alone, not the delta).
