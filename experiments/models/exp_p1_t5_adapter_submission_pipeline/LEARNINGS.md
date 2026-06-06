# LEARNINGS.md — T5.3: User Adapter Submission Pipeline

## Core Finding
Full submit→validate→integrate→serve pipeline runs in 23.8s end-to-end (12.6× faster than
the 300s goal), with 100% routing accuracy and 100% behavioral quality preserved.

## Why It Works
IDF uniqueness guarantees user-token isolation (Theorem 2): a personal token with
IDF = log(N/1) dominates domain tokens at log(N/n_d), making TF-IDF cosine ≈ 1.0 for the
correct adapter and ≈ 0 for all others. Interference is structurally impossible when
personal vocabulary shares zero overlap with domain corpora.

## Implications for Next Experiment
T6.1 (adapter clustering) can now operate on accumulated user adapters. The 23.8s pipeline
is fast enough that T6 can treat submission as near-instantaneous. Watch for routing
degradation when N_personal grows — T4.1 showed N=25 drops to 86.1%, so clustering is
needed before that scale is reached.
