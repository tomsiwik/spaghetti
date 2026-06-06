# LEARNINGS: exp_p0_two_stage_mcq

## Core Finding
Joint NTP+MCQ training is optimal under TT-LoRA r6; sequential (NTP→MCQ-only) achieves 33.5% vs 34.5% for mixed — effectively tied, not better. The 34.5% ceiling is TT rank-6 information capacity, not training procedure.

## Why
NTP and MCQ gradients are **synergistic**, not competitive. MCQ-only gradient provides ~2 bits/example (log2(4)) vs NTP's sequence_length × log2(V) bits. Without concurrent NTP signal, MCQ-only cannot reshape TT cores — it needs the high-bandwidth NTP signal as a knowledge scaffold. MCQ-only from scratch achieves only 15.0% (below base 30.5%), confirming NTP is load-bearing (+18.5pp). Refs: arXiv:2504.21190 (TT-LoRA), arXiv:2410.21228 (sequential LoRA intruder dims).

## Implications for Next Experiment
Training-procedure avenue is **closed**. To exceed the 35% ceiling, must increase TT rank (r=8/r=10) or use standard LoRA (Finding #521: 52.5% at r=8). Next step: benchmark TT-LoRA at higher ranks or shift to the P1 polar adapter / null-space isolation experiments.
