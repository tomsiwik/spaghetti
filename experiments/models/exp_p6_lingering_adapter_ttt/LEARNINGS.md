# LEARNINGS: exp_p6_lingering_adapter_ttt — KILLED

## Core Finding
TTT-style self-supervised (all-token) loss achieves only 40% factual recall vs P6.A0's 60%, with 50.6% backward pass overhead — all three kill criteria fail. The approach is structurally inferior to response-only supervised loss for factual LoRA adaptation.

## Why
Two independent structural failures: (1) The TTT "zero-cost" trick requires closed-form gradients, impossible when LoRA sits inside attention (softmax + GELU + RMSNorm are non-linear in Δy — Finding #491); (2) All-token loss distributes ~60% gradient signal to non-factual chat/prompt tokens, causing 2.5x signal dilution and topic contamination so severe the model generates "ZephyrFlow" in response to "capital of Japan." (arXiv:2407.04620)

## What P6.A0 Proved That TTT Broke
Response-only masking is not a convenience — it is the signal concentrator. The QA format ensures 100% of gradient lands on factual tokens. Removing it doesn't save cost; it costs quality and contaminates general knowledge.

## Impossibility Structure
TTT zero-cost for transformer LoRA is mathematically closed: ∂L/∂A requires backprop through the full non-linear chain. The minimum factual-recall cost is one backward pass (~24ms on M5 Pro). No hyperparameter tuning escapes this.

## Implications for Next Experiment
P6.A0 (response-only, 60%, 110ms/turn) remains the baseline. Future lingering-adapter work should explore: (a) factual precision — can response-only loss reach 80%+ with more turns or higher rank, (b) multi-domain composition — do two online adapters interfere, (c) distillation-style supervision using a teacher for harder facts.

## References
- arXiv:2407.04620 — TTT: Test-Time Training (confirms zero-cost requires linear inner model)
- Finding #491 — gradient path necessity for transformer LoRA
- exp_p6_lingering_adapter_online — P6.A0 baseline (60%, 110ms)
