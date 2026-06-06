# LEARNINGS — exp_p0_vproj_quality_floor

**Status:** KILLED — Finding #506

## Core Finding

Training v_proj+o_proj adapters on HuggingFace task-completion datasets (CodeAlpaca, MedMCQA, Finance-Alpaca) makes behavioral quality **worse** than P8's 10 handcoded examples in 4/5 domains. Only math (GSM8K +30%) survived because step-by-step solutions naturally contain mathematical prose.

## Why

v_proj+o_proj directly modifies the output token distribution (Finding #504). Training on code/MCQ/instruction data shifts the distribution toward task-completion tokens (code blocks, letter answers, arithmetic) — the opposite of explanatory domain vocabulary. The failure is geometrically inevitable: you cannot increase domain vocabulary density by training on a distribution sparse in domain vocabulary.

**Data distribution alignment > data quantity.** Citing Finding #149 (saturation at N=200-500) to justify scaling was the wrong frame — saturation applies within a fixed distribution, not across mismatched distributions.

## Implications for Next Experiment

Two structurally different paths forward:

1. **Generation quality adapters:** Train on vocabulary-dense explanatory data (P8-style, 30-50 curated examples per domain). These serve the VISION goal of behavioral quality.
2. **Benchmark accuracy adapters:** Train on HuggingFace task datasets, evaluate on matching benchmarks (GSM8K accuracy, HumanEval pass@1). These serve P0 benchmark goals but require aligned evaluation.

The P0 goal of "generation quality AND benchmarks" requires domain-aware data curation — the adapters are not interchangeable. Next experiment should pick ONE path and define the eval accordingly.
