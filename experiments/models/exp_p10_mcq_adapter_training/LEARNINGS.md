# LEARNINGS: exp_p10_mcq_adapter_training

## Status: KILLED

## Core Finding

MCQ adapter trained without thinking mode completely suppresses thinking chains (0 chars vs 757,251 base) and catastrophically degrades generative quality (HumanEval 25% vs ~60% base). The best MMLU-Pro strategy is base + thinking (62.1%), no adapter needed.

## Why

Thinking mode requires entering a separate token channel (`<|channel>thought\n`) at the first generation position. LoRA trained on direct MCQ classification increases logit mass on answer tokens at position 1, which blocks thinking channel entry entirely — training-inference mode mismatch. MCQ loss gradient also propagates through full rank-6 ΔW=(α/r)BA perturbation across all 42 layers, poisoning the entire output distribution toward letter-dominated outputs. Separately: quantization error compounding is depth-dependent — O(ε^N) where N = reasoning steps. MMLU-Pro (N=1-3) benefits from thinking (+20.4pp); GPQA Diamond (N=10-20) does not (-1.0pp, Finding #528).

## Implications for Next Experiment

1. **Thinking adapters must be trained with `enable_thinking=True`** — any adapter intended for use with thinking mode must be trained in thinking mode or it will suppress the channel.
2. **62.1% closes exp_bench_mmlu_pro_thinking** — base+thinking is the MMLU-Pro ceiling under 4-bit (7.3pp below Google's 69.4%, gap is quantization not capability).
3. **MCQ loss is domain-specific** — only useful for non-thinking inference pipelines; net-negative for thinking-enabled Pierre v3.
