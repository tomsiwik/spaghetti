# LEARNINGS: P11.A1 — LIMO Reasoning SFT (Design Phase)

## Core Finding (Design)
LIMO (arXiv:2502.03387) selects "barely solvable" traces (1-3/32 attempts = p_x ≈ 3-9%) which
sit at the capability boundary where gradient signal p_x(1-p_x) is near-maximal AND valid
reasoning traces exist (p_x > 0). This is mathematically stronger curation than s1K's
difficulty+diversity+quality filter.

## Why
Teaching dimension theory (Goldman & Kearns 1995): boundary examples minimize teaching set size.
LIMO's selection directly implements this for generative models: hardest solvable problems give
strongest parameter updates with valid supervision targets. 817 competition-math traces ≈ 13 min
training on E4B.

## Key Risks (from REVIEW-adversarial.md)
1. **Format mismatch**: Training uses `<think>…</think>` but Gemma 4 generates `<|channel>thought…<channel|>`.
   PAPER.md MUST report avg_thinking_chars for base and adapted — 0 chars = thinking suppression.
2. **Capability ceiling**: LIMO curation model >> E4B 4-bit. "Barely solvable" for large model
   may be "impossible" for E4B. Watch training loss: if no decrease vs random, this is the kill.
3. **K1493 (≥65%) is aggressive**: Competition-math focus → math/physics categories improve,
   others flat/degrade. Net MMLU-Pro gain may be < 2.9pp. Per-category breakdown required.

## Implications for Next Experiment
If LIMO's adapted eval shows 0 avg_thinking_chars: both s1K + LIMO need retraining with
`<|channel>thought…<channel|>` tokens. Next experiment after LIMO: GRPO reasoning (RL
refinement after SFT baseline is established).

## When Results Arrive (pueue task 5)
- Check: avg_thinking_chars (base vs adapted)
- Check: per-category MMLU-Pro (math vs non-math delta)
- Check: GSM8K vs s1K (competition-math transfer to arithmetic)
- K1493: ≥65% MMLU-Pro thinking | K1494: ≥85% GSM8K | K1495: <1h training
