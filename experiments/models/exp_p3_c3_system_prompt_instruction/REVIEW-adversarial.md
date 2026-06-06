# REVIEW-adversarial.md — P3.C3: System Prompt Instruction

**Status:** PROCEED (early kill)  
**Date:** 2026-04-11  

## Adversarial Assessment

### Is the early kill justified?

Yes. 0/5 with degenerate outputs is not noise — the ablation confirms all three
instruction variants give 0% with hallucinations. The control (66.7% on ablation
prompts) proves the adapter works without system prompt.

### Is the Gemma 4 system role conclusion correct?

Most likely. Gemma 4 instruction-tuned variants use `<start_of_turn>user` / `<start_of_turn>model`
format. The system role may route through apply_chat_template which may produce unexpected tokens.

Counter-argument: some Gemma 4 variants DO support system role in later fine-tunes.
Could test by running generate without adapter to see if system prompt works on base model.

But: the personal adapter was trained without system prompts regardless, so even if
the template worked, adapter distribution shift would remain. Kill is valid.

### Is degenerate output (PHP/Chinese) explained?

The token `system` in `<start_of_turn>system` appears OOD → model generates from
an unexpected distribution → random tokens from training data (PHP, Chinese, repeated greetings).
This is consistent with known behavior of instruction-tuned models when their template
is violated.

### Should P3.C4 use rank-16?

Yes. It directly addresses the rank bottleneck (Finding #468) proven as the primary
constraint. Neither context tricks (C2) nor template tricks (C3) work due to Gemma 4
specifics. Rank-16 is the only remaining adapter-level solution.

## Verdict: PROCEED (KILLED)
