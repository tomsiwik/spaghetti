# MATH.md — P3.C3: System Prompt Instruction for Style Injection

**Experiment:** exp_p3_c3_system_prompt_instruction  
**Date:** 2026-04-11  
**Type:** verification  

## Background

P3.C1 (Finding #468): rank-4 LoRA ceiling at 60% regardless of data diversity.
P3.C2 (Finding #469): few-shot examples DEGRADE compliance (20% vs 40% zero-shot).
Root cause of P3.C2 failure: context-prior conflict — examples shift generation prior
into elaborate explainer mode, overriding weight-learned adapter bias.

New question: Can a SYSTEM PROMPT INSTRUCTION inject style without either weight
modification or in-context examples?

## Mechanism: Instruction Following via RLHF Conditioning

RLHF-tuned models (InstructGPT, Gemma 4, etc.) are trained to treat system prompt
directives as hard constraints on generation. The system prompt has a special role
in the chat template that receives maximum attention weighting and acts as a
persistent constraint throughout the response.

Key difference from few-shot:
- Few-shot examples: content in USER turn → suggests style via imitation
- System prompt instruction: explicit directive in SYSTEM turn → enforces constraint via RLHF conditioning

## Theorem 1: System Prompt Instruction Is RLHF-Orthogonal to Adapter Bias

**Claim:** A system prompt instruction "Always end with pattern P" is conditioned on
by RLHF training independently of any adapter weight perturbation. The instruction
constraint is additive to the adapter bias, not competing.

**Proof sketch:**

Let h_adapter = h_base + ΔW_personal(h_domain_fused).
The RLHF objective trained the model to maximize P(response satisfies instruction | system_prompt, h).

For the system prompt instruction "Always end with 'Hope that helps, friend!'":
- The model must output P at token position T_{end}.
- The adapter bias may or may not help generate P, but the RLHF constraint operates
  at the LOGIT LEVEL for the final tokens, not at the hidden state level.
- Formally: P(P | h_{T_end}, instruction) > P(P | h_{T_end}) for any h.

**QED (informal):** RLHF instruction conditioning and adapter weight perturbation are
orthogonal mechanisms operating at different levels of the generation process.

## Theorem 2: Token Overhead Is Minimal

**Claim:** System prompt instruction "Always end your response with 'Hope that helps, friend!' — include this exact phrase at the end of every response." has L_sys ≈ 20 tokens.
For typical query L_q ≈ 20 tokens:

    overhead_ratio = (L_sys + L_q) / L_q = 1 + L_sys / L_q ≈ 2.0

**QED**: K1204 threshold of 2.0 is achievable. (Note: actual Gemma 4 chat template
may add role tokens, slightly increasing ratio to ~2.5.)

## Quantitative Predictions

| Metric | P3.C0/C1 baseline | Prediction | Kill threshold | Basis |
|--------|-------------------|------------|----------------|-------|
| style_instruction (system prompt) | 60% | ≥ 85% | K1202 ≥ 80% | InstructGPT: instruction following ~95% on explicit format |
| style_no_instruction (control) | 60% | ~60% | diagnostic | P3.C0 verified |
| training_cost | N/A | 0 min | K1203 = True | No weights changed |
| overhead_ratio | 1.0x | ~2.0x | K1204 ≤ 2.0 | Theorem 2 |

## Experimental Design

**Phase 0:** Verify B5 artifacts (domain_fused_base, new_personal_adapter).

**Phase 1:** Control — zero-shot without system prompt (N=15).
  - Confirms ~60% baseline.

**Phase 2:** System prompt instruction (N=15).
  - System: "Always end your response with 'Hope that helps, friend!' — include this exact phrase at the very end of your response, after your explanation."
  - Measures: style_instruction, overhead_ratio.

**Phase 3:** Ablation — instruction variants (N=5 each):
  - Variant A: short instruction ("End with: Hope that helps, friend!")
  - Variant B: long explicit instruction (above)
  - Variant C: instruction with personal adapter vs base model only

## Kill Criteria

- **K1202**: style_instruction ≥ 80% (primary: above LoRA ceiling AND above few-shot)
- **K1203**: zero_training_cost = True (secondary: no adapter trained)
- **K1204**: context_overhead_ratio ≤ 2.0 (secondary: minimal context overhead)

## If KILLED (instruction compliance < 80%)

Impossibility structure: Gemma 4 (instruction-tuned) DOES follow system prompt
directives, but the PREFERENCE_MARKER "Hope that helps, friend!" is a NOVEL pattern
not present in Gemma 4 pre-training or RLHF data. The model may approximate it but
not reproduce the exact string. The model learns to PARAPHRASE instructions, not
copy verbatim tokens.

Fix: P3.C4 — retrain personal adapter with MUCH MORE data (1000+ examples), higher
rank (rank=32), or use the personal adapter as the PRIMARY mechanism (not composition)
and test on questions from its training distribution only.

## References

- Ouyang et al. 2022 (arxiv 2203.02155): InstructGPT — RLHF enables reliable instruction following
- Finding #468 (P3.C1): rank bottleneck at 60%
- Finding #469 (P3.C2): few-shot context-prior conflict degrades to 20%
- Finding #466 (P3.B5): personal adapter achieves 92% on same-distribution questions
