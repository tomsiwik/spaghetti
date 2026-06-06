# LEARNINGS — exp_g4_zs_base_transfer_4bit_fp16_full

## Core Finding
4→8-bit adapter transfer on Gemma 4 E4B is functionally lossless and strictly improving across all three domains. K1814 (PPL-ratio proxy, inherited from F#680): FAIL at median R_ppl=0.946. K1815 (target task accuracy): PASS, with R_task strictly > 1: HumanEval 70→80 (1.143), GSM8K 72→82 (1.139), MedQA 68→70 (1.029). Per F#666 truth table this is a finding-about-the-proxy, not a kill — registered as F#767.

## Why
Two non-exclusive mechanisms (PAPER §Mechanism): (1) the 8-bit base is a more accurate substrate than the 4-bit base the adapter was trained against; the higher-precision host dominates the slight adapter mis-targeting — consistent with **F#97** (ZS transfer is lossless across higher-fidelity rungs of the same model). (2) Quantization-induced decode hallucination on 4-bit was holding accuracy back, not adapter capacity (+10pp on HumanEval/GSM8K vs +2pp on MedQA tracks code/math being more decode-quantization-sensitive than 4-way argmax).

This empirically inverts parent **F#680**'s proxy framing ("marginally failing at the 5% PPL threshold"). It also confirms **guardrail 1006** (`r ≈ 0.08(PPL, task)` in this codebase): a marginal proxy-FAIL predicts ~nothing about behavior. **F#666** target-gated discipline worked exactly as designed — a proxy-only kill rule would have killed an apparent *win*.

## Implications for Next Experiment
- Treat F#680's PROVISIONAL status as resolved on the behavioral side. Any future work citing F#680 must also cite F#767; the "marginally failing PPL ratio" framing is no longer load-bearing.
- Default toward 8-bit base when serving a 4-bit-trained adapter if the slot has 8-bit weights resident: expect +2 to +10pp on this model class without retraining. Free win for any production path.
- Highest-value content-level follow-ups (deferred, not auto-filed): (1) 4→bf16 strict rung if a slot has bf16 E4B resident; (2) adapter-retrained-per-precision to discriminate mechanism (1) vs (2). Claim only with a fresh slot — do not re-trigger this experiment.
- Guardrail reinforcement: every PPL-or-cosine-only KC remaining in the open backlog is under F#666 risk. Continue retrofitting target-paired KCs *before* claiming, not after.
