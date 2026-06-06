# Peer Review: Adapter Distillation from Large Teacher

## NotebookLM Findings

Skipped -- the experiment is already killed with unambiguous results, and the post-mortem analysis in MATH.md Section 9 is thorough. A NotebookLM deep review would not surface additional concerns beyond what follows.

## Mathematical Soundness

The MATH.md is well-structured and the core KD derivations (Sections 1-3) are correct:

- Temperature-scaled KL with tau^2 gradient compensation: correct.
- Forward KL vs reverse KL trade-off: correctly stated.
- O(V) gradient signal per position from soft targets vs O(1) from hard targets: correct.

**However, the math describes logit-level KD (Sections 1-3) while the experiment implements sequence-level KD (Section 7).** The mathematical framework built in Sections 1-3 (temperature scaling, KL divergence, dark knowledge) is almost entirely irrelevant to what was actually tested. Sequence-level distillation reduces to standard cross-entropy on teacher-generated text. There is no temperature parameter in the loss, no KL divergence, no soft targets. The tau^2 derivation, the forward-vs-reverse KL analysis, and the "O(V) bits of gradient signal" argument all apply to a method that was explicitly abandoned due to vocabulary mismatch.

The post-mortem (Section 9) correctly identifies the distribution mismatch. The formal statement is clean:

```
L_distilled = E_{x ~ D_teacher} [-log p_S(x)]
PPL_eval = exp(E_{x ~ D_orig} [-log p_S(x)])
D_teacher != D_orig => minimizing L_distilled does NOT minimize PPL_eval
```

This is the correct diagnosis. The quantitative signature (lower training loss + higher eval PPL) is textbook distribution shift.

**One hidden assumption not flagged:** The MATH.md claims "BitNet's vocab is a subset" of Qwen's (Section 3, vocabulary mismatch). This is unverified and likely false. BitNet-2B-4T uses a custom 32K tokenizer; Qwen2.5 uses a 152K tokenizer. These are independently trained BPE vocabularies. Overlap is likely high for common tokens but "subset" is a mathematical claim that was not checked. This matters because the document uses it to justify the possibility of vocabulary projection for logit-level KD -- which was never implemented, but would be relevant for the "what would have worked" section.

## Novelty Assessment

**Low novelty, but that is acceptable for a micro-experiment testing a hypothesis.**

Sequence-level KD from a larger teacher is standard practice (DistilBERT, TinyBERT, Alpaca-style data generation). The paper correctly cites Hinton 2015, MiniLLM, TinyBERT. The specific application to ternary LoRA adapters is new, but the mechanism is not.

The failure mode (distribution mismatch when teacher generates different-style text than evaluation distribution) is well-known in the data augmentation literature but is often overlooked when people apply "teacher generates training data" approaches. This is a useful negative result for the project.

**Important prior art within the project:** FINDINGS.md already documents a successful distillation pipeline (50-expert distillation, 98% win rate, 42.2% avg PPL improvement). That pipeline distills on instruction-format data with the same evaluation distribution. The difference is instructive: same-distribution distillation works; cross-distribution does not.

## Experimental Design

The experimental design is adequate for falsifying the hypothesis, with good controls:

1. **Fair control condition**: The self-supervised baseline was retrained with the same 100-sample budget, same hyperparameters, and same random seed mechanics. This controls for sample count effects (the 500-sample baseline PPL of 6.40 vs 100-sample self-supervised PPL of 7.50 confirms sample count matters).

2. **Same validation set for both conditions**: Correct. Both distilled and self-supervised are evaluated on the original domain validation data.

3. **Clear kill threshold**: 5% improvement, measured on the correct metric (PPL on held-out original data).

**Design flaw that was predictable in advance:**

The most significant weakness is that the distribution mismatch was foreseeable from the experimental design. MATH.md Section 3 explicitly identifies the vocabulary mismatch problem and proposes sequence-level distillation as the workaround. But the document does not analyze the consequence: when you use an instruct model (Qwen2.5-7B-Instruct-4bit) with chat templates and system prompts to generate text, the output distribution will be instruction-tuned chat style (verbose, markdown, "Certainly!"), not the terse domain text in the evaluation set.

A cheaper falsification would have been to simply inspect 5 teacher generations per domain before committing to full training. The PAPER.md itself describes the distribution mismatch vividly (Section "Why This Failed"). This inspection step would have killed the experiment in minutes rather than 24 minutes of compute.

**Missing ablation:** Style-constrained generation (MATH.md Section 9, item 3) was identified as a potential fix but not tested. A single-domain test with few-shot style matching would have been informative and cheap. However, this is an observation, not a blocking critique -- the experiment's stated hypothesis was about unconstrained sequence-level KD, and it was cleanly falsified.

## Macro-Scale Risks (advisory)

Not applicable -- the mechanism is killed. The findings do have macro relevance:

1. The project's existing 50-expert distillation pipeline (FINDINGS.md) works because it uses same-distribution data. Any future attempt to use cross-model generation for training data must control for style distribution.

2. Logit-level KD with vocabulary projection remains untested. If pursued at macro scale, the vocabulary alignment problem (32K vs 152K) needs a concrete projection method, not the hand-wave in MATH.md Section 3.

3. The post-mortem's suggestion of DPO/RLHF is consistent with Track C in VISION.md and should be the preferred path for quality improvement.

## Verdict

**KILL**

The kill is correct and well-justified. The evidence is unambiguous: 0/5 domains improved, -34.4% average degradation, with a clean mechanistic explanation (distribution mismatch between teacher output style and evaluation distribution). The post-mortem in MATH.md Section 9 is honest and thorough.

**Quality of the negative result:** High. The experiment was well-controlled, the failure mode is clearly diagnosed, and the post-mortem correctly identifies three alternative approaches that might work. The lesson -- sequence-level KD across different tokenizers and output distributions does not transfer -- is a useful addition to the project's knowledge base.

**Minor notes for recordkeeping:**

1. MATH.md Sections 1-3 should be flagged as describing the logit-level method that was NOT implemented. The document reads as if the mathematical framework motivates the experiment, but the actual experiment uses a different (simpler) method. This is a presentation issue, not a correctness issue, since Section 7 correctly describes what was done.

2. The "BitNet vocab is a subset" claim (Section 3) is unverified and should not be carried forward to future experiments without checking.

3. The self-supervised 100-sample control shows 17.1% worse PPL than the 500-sample baseline (7.50 vs 6.40). This confirms sample efficiency matters and is a useful finding independent of the distillation result.
