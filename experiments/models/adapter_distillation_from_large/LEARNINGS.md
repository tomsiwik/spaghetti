# Learnings: Adapter Distillation from Large Teacher

## Core Finding

Sequence-level knowledge distillation from Qwen2.5-7B-Instruct into BitNet-2B-4T LoRA adapters produces **34.4% worse PPL** than self-supervised training (0/5 domains improved). The failure is fundamental: cross-tokenizer sequence-level KD trains the adapter on the teacher's output distribution (verbose, markdown-heavy), not the evaluation distribution (terse domain text). Lower training loss + higher eval PPL is the diagnostic signature of distribution mismatch, not capacity failure.

## Why This Happened

### Distribution Mismatch Is the Mechanism, Not Capacity

The experiment used **sequence-level distillation**: teacher generates text, student trains on it with standard cross-entropy. This reduces to training on a different data distribution. When Qwen2.5-7B-Instruct generates "Certainly! Below is a detailed overview..." for a medical prompt, the adapter learns to predict verbose explanation tokens. Evaluating on terse flashcards ("Very low Mg2+ levels correspond to low PTH levels...") measures a completely different distribution.

Formally: minimizing `E_{x ~ D_teacher}[-log p_S(x)]` does not minimize `exp(E_{x ~ D_orig}[-log p_S(x)])` when `D_teacher != D_orig`. The 0.355 vs 0.699 training loss gap (python domain) confirms the adapter learned D_teacher *better* than the self-supervised adapter learned D_orig -- it just learned the wrong thing.

### MATH.md Describes a Different Method Than Was Tested

The MATH.md derives logit-level KD (temperature scaling, KL divergence, dark knowledge via soft targets) in Sections 1-3, but the experiment implements sequence-level KD (Section 7) because BitNet-2B-4T and Qwen2.5-7B have incompatible tokenizers (32K vs 152K vocab). The mathematical framework -- tau^2 gradient compensation, forward KL vs reverse KL, O(V) gradient signal -- is irrelevant to what was actually tested. Sequence-level KD has no temperature parameter in the loss, no KL divergence, no soft targets. It is standard next-token prediction on teacher-generated text.

### Instruct Model Guarantees Distribution Mismatch

Using Qwen2.5-7B-**Instruct** (not base) with chat templates guarantees verbose, markdown-formatted, "Certainly!"-prefixed output. The reviewer correctly notes this was foreseeable -- inspecting 5 teacher samples per domain would have killed the hypothesis in minutes.

## Confirming Evidence

1. **Our own generation_quality_test (exp_generation_quality_test)**: PPL improvement does not predict generation quality for prose domains (r=0.08 correlation). The "Two-World Pattern" -- adapters help structured tasks, hurt prose -- is the same distribution-fit problem at generation time that we see here at training time.

2. **MiniLLM (arxiv 2306.08543)**: Uses reverse KL specifically to handle teacher-student distribution gaps. Even MiniLLM assumes same-distribution evaluation. No paper in the KD literature tests cross-distribution sequence-level distillation with different tokenizers and expects it to work.

3. **DistilBERT (Sanh et al., 2019) and TinyBERT (arxiv 1909.10351)**: Both succeed because teacher and student share the same tokenizer AND distillation data matches the evaluation distribution. Our setup violates both conditions.

4. **DeepSeek-V3 (arxiv 2412.19437)**: Successfully distills from DeepSeek-R1, but uses token-level KD with a shared tokenizer during pre-training. Different mechanism, different success conditions.

5. **Our own 50-expert distillation pipeline (FINDINGS.md)**: Achieves 98% win rate and 42.2% PPL improvement. That pipeline distills on *same-distribution* instruction-format data. Same-distribution KD works; cross-distribution does not.

## Contradicting Evidence

1. **Alpaca-style data generation** (Stanford Alpaca, 2023) successfully uses GPT-4-generated instruction-following data to train smaller models. However, the evaluation there is on instruction-following quality -- the same distribution as the generated data. When evaluation matches teacher output style, the approach works. Our evaluation deliberately uses the *original* domain data style, creating the mismatch.

2. **The "BitNet vocab is subset of Qwen" claim** (MATH.md Section 3) is unverified and likely false. Both are independently trained BPE vocabularies. If true, it would enable vocabulary projection for logit-level KD. The reviewer flags this as an unvalidated assumption that should not be carried forward.

## Alternative Approaches

### 1. Logit-Level KD on Original Data
Run teacher forward pass on original domain text. Use teacher's softmax probabilities as soft targets with vocabulary projection. This keeps D_orig as training distribution while adding dark knowledge. Requires solving the 32K-to-152K vocabulary alignment problem.
- **Literature**: Hinton et al. (2015) original formulation; MiniLLM (arxiv 2306.08543) for LLM-specific implementation.
- **Challenge**: Vocabulary projection between independently trained BPE tokenizers is non-trivial. No standard method exists for BitNet's custom tokenizer.

### 2. DPO/RLHF on Adapter's Own Generations
Train adapters with preference optimization rather than next-token prediction. Generate pairs from the adapted model, label preferences, optimize directly for generation quality.
- **Literature**: DPO (arxiv 2305.18290). Already identified as the most promising quality fix in generation_quality_test LEARNINGS.
- **Advantage**: Trains on the model's own output distribution, eliminating the mismatch by definition.

### 3. Style-Constrained Few-Shot Generation
Few-shot prompt the teacher to generate text matching the original data style: "Generate a single-sentence medical fact about [topic]." Reduces distribution gap.
- **Trade-off**: Limits dark knowledge transfer. The teacher becomes a data augmenter, not a knowledge distiller.
- **Not tested, cheap to try**: Could be a quick ablation if distillation is revisited.

### 4. Cross-LoRA Transfer (arxiv 2508.05232)
Data-free LoRA transfer across base models. Transfers adapter capabilities without regenerating training data. Avoids the distribution mismatch entirely by operating in weight space.
- **Relevance**: Directly addresses our problem (different base models, different tokenizers).
- **Untested on ternary models**: Would need validation that cross-LoRA works with BitNet's ternary weights.

## Implications for Next Experiments

1. **Sequence-level KD is dead for cross-tokenizer scenarios.** Any future distillation must either (a) use a same-tokenizer teacher, (b) solve vocabulary projection for logit-level KD, or (c) constrain the teacher to match the evaluation distribution exactly.

2. **DPO remains the most promising quality improvement path.** Both generation_quality_test and this experiment converge on the same conclusion: next-token prediction on static data (whether original or teacher-generated) is insufficient for generation quality. Preference-based training is the fix.

3. **Self-supervised adapters are already strong.** The 100-sample self-supervised control (PPL 7.50) vs 500-sample baseline (PPL 6.40) shows 17.1% degradation from reduced data, but the adapters still learn meaningful domain shifts. Quality improvement should focus on training objectives, not training data source.

4. **The project's existing 50-expert pipeline works *because* it uses same-distribution data.** This is now confirmed from both positive (98% win rate in FINDINGS.md) and negative (this experiment) evidence. Any data generation approach must match the evaluation distribution.

## Recommended Follow-Up

No new experiment recommended from this result. The quality improvement path is already identified: DPO/RLHF-based adapter training (from generation_quality_test LEARNINGS). This experiment confirms that the quality bottleneck cannot be fixed by switching data sources -- it requires switching training objectives.
