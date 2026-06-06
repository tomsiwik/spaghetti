# Knowledge Distillation for Ternary LoRA Adapters

## 1. Mechanism Definition

### Knowledge Distillation Loss (Hinton et al., 2015)

Given a teacher model T with parameters theta_T and a student model S with
trainable LoRA parameters phi, the distillation loss combines two terms:

```
L_total = alpha * L_KD(T, S; x) + (1 - alpha) * L_CE(S; x, y)
```

where:
- L_CE is the standard cross-entropy (hard target) loss
- L_KD is the KL-divergence soft target loss
- alpha in [0, 1] controls the mixture (typically alpha = 0.5-0.9)

### Soft Target Loss via Temperature Scaling

For input sequence x = (x_1, ..., x_n), the teacher produces logits
z_T in R^{n x V} and the student produces z_S in R^{n x V}, where V is
vocabulary size (V = 32,000 for BitNet-2B-4T).

Temperature-scaled softmax:
```
p_T^(t)(i) = exp(z_T_i / tau) / sum_j exp(z_T_j / tau)
p_S^(t)(i) = exp(z_S_i / tau) / sum_j exp(z_S_j / tau)
```

where tau > 1 is the temperature. Higher tau softens the distribution,
revealing the teacher's "dark knowledge" (relative probabilities of
non-top tokens that encode similarity structure).

KL-divergence loss (forward KL):
```
L_KD = tau^2 * sum_i p_T^(t)(i) * log(p_T^(t)(i) / p_S^(t)(i))
```

The tau^2 factor compensates for the gradient magnitude reduction caused
by temperature scaling: d/dz [softmax(z/tau)] = (1/tau) * d/dz [softmax(z)],
so the gradient of KL w.r.t. z_S is scaled down by 1/tau^2. Multiplying
by tau^2 restores the gradient magnitude to match L_CE scale.

### Why Forward KL, Not Reverse KL

Forward KL (KL(p_T || p_S)) is mean-seeking: the student tries to cover
all modes of the teacher. This is appropriate when the student has enough
capacity to represent the teacher's distribution approximately.

Reverse KL (KL(p_S || p_T), used by MiniLLM arxiv 2306.08543) is
mode-seeking: the student picks high-probability modes and ignores the
rest. Better when student capacity is severely limited.

Our student (BitNet-2B + rank-16 LoRA) has 2.4B base params + 21.6M LoRA
params. The adapter modifies only 0.9% of parameters. We use forward KL
because:
1. The adapter's role is to shift the full distribution toward the domain,
   not to pick specific modes
2. We want the adapter to learn the teacher's similarity structure
   (which tokens the teacher considers "almost right")
3. Forward KL is simpler and more stable for LoRA fine-tuning

### What the Student Actually Learns

The student model S computes:
```
z_S = f(x; theta_base + sum_l B_l @ A_l)
```

where theta_base is the frozen BitNet-2B-4T base, and {A_l, B_l} are the
rank-16 LoRA matrices at each linear layer l.

With self-supervised training (L_CE only on domain text):
```
L_self = -sum_t log p_S(x_{t+1} | x_{1:t})
```

The student learns to predict the next token in domain text. The gradient
signal comes only from the hard target (the correct next token).

With distillation (L_KD from teacher):
```
L_KD = tau^2 * sum_t KL(p_T^(tau)(. | x_{1:t}) || p_S^(tau)(. | x_{1:t}))
```

The student learns to match the teacher's FULL distribution at each position.
This provides O(V) bits of gradient signal per position (one for each vocab
token) vs O(1) bit from the hard target. The teacher's soft targets encode:
- Which alternative tokens are plausible (dark knowledge)
- The relative confidence across alternatives
- Domain-specific linguistic patterns the teacher has internalized

## 2. Why It Should Work for Ternary Adapters

### The Quality Gap is Real

From generation_quality_test LEARNINGS: 3/5 domains produce worse generation
quality despite PPL improvements. The legal adapter causes repetitive collapse
(13 repetitions of "hoa" in 100 words). This is mode collapse transferred
from narrow training data.

### How Distillation Addresses Mode Collapse

Self-supervised training on narrow domain data:
```
L_self = -log p_S(x_{t+1} | x_{1:t})
```
Only the single correct token receives gradient. If training data is
narrow (legal language is formulaic), the adapter collapses to a narrow
high-probability region.

Distillation from a larger teacher:
```
L_KD = tau^2 * KL(p_T^(tau) || p_S^(tau))
```
The teacher (Qwen2.5-7B, trained on diverse data including legal text) has
NOT collapsed. Its soft targets spread probability across plausible
continuations, preventing the student's adapter from collapsing to a
single mode.

Specifically, if the teacher assigns probability mass to K plausible
continuations at each position, the gradient has K non-zero components
instead of 1. The adapter learns a smoother distribution that resists
the repetition attractor.

### Ternary-Specific Benefit

BitNet-2B-4T has ternary {-1, 0, 1} base weights. The LoRA adapter adds
a continuous correction B @ A (bf16). With self-supervised training, the
adapter may overfit to domain-specific token frequencies because the
ternary base provides a coarser starting distribution than fp16 models.

The teacher (Qwen2.5-7B, fp16 precision internally) captures finer
distributional nuances. Distilling these into the adapter gives the
adapter a higher-quality learning signal than self-supervised NTP on
narrow domain text.

## 3. What Breaks It

### Capacity Bottleneck
The adapter has only 21.6M trainable parameters (rank-16, 7 projection
types, 30 layers). If the teacher's domain knowledge requires more than
rank-16 capacity to represent, the KD loss won't converge below the
self-supervised baseline.

Kill condition: if L_KD does not decrease below L_self after convergence,
the adapter lacks capacity to absorb the teacher's knowledge.

### Teacher-Student Vocabulary Mismatch
BitNet-2B-4T uses a 32K vocab. Qwen2.5-7B uses a 152K vocab. Direct KL
on logits is impossible without vocabulary alignment.

Solution: project both sets of logits to shared vocabulary. Since BitNet's
vocab is a subset (both derive from GPT-style tokenizers), we can:
1. Tokenize with EACH model's tokenizer separately
2. Use the teacher's probabilities at token positions that align
3. Or: use only the hard target loss from teacher (teacher generates,
   student trains on teacher's text) -- "sequence-level distillation"

For this micro experiment, we use SEQUENCE-LEVEL DISTILLATION: the teacher
generates soft continuation text on domain prompts, and the student trains
on this generated text with standard CE loss. This avoids the vocabulary
alignment problem entirely and is the standard approach when tokenizers
differ (TinyBERT, DistilBERT).

Alternatively, we can use LOGIT-LEVEL DISTILLATION when both models share
the same tokenizer. Since they don't here, we fall back to sequence-level.

### Temperature Sensitivity
Too high tau (>10): distributions become nearly uniform, losing signal.
Too low tau (<2): soft targets approximate hard targets, losing dark knowledge.
Optimal range: tau in [2, 6] for LLM distillation (Hinton et al., MiniLLM).

For sequence-level distillation, temperature is applied at generation time
via the teacher's sampling temperature.

### Training Data Quality
If the teacher generates low-quality domain text (hallucinations, errors),
the student learns those errors. This is especially risky for specialized
domains (medical, legal) where the teacher may lack genuine expertise.

Mitigation: use the teacher to REWRITE domain text (condition on domain
input, generate improved output) rather than generate from scratch. This
keeps the factual grounding of the original data while adding the teacher's
linguistic fluency.

## 4. Connection to Architecture

### How Distilled Adapters Fit SOLE

The SOLE architecture composes adapters via:
```
W_composed = W_base + sum_i (1/N) * B_i @ A_i
```

Distillation only changes how B_i is trained (the A_i are frozen
Grassmannian). The composition mechanism is unchanged. Therefore:
- Orthogonality guarantees still hold (A matrices are the same)
- Composition scaling (gamma) should be the same or better
- Memory footprint is identical (same rank, same number of parameters)

The hypothesis is that distilled B matrices produce better individual
adapter quality, which directly translates to better composed quality.

### Production Context

DeepSeek-V3 (arxiv 2412.19437) uses KD from DeepSeek-R1 to train its
MoE experts. They distill reasoning capability from a larger model into
smaller expert modules. Our approach is analogous: distilling domain
knowledge from Qwen-7B into LoRA expert modules.

Key difference: DeepSeek distills during pre-training (token-level KD
with shared tokenizer). We distill during adapter fine-tuning with
different tokenizers, hence sequence-level distillation.

## 5. Complexity Analysis

### Per-Step Cost Comparison

Self-supervised training:
- Forward pass (student): O(n * d^2 * L) for n tokens, d=2560, L=30 layers
- Backward pass: O(n * d^2 * L) (same as forward)
- Total per step: 2 * O(n * d^2 * L)

Distillation (sequence-level):
- Teacher generation (offline, once): O(n_gen * d_T^2 * L_T) per sample
  where d_T=3584 (Qwen-7B hidden dim), L_T=28 layers
- Student training: same as self-supervised (2 * O(n * d^2 * L))
- Total amortized: same as self-supervised (teacher cost is one-time)

The teacher forward pass adds ~25% overhead per token (d_T^2/d_S^2 =
3584^2/2560^2 = 1.96, but 4-bit quantization reduces this). Since teacher
generation is done once and cached, the training loop cost is identical
to self-supervised.

### Memory

- Teacher (Qwen2.5-7B-4bit): ~4.5 GB
- Student (BitNet-2B unpacked): ~3.9 GB
- LoRA params + optimizer: ~0.26 GB
- Total: ~8.7 GB + activations (~2 GB) = ~10.7 GB
- Budget: 40 GB usable on M5 Pro 48GB
- Fits comfortably (K2 trivially passes)

## 6. Worked Example (Micro Scale)

Domain: medical. Prompt: "Symptoms of diabetes include"

Teacher (Qwen-7B) generates at temperature=0.8:
"increased thirst, frequent urination, unexplained weight loss, fatigue,
and blurred vision. Type 2 diabetes may also present with slow-healing
sores and areas of darkened skin."

Self-supervised training text (from medical flashcards):
"Diabetes mellitus is characterized by hyperglycemia resulting from defects
in insulin secretion, insulin action, or both."

The distilled text is more fluent, covers more symptoms, and uses natural
clinical language. The adapter trained on this text learns:
- Broader vocabulary (teacher knows "darkened skin", "slow-healing sores")
- Natural phrasing (teacher writes fluently vs. flashcard style)
- Appropriate level of detail (teacher calibrates depth)

At training step t, the student sees teacher-generated text and computes:
```
L_CE = -log p_S("increased" | "Symptoms of diabetes include")
     + -log p_S("thirst" | "Symptoms of diabetes include increased")
     + ...
```

This is standard CE, but on higher-quality text than the original dataset.

## 7. Experimental Design

### Approach: Two-Phase Distillation

Phase 1: Teacher generates enhanced domain text
- Load Qwen2.5-7B-4bit
- For each domain: take N prompts from training data, generate completions
- Save generated text as distillation dataset
- Unload teacher to free memory

Phase 2: Student trains on distilled data
- Load BitNet-2B-4T, unpack, apply LoRA
- Train adapter on teacher-generated text (standard CE loss)
- Compare PPL against self-supervised baseline

### Kill Criteria

K1: Distilled adapter PPL must be >= 5% better than self-supervised
    adapter PPL on the SAME validation set.
    - Measure: PPL_distilled / PPL_self < 0.95

K2: Teacher + Student must both fit in 48GB during the generation phase.
    (Teacher alone during generation, student alone during training.)
    - Measure: peak memory < 40GB at any point

### Baseline

Self-supervised adapter PPL from bitnet_2b_real_composition:
- python: 2.22
- math: 3.60
- medical: 4.74
- legal: 16.53
- creative: 4.92
- Average: 6.40

Distilled adapters must achieve average PPL < 6.08 (5% improvement).

## 8. Assumptions

1. **Teacher quality assumption**: Qwen2.5-7B produces better domain text
   than the original training data. Justified by: 7B model trained on
   trillions of diverse tokens vs. narrow HF datasets.
   If wrong: distilled adapter PPL will be worse, K1 fails.

2. **Sequence-level sufficiency**: Training on teacher-generated text
   transfers enough knowledge without logit-level alignment.
   Justified by: DistilBERT, TinyBERT both show sequence-level KD works
   when tokenizers differ.
   If wrong: need shared-tokenizer teacher or vocabulary projection.

3. **Adapter capacity**: Rank-16 LoRA can absorb teacher knowledge.
   Justified by: rank-16 already learns domain-specific patterns from
   self-supervised training (26.3% PPL improvement).
   If wrong: need higher rank, which changes memory/composition math.

4. **Domain coverage**: Teacher has genuine knowledge in all 5 domains.
   Justified by: Qwen2.5-7B trained on diverse data including code, math,
   medical, legal, creative text.
   If wrong: domain-specific PPL will not improve for those domains.

## 9. Post-Mortem: Why the Hypothesis Failed

### The Critical Flaw: Output Distribution != Evaluation Distribution

The hypothesis assumed teacher-generated text would be "better" training data.
This is false because "better" was conflated with "higher quality prose."

The evaluation measures PPL on the ORIGINAL domain data (terse flashcards,
raw code, GSM8K answers). The teacher generates DIFFERENT-STYLE text
(verbose, markdown-formatted, explanation-heavy). Training the adapter
to predict Qwen-style text optimizes for the WRONG distribution.

Formally: let D_orig = original data distribution, D_teacher = teacher
output distribution. The adapter optimizes:
```
L_distilled = E_{x ~ D_teacher} [-log p_S(x)]
```
But evaluation measures:
```
PPL_eval = exp(E_{x ~ D_orig} [-log p_S(x)])
```

Since D_teacher != D_orig (different style, different length, different
vocabulary emphasis), minimizing L_distilled does NOT minimize PPL_eval.

### Quantitative Evidence

The distilled adapter achieves training loss 0.355 (python) vs 0.699
(self-supervised). It learned D_teacher BETTER than the self-supervised
adapter learned D_orig. But on D_orig evaluation, it scores 2.95 vs 2.35
(25.6% worse). Lower training loss + higher eval PPL is the diagnostic
signature of distribution mismatch.

### What Would Have Worked

1. **Logit-level KD on original data**: Run teacher forward pass on
   D_orig text. Use teacher's softmax probabilities as soft targets.
   This keeps D_orig as the training distribution while adding the
   teacher's dark knowledge. Requires vocabulary alignment (different
   tokenizers with different vocab sizes).

2. **Same-distribution teacher**: A teacher that uses the SAME tokenizer
   as the student. Then sequence-level KD on original prompts would
   produce text in the same tokenization space.

3. **Style-constrained generation**: Few-shot prompting the teacher to
   generate text in the EXACT style of the original data. For medical:
   "Generate a single-sentence medical fact: [topic]." This reduces
   the distribution gap but limits dark knowledge transfer.

### Lesson for the Project

Sequence-level knowledge distillation across different tokenizers and
evaluation distributions is not viable. The distribution mismatch
problem is fundamental, not a hyperparameter issue. Future quality
improvements should use:
- DPO/RLHF on the adapter's OWN generations
- Logit-level KD with vocabulary projection
- Better training data selection (not generation)
