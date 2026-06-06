# Answer-Conditioned Scoring: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| V | Vocabulary size (char-level) | 42 |
| T | Sequence length | up to 48 |
| T_p | Number of prompt tokens | varies by domain |
| T_a | Number of answer tokens | varies by domain |
| d | Model embedding dimension | 32 (micro) |
| d_ff | FFN intermediate dimension | 4d = 128 |
| H | Number of attention heads | 2 |
| L | Number of transformer layers | 2 |
| N | Number of domain experts | 5 |
| theta | Base model parameters | ~29K |
| delta_i | Expert i's parameter delta | ~29K (full-rank) |
| p_theta | Model's output distribution | Simplex over V |
| x | Sequence (x_1, ..., x_T) | token IDs |

## 2. The Three PPL Variants

### 2.1 Full-Sequence PPL (predecessor metric)

For a sequence x = (x_1, ..., x_T) under model theta:

    PPL_full(theta, x) = exp(-1/T * sum_{t=1}^{T} log p_theta(x_t | x_{<t}))

This averages the cross-entropy loss over ALL token positions.

### 2.2 Answer-Only PPL (this experiment's metric)

Partition the sequence at the delimiter position d*:
- Prompt tokens: x_1, ..., x_{d*} (including delimiter)
- Answer tokens: x_{d*+1}, ..., x_T (after delimiter)

    PPL_answer(theta, x) = exp(-1/T_a * sum_{t=d*+1}^{T} log p_theta(x_t | x_{<t}))

where T_a = T - d* is the number of answer tokens.

Key: the conditioning context x_{<t} still includes all previous tokens
(both prompt and earlier answer tokens). We only RESTRICT which positions
contribute to the PPL average, not what the model sees.

### 2.3 Prompt-Only PPL (diagnostic)

    PPL_prompt(theta, x) = exp(-1/T_p * sum_{t=1}^{d*} log p_theta(x_t | x_{<t}))

where T_p = d* is the number of prompt tokens.

## 3. Decomposition Identity

Full-sequence PPL decomposes into prompt and answer components:

    log PPL_full = (T_p / T) * log PPL_prompt + (T_a / T) * log PPL_answer

This is because:

    -(1/T) * sum_{t=1}^{T} log p(x_t|x_{<t})
    = -(T_p/T) * (1/T_p) * sum_{t=1}^{d*} log p(x_t|x_{<t})
      -(T_a/T) * (1/T_a) * sum_{t=d*+1}^{T} log p(x_t|x_{<t})

Therefore: log PPL_full = (T_p/T) * log PPL_prompt + (T_a/T) * log PPL_answer

## 4. Why Full-Sequence PPL Misleads

### 4.1 The Dilution Effect

When a domain expert specializes on answer generation:
- PPL_answer improves (expert learns correct answer patterns)
- PPL_prompt may degrade (expert loses broad prompt modeling)
- PPL_full = weighted average of both, potentially dominated by prompt degradation

For the expert to show improved PPL_full, we need:

    T_a * Delta(log PPL_answer) > T_p * |Delta(log PPL_prompt)|

where Delta denotes the change from base to expert. When T_p >= T_a
(prompt is at least as long as answer, which is common), even moderate
prompt degradation can mask large answer improvement.

### 4.2 The Reverse Domain Example

For "abc>cba" (reverse domain):
- Prompt "abc>": character bigrams follow forward English-like patterns
- Answer "cba": character bigrams are reversed (anti-correlated with training data)

The base model, trained on mixed domains, has moderate PPL on both portions.
The reverse expert:
- Learns to produce correct reversed strings (PPL_answer drops dramatically)
- But the reversed bigram patterns are unlike ANY other domain's prompt patterns
- PPL_prompt may spike because the expert's internal representations shift
  to favor reverse-order character predictions

Result: PPL_answer improves by 58-68%, PPL_prompt worsens by 100-300%,
PPL_full shows a small change or degradation, while accuracy jumps 30-50pp.

### 4.3 Formal Condition for Divergence

Full-seq PPL improvement and accuracy improvement diverge when:

    sign(Delta PPL_full) != sign(Delta Accuracy)

This occurs whenever:
1. Delta PPL_prompt < 0 (prompt modeling degrades)
2. Delta PPL_answer > 0 (answer modeling improves)
3. |T_p * Delta(log PPL_prompt)| > |T_a * Delta(log PPL_answer)|

Condition 3 is met when T_p/T_a > |Delta(log PPL_answer)| / |Delta(log PPL_prompt)|,
i.e., when the prompt is long relative to the answer AND the prompt
degradation rate exceeds the answer improvement rate.

## 5. Accuracy-PPL Alignment

For greedy decoding accuracy to correlate with PPL, we need:

    argmax_v p_theta(v | x_{<t}) = x_t  IFF  p_theta(x_t | x_{<t}) is high

Answer-only PPL satisfies this more closely because:
1. It measures exactly the tokens that determine accuracy (answer positions)
2. High answer-PPL implies confident predictions on answer tokens
3. Confident correct predictions = both low answer-PPL AND high accuracy

The alignment is NOT perfect because:
- PPL weights all answer positions equally; accuracy requires ALL positions correct
- A model with 99% answer-PPL improvement on 4 of 5 positions but 0% on the
  5th has great PPL but only ~0% whole-answer accuracy

## 6. Experimental Design

### 6.1 Domains (identical to predecessor)

| Domain | Format | Delimiter | T_p (avg) | T_a (avg) |
|--------|--------|-----------|-----------|-----------|
| arithmetic | "A+B=C" | = | ~4 | ~2 |
| reverse | "abc>cba" | > | ~5 | ~5 |
| repeat | "ab*3=ababab" | = | ~5 | ~6 |
| sort | "bca>abc" | > | ~5 | ~5 |
| parity | "1011>even" | > | ~5 | ~4 |

### 6.2 Model

2-layer causal transformer (d=32, H=2, V=42). Pure numpy + autograd.
~29K parameters. Trained with Adam (lr=0.001, gradient clipping at 1.0).

### 6.3 Expert Training

Full fine-tuning on domain-specific data (500 examples, 30 epochs).
Expert delta = trained_params - base_params.

### 6.4 Evaluation

For each domain and model variant (base, expert_i):
- PPL_full: full-sequence cross-entropy on 500 held-out examples
- PPL_answer: cross-entropy only on answer tokens (after delimiter)
- PPL_prompt: cross-entropy only on prompt tokens (before delimiter)
- Accuracy: greedy decode on 200 fresh examples (exact match)

## 7. Worked Example

From seed 42 results, reverse domain:

Base model:
- FullPPL = 7.38, AnsPPL = 3.01, PromptPPL = 22.83, Acc = 0.330

Reverse expert:
- FullPPL = 7.51 (+1.8% WORSE), AnsPPL = 1.25 (+58.5% BETTER), Acc = 0.730 (+40pp)

Decomposition check (approximate, since PPL is geometric mean):
- log(7.38) = 2.00, log(3.01) = 1.10, log(22.83) = 3.13
- Weighted: (5/10)*3.13 + (5/10)*1.10 = 2.12 vs actual 2.00 (close, with EOS)

Expert:
- log(7.51) = 2.02, log(1.25) = 0.22, log(new_prompt) = ?
- The expert's FullPPL barely changed (+1.8%) because the massive answer
  improvement (+58.5%) was cancelled by prompt degradation.

This is precisely the phenomenon: **full-sequence PPL is blind to
the expert's dramatic improvement on the answer tokens**.

## 8. Statistical Considerations

With N=5 domains, the critical value for Pearson r at p<0.05 (one-tailed)
is r >= 0.687. Our answer-only r values (0.91, 0.94, 0.58) exceed this
threshold in 2 of 3 seeds, with the third at r=0.58 still above our
kill criterion of 0.5.

The Spearman rank correlation (rho=0.90, p=0.037 for seeds 42 and 123)
provides non-parametric confirmation that the ranking relationship is
statistically significant despite the small sample size.

## 9. Computational Cost

| Operation | Time (micro, CPU) |
|-----------|-------------------|
| Base training (20 epochs, 2500 seqs) | ~41s |
| Expert training (30 epochs, 500 seqs) | ~10s each |
| PPL evaluation (3 types, 500 seqs) | ~10s per domain |
| Accuracy evaluation (200 examples) | ~8s per domain |
| Total per seed | ~170s |
| Total (3 seeds) | ~8 min |
