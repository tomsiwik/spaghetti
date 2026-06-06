# PPL vs Task Performance: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| d | Model embedding dimension | 64 (micro) |
| d_ff | FFN intermediate dimension | 4d = 256 |
| r | LoRA rank (expert) | 8 |
| L | Number of transformer layers | 4 |
| N | Number of domain experts | 5 |
| V | Vocabulary size (char-level) | 42 |
| T | Sequence length | up to 48 |
| alpha | LoRA scaling factor | 1.0 |
| W | Base model weight matrix | R^{d_out x d_in} |
| A_i, B_i | LoRA factors for expert i | A: R^{r x d_in}, B: R^{d_out x r} |
| dW_i | Expert i's LoRA delta: (alpha/r) * B_i @ A_i | R^{d_out x d_in} |
| p_theta | Model's output distribution | Simplex over V |
| p_data | Empirical data distribution | Simplex over V |

## 2. The Metrics Under Study

### 2.1 Perplexity (PPL)

For a sequence x = (x_1, ..., x_T) under model theta:

    PPL(theta) = exp(-1/T * sum_{t=1}^{T} log p_theta(x_t | x_{<t}))

This is the exponentiated average cross-entropy loss. Lower PPL = better
language model for the domain's text distribution.

PPL improvement for expert i on domain i:

    PPL_imp_i = (PPL_base(D_i) - PPL_expert_i(D_i)) / PPL_base(D_i)

where D_i is the held-out eval set for domain i.

### 2.2 Task Accuracy

For structured tasks with deterministic answers:

    Acc_i = (1/M) * sum_{j=1}^{M} 1[greedy_decode(prompt_j) == answer_j]

where M is the number of eval prompts, prompt_j is the input side of the
structured sequence (e.g., "2+3="), and answer_j is the expected completion.

Accuracy improvement for expert i on domain i:

    Acc_imp_i = Acc_expert_i(D_i) - Acc_base(D_i)

### 2.3 The Correlation Question

Does PPL_imp_i positively correlate with Acc_imp_i across domains i=1..N?

    rho = Cor(PPL_imp, Acc_imp) = Cov(PPL_imp, Acc_imp) / (sigma_PPL * sigma_Acc)

Kill criterion: rho < 0.5 (weak or no correlation).

## 3. Why PPL and Task Accuracy Can Diverge

### 3.1 The Decomposition Problem

For a structured sequence "prompt>answer", PPL averages over ALL positions:

    PPL = exp(-(L_prompt + L_answer) / (T_prompt + T_answer))

where L_prompt = sum of log-probs on prompt tokens, L_answer = same for
answer tokens.

Task accuracy depends ONLY on the answer portion:

    Acc = 1[argmax p_theta(x_t | x_{<t}) == x_t  for all t in answer]

A model can have worse PPL (due to forgetting prompt distribution) while
having better task accuracy (due to learning answer patterns). This is
especially true for domain-specific LoRA experts that train on one domain
but are evaluated against a base model trained on all domains.

### 3.2 Formal Conditions for Divergence

Let PPL_full = PPL over entire sequence, PPL_answer = PPL over answer only.

Claim: PPL_full improvement can be negative while PPL_answer improvement
is positive, whenever:

    Delta_L_prompt > 0  (prompt modeling degrades)
    Delta_L_answer < 0  (answer modeling improves)
    |Delta_L_prompt| > |Delta_L_answer| * T_answer / T_prompt

In our structured domains (e.g., "bca>abc"), the prompt and answer are
roughly equal length, so even moderate prompt degradation can mask large
answer improvement in the full-sequence PPL.

### 3.3 The Accuracy-PPL Alignment Condition

For greedy decoding accuracy and PPL to correlate perfectly, we need:

    For all t in answer positions:
    argmax_v p_theta(v | x_{<t}) = x_t
    IFF
    log p_theta(x_t | x_{<t}) is high

This holds when the model is confident and correct. But it fails when:
1. The model assigns nearly equal probability to correct and incorrect
   tokens (low confidence but correct top-1 = high accuracy, moderate PPL)
2. The model is very confident but wrong (low accuracy, low PPL on the
   actual next token)

At micro scale with small vocabulary (V=42), the model often has "sharp"
distributions where a single wrong prediction costs a lot of PPL but
doesn't affect accuracy if it's on a prompt token.

## 4. Experimental Design

### 4.1 Domains

Five synthetic structured domains, each with deterministic correct answers:

| Domain | Format | Answer Type | Difficulty |
|--------|--------|-------------|------------|
| arithmetic | "A+B=C" | Integer sum | Variable (0-99) |
| reverse | "abc>cba" | String reversal | Length 2-6 |
| repeat | "ab*3=ababab" | Pattern repetition | Length 1-3, rep 2-4 |
| sort | "bca>abc" | Character sorting | Length 2-6 |
| parity | "1011>even" | Bit parity | Length 2-8 |

### 4.2 Training Protocol

1. Base model: trained on union of all 5 domains (2000 examples each,
   10000 total) for 30 epochs.
2. Expert i: LoRA (rank=8) trained on domain i only (2000 examples)
   for 40 epochs, base frozen.
3. Expert deltas merged into base weights for evaluation.

### 4.3 Evaluation

For each domain i and model variant (base, expert_i):
- PPL: computed on 500 held-out examples (full sequence)
- Accuracy: greedy decode on 200 fresh examples (answer portion only)

### 4.4 Statistical Test

Pearson correlation across N=5 (domain, expert) pairs. With N=5, the
critical value for significance at p<0.05 (one-tailed) is r >= 0.687.
Our kill criterion of r >= 0.5 is more lenient than statistical significance.

## 5. Worked Example

Suppose base model achieves:
- arithmetic: PPL=3.0, Acc=0.70
- reverse:    PPL=5.0, Acc=0.80

Expert_arithmetic achieves:
- arithmetic: PPL=2.5, Acc=0.90
- PPL_imp = (3.0-2.5)/3.0 = 0.167
- Acc_imp = 0.90 - 0.70 = 0.20

Expert_reverse achieves:
- reverse: PPL=6.0, Acc=0.90
- PPL_imp = (5.0-6.0)/5.0 = -0.20 (WORSE PPL!)
- Acc_imp = 0.90 - 0.80 = 0.10

Here PPL_imp and Acc_imp are negatively correlated: reverse got worse PPL
but better accuracy. This is the divergence scenario from Section 3.1 --
the expert learned to produce correct reversals but lost prompt modeling
quality (or became less "smooth" in its probability distribution).

## 6. Implications for SOLE Shadow Scoring

The SOLE evolution mechanism uses next-token perplexity as the quality
signal for shadow scoring (clone-and-compete). If PPL doesn't correlate
with task accuracy:

1. **Shadow scoring is misleading.** A clone with worse PPL but better
   task accuracy would be pruned.
2. **Answer-conditioned PPL** may be more appropriate: only compute PPL
   on tokens after the task delimiter.
3. **Task-specific metrics** may be necessary for the evolution signal,
   but this requires domain-specific evaluation infrastructure.
4. **The tension is fundamental:** PPL measures distributional fit, while
   accuracy measures functional correctness. A model that memorizes one
   correct answer perfectly has high accuracy but not necessarily low PPL
   across the full distribution.

## 7. Computational Cost

| Operation | FLOPs | Time (micro, CPU) |
|-----------|-------|-------------------|
| Base training (30 epochs x 10K examples) | ~1.2G | ~90s |
| Expert training (40 epochs x 2K examples) | ~500M per expert | ~25s each |
| PPL evaluation (500 examples) | ~15M per domain | <1s |
| Task accuracy (200 examples, greedy) | ~60M per domain | ~5s |
| Total per seed | ~4.5G | ~4 min |
| Total (3 seeds) | ~13.5G | ~12 min |
