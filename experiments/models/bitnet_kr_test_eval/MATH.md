# KR-Test Evaluation: Mathematical Foundations

## Setup

We adapt the KR-Test protocol (Ziabari et al., arXiv:2601.03505) for LoRA adapter
quality evaluation on BitNet-2B-4T.

### Variables

| Symbol | Shape/Type | Description |
|--------|------------|-------------|
| x_c | (T_c,) | Context tokens (instruction prompt) |
| x_t^+ | (T^+,) | Correct continuation tokens |
| x_t^- | (T^-,) | Wrong continuation tokens |
| T_min | scalar | min(T^+, T^-) — truncation length |
| p_theta | R^V | Model probability distribution (base + adapter) |
| S^+ | scalar | Cumulative log-prob of correct continuation |
| S^- | scalar | Cumulative log-prob of wrong continuation |

### KR-Test Score

For a single contrastive pair (x_c, x_t^+, x_t^-):

S^+ = sum_{t=1}^{T_min} log p_theta(x_{T_c+t}^+ | x_c, x_{T_c+1:T_c+t-1}^+)
S^- = sum_{t=1}^{T_min} log p_theta(x_{T_c+t}^- | x_c, x_{T_c+1:T_c+t-1}^-)

Correct classification: S^+ > S^-

KR-Test score = (1/N) sum_{i=1}^N 1[S_i^+ > S_i^-]

### Contrastive Pair Generation

The original paper uses a teacher LLM for semantic segmentation and contrastive
generation. We use cross-item pairing ($0, no teacher):

For domain D with items {(q_i, a_i)}, pair (q_i, a_i, a_j) where j is a distant
index (offset by n/3) within the same domain. The wrong answer a_j is:
- Fluent and domain-appropriate (it is a real answer to a real question)
- Factually incorrect for q_i (it answers a different question)
- Not trivially distinguishable by surface cues alone

This is harder than rule-based perturbation (number swaps, entity replacements)
because the wrong answer shares domain vocabulary and grammatical structure.

### Correlation Analysis

To test K1 (correlation with task accuracy), we compute:

KR-delta_d = KR(adapter_d, domain_d) - KR(base, domain_d)

This delta measures the adapter's marginal improvement on its own domain.
We compare this against the task accuracy delta from instruction_task_eval:

Task-delta_d = TaskMetric(adapter_d, domain_d) - TaskMetric(base, domain_d)

Spearman rank correlation rho tests whether the ordering is preserved.

### Statistical Power

With n=50 samples per domain and p=0.9 baseline accuracy:
- SE(p) = sqrt(p(1-p)/n) = sqrt(0.09/50) = 0.042
- For a 5pp improvement to be significant at alpha=0.05: need z=1.96, so
  delta >= 1.96 * 0.042 = 0.083 (8.3pp)
- Our observed deltas (0-10pp) are at the edge of detectability
- Larger sample sizes (n=200+) would improve power substantially

### Noise Floor Definition

Two candidate noise floor definitions:

1. **Distance from chance**: |KR(random) - 0.5| = |0.895 - 0.5| = 0.395
   This is dominated by the pretrained base model's existing knowledge.
   Not useful for adapter discrimination.

2. **Random adapter deviation from base**: |KR(random) - KR(base)| = 0.000
   Random adapter (zero B matrix) is functionally identical to base.
   This is the correct noise floor: any non-zero delta is signal.

3. **Statistical noise floor**: SE of KR score at p=0.9, n=50 = 0.042
   An adapter must improve by >0.042 to be distinguishable from base
   at one standard error.

We use definition (3) for the kill criterion: discrimination ratio =
mean(KR_delta_trained) / SE = 0.055 / 0.042 = 1.3. At n=200, this
would be 0.055 / 0.021 = 2.6, clearing the 2x threshold.

## Worked Example

Domain: Code (hardest, base=74%)

Base model: 37/50 correct (p=0.74)
Code adapter: 41/50 correct (p=0.82)
Random adapter: 37/50 correct (p=0.74)

KR-delta(code adapter on code) = 0.82 - 0.74 = 0.08
Task-delta(code) = 1.0 - 0.9 = 0.10 (syntax validity)

Both metrics identify code as a domain where the adapter adds value.
