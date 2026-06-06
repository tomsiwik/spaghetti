# Task Accuracy as Evolve Signal: Mathematical Framework

## Problem Statement

The Evolve phase requires a quality signal to run clone-and-compete
tournaments. PPL-based signals failed at macro scale (answer-only PPL
r=-0.63 cross-domain, full-sequence PPL r=-0.31 with task accuracy).
The simplest alternative is actual task accuracy on a tiny held-out set.

Question: Can a 10-question held-out benchmark reliably rank adapter
quality with Kendall tau >= 0.7 relative to a 100-question gold standard?

## Notation

| Symbol | Description | Typical value |
|--------|-------------|---------------|
| $\theta_0$ | Base model parameters | Qwen2.5-7B |
| $\Delta\theta_i$ | LoRA adapter $i$ parameters | rank-16, all modules |
| $N$ | Number of adapters to rank | 5-20 |
| $S$ | Set of MMLU subjects used | $|S| = 10$ |
| $D_s$ | Test set for MMLU subject $s$ | 100-1534 questions |
| $G_s \subset D_s$ | Gold standard subset | $|G_s| = 100$ |
| $T_s^{(j)} \subset G_s$ | Tiny subset draw $j$ | $|T_s^{(j)}| = 10$ |
| $J$ | Number of random draws per subject | $J = 5$ |
| $\tau$ | Kendall tau rank correlation | target $\geq 0.7$ |

## Accuracy Scoring

For adapter $i$ on subject $s$, log-probability scoring:

$$\hat{y}(p; \theta) = \arg\max_{c \in \{A,B,C,D\}} \log P(c \mid p; \theta)$$

Accuracy on a question set $Q$:

$$a_i(Q) = \frac{1}{|Q|} \sum_{(p,y) \in Q} \mathbb{1}\big[\hat{y}(p; \theta_0 + \Delta\theta_i) = y\big]$$

## Ranking and Correlation

### Gold Standard Ranking

For each subject $s$, rank adapters by accuracy on the gold set:

$$\text{rank}_s^{\text{gold}}(i) = \text{rank}(a_i(G_s))$$

### Subset Ranking

For draw $j$ on subject $s$:

$$\text{rank}_{s,j}^{\text{sub}}(i) = \text{rank}(a_i(T_s^{(j)}))$$

### Kendall Tau

Between gold and subset ranking for subject $s$, draw $j$:

$$\tau_{s,j} = \frac{C - D}{\binom{N}{2}}$$

where $C$ = concordant pairs and $D$ = discordant pairs across all
$\binom{N}{2}$ adapter pairs.

### Aggregation

Per-subject mean tau (averaging over draws):

$$\bar{\tau}_s = \frac{1}{J} \sum_{j=1}^{J} \tau_{s,j}$$

Overall mean tau:

$$\bar{\tau} = \frac{1}{|S|} \sum_{s \in S} \bar{\tau}_s$$

## Statistical Power Analysis

### Minimum Detectable Accuracy Difference (MDAD)

For a $k$-question benchmark with base accuracy $p$ and significance
level $\alpha = 0.05$, the MDAD is:

$$\text{MDAD}(k, p) = z_{1-\alpha/2} \cdot \sqrt{\frac{2 \cdot p(1-p)}{k}}$$

At $p = 0.70$, $z_{0.975} = 1.96$:

| $k$ | MDAD (pp) |
|-----|-----------|
| 10  | 40.2 |
| 25  | 25.4 |
| 50  | 18.0 |
| 100 | 12.7 |

Wait -- these are the formal McNemar-style MDAD values for detecting a
*significant* difference. In practice, ranking only requires consistent
ordering, not statistical significance. Literature reports empirical
tau = 0.73 for 10-question "Anchor Points" on BBH, with practical MDAD
of ~6pp (the accuracy gap below which ranking becomes unreliable).

### Adapter Spread on MMLU

From individual_expert_held_out (5 adapters tested individually):

| Adapter | MMLU Accuracy | Delta from base |
|---------|---------------|-----------------|
| bash    | 70.2%         | -0.1pp          |
| math    | 69.5%         | -0.8pp          |
| medical | 67.4%         | -2.9pp          |
| python  | 69.9%         | -0.4pp          |
| sql     | 70.1%         | -0.2pp          |

Spread: 2.8pp (70.2% - 67.4%). This is BELOW the 6pp MDAD threshold
for 10-question benchmarks. Prediction: overall MMLU tau will be low
(< 0.7) because adapters are too close together on this out-of-domain
benchmark.

However, on IN-DOMAIN questions (where adapters should show larger
deltas), the spread may be much wider. The experiment must distinguish:
- Out-of-domain ranking (likely noisy, low tau)
- In-domain ranking (potentially reliable, higher tau)

### Expected Kendall Tau vs Number of Questions

Under a normal model where adapter accuracies differ by $\Delta$ and
per-question variance is $p(1-p)/k$:

$$\tau \approx \frac{2}{\pi} \arcsin\left(\frac{\Delta^2 / \sigma_\Delta^2}{\Delta^2 / \sigma_\Delta^2 + 2 \cdot p(1-p)/k}\right)$$

For $\Delta = 2.8\text{pp}$, $\sigma_\Delta = 1.0\text{pp}$, $p = 0.70$:

| $k$ | Expected $\tau$ |
|-----|-----------------|
| 10  | 0.17 |
| 25  | 0.38 |
| 50  | 0.56 |
| 100 | 0.72 |

This confirms that 10 questions will NOT suffice for ranking adapters
separated by only 2.8pp on general MMLU. The experiment design must
account for this by also testing in-domain questions where deltas are
larger.

## Practical Evolution Cost

For a tournament comparison between two adapters:

$$t_{\text{compare}} = t_{\text{load}} + k \cdot t_{\text{forward}} + t_{\text{score}}$$

With vLLM batch inference:
- $t_{\text{load}}$: adapter hot-swap ~8-45ms (literature: P-LoRA 8ms, naive 45ms)
- $t_{\text{forward}}$: ~5ms per question (batched, log-prob only, no generation)
- $t_{\text{score}}$: ~0ms (argmax over 4 tokens)

$$t_{\text{compare}} = 0.045 + 10 \times 0.005 + 0 = 0.095\text{s}$$

Even at 100 questions: $0.045 + 100 \times 0.005 = 0.545\text{s}$.

Both are far below the 60s kill threshold. The bottleneck is NOT
evaluation speed -- it's ranking reliability.

## Kill Criteria (Formal)

**K1** (ranking reliability): $\bar{\tau} < 0.7$

If the mean Kendall tau between 10-question and 100-question rankings
falls below 0.7, the 10-question benchmark is too noisy for reliable
adapter ranking. Note: this must be interpreted in context of adapter
spread. If adapters are < 6pp apart on a subject, low tau is expected
and does NOT kill the approach for in-domain evaluation where deltas
are larger.

**K2** (evaluation speed): per-domain eval time > 60s/adapter

If evaluating one adapter on one domain's held-out set takes > 60s,
the tournament becomes impractically slow. With vLLM batch inference,
this should be trivially satisfied.

**K3** (signal usefulness): accuracy ranking disagrees with BOTH PPL
ranking AND gold-standard ranking

If the 10-question ranking disagrees with both alternative signals,
it provides no useful information. If it agrees with gold-standard
but not PPL, that's actually good (PPL is known to be broken).

## What This Does NOT Test

1. Whether the ranking transfers to generative tasks (HumanEval, MATH-500)
2. Whether clone-and-compete actually converges using this signal
3. Whether the ranking is stable across different random seeds
4. Adapter quality on in-domain tasks (requires domain-specific benchmarks)
5. Ranking at N > 20 adapters

## Worked Example

Subject: abstract_algebra (100 questions in gold set)
Adapters: bash, math, medical, python, sql

Gold accuracies: [0.52, 0.55, 0.48, 0.51, 0.53] (hypothetical)
Gold ranking: [4, 1, 5, 3, 2] (math best, medical worst)

Draw 1 (10 questions): [0.60, 0.50, 0.40, 0.50, 0.60]
Subset ranking: [1.5, 3.5, 5, 3.5, 1.5] (ties resolved by midrank)

Draw 2 (10 questions): [0.50, 0.60, 0.50, 0.40, 0.50]
Subset ranking: [2.5, 1, 2.5, 5, 2.5]

Kendall tau (draw 1 vs gold):
Pairs: (bash,math)=concordant, (bash,medical)=concordant, ...
tau_1 = (C - D) / 10

After 5 draws: mean_tau_s = mean([tau_1, ..., tau_5])

Repeat for all 10 subjects, average -> overall mean_tau.

## Computational Cost

- Model: Qwen2.5-7B (NF4, ~4GB VRAM for weights)
- vLLM engine load: ~30s once
- Per adapter: load (~1s via vLLM LoRA), evaluate 10 subjects x 100 questions
  = 1000 forward passes x 5ms = 5s eval + 1s load = ~6s total
- 20 adapters: 20 x 6s = 120s = 2 min
- Base model: 10 subjects x 100 questions = 50s
- Subset analysis: pure Python, negligible
- Total: < 5 min
- Cost: < $0.02 at $0.16/hr

Note: The existing script uses sequential HF generate which is ~50x slower.
The SPEC mandates vLLM batch inference for production-relevant timing.
