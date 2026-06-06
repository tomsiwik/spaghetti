# Clone-and-Compete Evolution: Mathematical Foundations

## 1. Setup and Notation

| Symbol | Definition | Shape/Value |
|--------|-----------|-------------|
| N | Number of experts on hash ring | 50 (pilot) |
| d | Model hidden dimension | 4096 (Qwen2.5-7B) |
| r | LoRA rank | 16 |
| theta | Base model parameters | ~7B (frozen) |
| Delta_i | Expert i's LoRA delta: B_i @ A_i | R^{d x d} per module |
| Delta_i' | Clone of expert i after correction fine-tuning | R^{d x d} per module |
| Q | Number of tournament queries | variable (convergence-dependent) |
| S_i(x) | Scoring function for expert i on input x | R (scalar, lower = better) |
| C_i | Set of correction examples for expert i | |C_i| = 50 |
| E_i | Set of evaluation examples for expert i | |E_i| = 200 |
| alpha | Fine-tuning learning rate | 1e-4 |
| T_ft | Fine-tuning steps for clone | 50 |

## 2. The Clone-and-Compete Protocol

### 2.1 Clone Creation

Given expert i with adapter Delta_i = B_i @ A_i, the clone Delta_i' is created by:

1. **Copy**: Delta_i' <- Delta_i (identical weights)
2. **Correct**: Fine-tune Delta_i' on correction set C_i for T_ft steps

The correction set C_i is constructed by selecting examples where expert i
performs worst (highest loss), ordered by decreasing loss:

    C_i = argsort_{x in E_i}[-L(theta + Delta_i, x)][:50]

where L(theta + Delta_i, x) = -(1/T) sum_{t=1}^{T} log p_{theta+Delta_i}(x_t | x_{<t})

### 2.2 Fine-Tuning the Clone

The clone is fine-tuned with standard SFT on the correction examples:

    Delta_i' = argmin_{Delta'} sum_{x in C_i} L(theta + Delta', x)

starting from Delta_i' = Delta_i (warm start).

**Key insight**: Because the clone starts from the original adapter, it only
needs to learn the *correction delta*:

    Delta_i' = Delta_i + epsilon_i

where epsilon_i is the correction term. With T_ft = 50 steps and lr = 1e-4,
the magnitude ||epsilon_i|| << ||Delta_i||, ensuring the clone remains close
to the original in weight space.

### 2.3 Tournament Scoring

We use within-domain answer-conditioned PPL as the scoring signal.

**Critical context**: Answer-only PPL was KILLED at macro scale for CROSS-domain
comparison (r = -0.63). However, clone-and-compete only compares two adapters
on the SAME domain. Within-domain PPL ranking is hypothesized to be preserved
(exp_relative_ppl_within_domain, open). This experiment implicitly tests that
hypothesis.

For a sequence x = (x_1, ..., x_T) with prompt tokens x_{1..d*} and answer
tokens x_{d*+1..T}:

    S_i(x) = PPL_answer(theta + Delta_i, x)
           = exp(-(1/T_a) sum_{t=d*+1}^{T} log p_{theta+Delta_i}(x_t | x_{<t}))

where T_a = T - d* is the number of answer tokens.

The clone wins if it achieves lower average PPL on the evaluation set:

    clone_wins <=> (1/|E_i|) sum_{x in E_i} S_i'(x) < (1/|E_i|) sum_{x in E_i} S_i(x)

### 2.4 Convergence Analysis

The tournament must determine a winner with statistical confidence. We model
the PPL difference per query as:

    delta_q = S_i(x_q) - S_i'(x_q)

The mean delta after Q queries is:

    delta_bar_Q = (1/Q) sum_{q=1}^{Q} delta_q

With variance sigma^2 = Var(delta_q), the standard error is sigma / sqrt(Q).
The tournament converges when the t-statistic exceeds the significance threshold:

    |delta_bar_Q| / (sigma / sqrt(Q)) > z_{alpha/2}

For z_{0.025} = 1.96 (95% confidence), the minimum queries to detect effect
size mu = E[delta_q]:

    Q_min = ceil((z_{alpha/2} * sigma / mu)^2)

**Worked example at macro scale:**
- If the correction improves PPL by mu = 0.5 nats (moderate effect)
- And per-query variance sigma^2 = 4.0 (sigma = 2.0)
- Then Q_min = ceil((1.96 * 2.0 / 0.5)^2) = ceil(61.5) = 62 queries

This is well within the 50K query budget. Even for a tiny effect (mu = 0.1):
- Q_min = ceil((1.96 * 2.0 / 0.1)^2) = ceil(1537) = 1,537 queries

Still far below the 50K threshold.

### 2.5 Regression Bound

During the tournament, both original and clone serve traffic. The question is
whether the clone's presence degrades the original's domain.

**In pre-merge composition**: If the clone replaces the original in the merged
weight matrix, the composition changes by:

    W_composed = W_base + sum_{j != i} Delta_j + Delta_i'
               = W_composed_original + (Delta_i' - Delta_i)
               = W_composed_original + epsilon_i

The regression is bounded by ||epsilon_i|| relative to ||W_composed||.

**In shadow scoring mode** (this experiment): The original expert is untouched.
The clone is only *scored*, not composed. Therefore K3 (regression < 2%)
measures the regression from *replacing* the original with the clone after
the tournament concludes, not during it.

For measurement, we compute:

    regression_pct = (PPL_clone_general - PPL_orig_general) / PPL_orig_general * 100

where PPL_*_general is measured on the domain's general evaluation set (not
the correction set).

## 3. Statistical Framework for Kill Criteria

### 3.1 K1: Clone Win Rate > 70%

Under the null hypothesis H0: corrections don't help (clone and original are
equally good), the win rate follows Binomial(n_domains, 0.5).

For n = 5 domains, P(>= 4 wins | p=0.5) = C(5,4)/32 + C(5,5)/32 = 6/32 = 0.1875.
For n = 5 domains, P(>= 5 wins | p=0.5) = 1/32 = 0.03125.

With 5 domains, we need at least 4/5 wins to exceed 70%. The one-sided
p-value for 4/5 is 0.19 (not significant). For 5/5: p = 0.031 (significant
at alpha = 0.05).

**Interpretation**: With only 5 domains, K1 at >70% is a lenient threshold.
5/5 would be strong evidence; 4/5 is suggestive but not definitive.

### 3.2 K2: Convergence < 50K Queries

We measure convergence by the smallest subset size at which the winner
is stable. With evaluation sets of 200 queries per domain, we check at
checkpoints [50, 100, 200] queries.

The 50K threshold is 250x our evaluation budget. If the winner is clear at
200 queries, we extrapolate that 200 << 50K and K2 passes trivially.

If the winner flips between checkpoints, we flag instability and note the
estimated Q_min from Section 2.4.

### 3.3 K3: Domain Regression < 2%

We compare clone PPL on general queries vs original PPL on general queries.
The 2% threshold means:

    max over domains (PPL_clone - PPL_orig) / PPL_orig < 0.02

This is a strict per-domain bound: even one domain exceeding 2% triggers K3.

## 4. Cost Analysis

| Operation | Time | GPU Cost |
|-----------|------|----------|
| Load base model (4-bit) | ~30s | - |
| Generate corrections (per domain) | ~2 min | ~$0.01 |
| Clone + fine-tune (50 steps) | ~1 min | ~$0.005 |
| Score original (200 queries) | ~3 min | ~$0.01 |
| Score clone (200 queries) | ~3 min | ~$0.01 |
| **Per domain total** | ~10 min | ~$0.035 |
| **5 domains total** | ~50 min | ~$0.17 |
| **With overhead + base load** | ~75 min | ~$0.20 |

Budget impact: ~$0.20 out of ~$28 remaining. Well within budget.

## 5. Relationship to Prior Work

### 5.1 Evolutionary Model Merging (Sakana AI, 2024)

Sakana's evolutionary optimization operates in parameter space AND data flow
space, using population-based search over merging recipes. Key differences:

- **Sakana**: Population of N>>2 candidates, evaluated on benchmarks, crossover+mutation
- **SOLE clone-compete**: Binary tournament (original vs clone), evaluated on domain traffic

SOLE is simpler: no crossover, no mutation, no population management. The
"evolution" is directed (corrections point toward better solutions) rather
than random search.

### 5.2 Self-Play Fine-Tuning (SPIN, 2024)

SPIN trains a model to distinguish its own outputs from human-generated data.
Each iteration produces a stronger discriminator. Differences:

- **SPIN**: Same model improves by self-play against previous version
- **SOLE**: Different adapter (clone with corrections) competes against original

SPIN's convergence guarantee (the fixed point is the target distribution)
does not directly apply to SOLE because the clone is trained on corrections,
not on self-play discrimination. However, the principle of improvement through
competition is shared.

### 5.3 Online Learning Regret Bounds

In the tournament framework, each adapter is an "arm" in a two-armed bandit.
The expected regret after Q queries of using the wrong arm is:

    R(Q) = Q * |mu_1 - mu_2| * P(wrong arm after Q queries)

With sequential hypothesis testing (SPRT), the expected number of queries
to a correct decision is:

    E[Q] = (1/KL(p || q)) * log(1/alpha)

where p,q are the PPL distributions under the two adapters and alpha is
the error probability. For well-separated adapters (mu difference > 1 nat),
this is typically O(10-100) queries.

## 6. Assumptions and Limitations

1. **Within-domain PPL ranking preserved at macro**: Unverified. Cross-domain
   PPL correlation is anti-correlated (r=-0.63), but clone-compete is strictly
   within-domain. If within-domain PPL ranking also fails, the tournament
   mechanism is fundamentally broken.

2. **50 correction steps suffice**: Based on micro correction_signal_quality
   simulation showing teacher corrections improve quality at 200 steps per
   source per domain. With 50 high-loss examples and 50 steps, we rely on
   warm-start from the original adapter to compensate for fewer steps.

3. **Training data tail as ground truth corrections**: We use the expert's
   highest-loss examples from training data as "corrections." This assumes
   the training data is correct (it was generated by a 70B teacher). This
   is a cleaner signal than real-time corrections but does not test the
   full correction pipeline.

4. **5 domains may be insufficient**: With n=5, the binomial test has low
   power. A 4/5 win rate (80%) has p=0.19 under H0, not statistically
   significant. The experiment is directional, not definitive.

5. **PPL as proxy for quality**: Even within-domain, PPL may not track actual
   task performance. The experiment measures PPL improvement but does not
   verify that lower PPL translates to better task accuracy at macro scale.
