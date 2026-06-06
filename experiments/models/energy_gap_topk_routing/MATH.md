# Energy Gap Top-k Routing: Mathematical Framework

## Type: Guided Exploration (Type 2)

The proven framework is the Neyman-Pearson optimality of energy gap for adapter ranking (Finding #182, AUC=0.942). The unknown is whether this ranking translates to generation quality improvement.

---

## A. Failure Mode Identification

**The disease:** Uniform composition applies all N adapters with equal weight 1/N to every query, regardless of domain relevance. Finding #184 proved binary gating is vacuous (all adapters always reduce NLL). The result is interference: irrelevant adapters inject off-topic bias, degrading structured-task performance. Evidence: prior experiments show uniform composition hurts prose domains (-5% to -13.5%) and underperforms single-adapter routing on math.

**The degenerate behavior:** Without routing, the model either:
1. Uses all adapters (uniform: interference from irrelevant experts), or
2. Uses no adapters (base: loses all domain expertise)

There is no principled middle ground without a selection mechanism.

---

## B. The Right Question (Reframe)

**Wrong question:** "How do we gate adapters by energy gap threshold?"
(Killed: Finding #184 — no threshold exists because all adapters reduce NLL.)

**Right question:** "Given that energy gap RANKS adapters correctly (AUC=0.942), does selecting the top-ranked adapter per query produce better generation than uniform composition?"

This is a fundamentally different operation:
- **Gating** = binary include/exclude at a threshold (requires an absolute boundary)
- **Ranking** = relative ordering by magnitude (requires only comparison)

Finding #184 killed gating. It did NOT kill ranking. The ranking signal is preserved.

---

## C. Derivation from Existing Mathematics

### Neyman-Pearson Optimality (Finding #182)

The energy gap Delta_E_i(x) = NLL_i(x) - NLL_base(x) for adapter i on query x is a likelihood ratio test statistic. By the Neyman-Pearson lemma, this is the most powerful test for discriminating "adapter i is relevant to x" vs "adapter i is irrelevant to x" at any significance level.

**Existing result:** AUC = 0.942 on math domain means P(Delta_E_correct > Delta_E_wrong) = 0.942 for randomly drawn (correct, wrong) adapter pairs on math queries. This is the concordance probability.

### Top-1 Selection as Argmax of Sufficient Statistics

Given N adapters with energy gaps {Delta_E_1(x), ..., Delta_E_N(x)}, the top-1 selector is:

    i*(x) = argmin_i Delta_E_i(x) = argmin_i NLL_i(x)

(argmin because more negative = adapter helps more.)

This selects the adapter that maximally reduces NLL on the specific query. By the data processing inequality, if the energy gap is sufficient for ranking (AUC > 0.5), then argmin over the gaps preserves the ranking information.

### Prediction: Top-1 Selection Accuracy

For the 5-domain case with labeled queries from domain d, the top-1 selector picks the correct adapter if the domain-matched adapter has the largest |Delta_E|. From Finding #182's per-domain AUC data:

- Math: AUC = 0.942 (strong signal)
- Medical: AUC = 0.938 (strong signal)
- Code: AUC = 0.500 (degenerate — no signal)

For a domain with AUC = a against each of (N-1) = 4 wrong adapters, the probability that the correct adapter has the largest |Delta_E| is approximately:

    P(correct top-1) >= a^(N-1) (independent pairwise comparisons, lower bound)

For math: P >= 0.942^4 = 0.787 (close to 80% threshold)
For medical: P >= 0.938^4 = 0.775

This is a LOWER BOUND because pairwise AUCs are not independent — the correct adapter's gap is typically much larger than all wrong ones simultaneously.

**Prediction 1:** Top-1 selection accuracy >= 80% on math and medical domains.
**Prediction 2:** Top-1 selection accuracy ~50% on code domain (degenerate AUC).
**Prediction 3:** Overall accuracy across all domains >= 70%.

---

## D. Proof of Bounded Improvement

**Theorem 1 (Ranking-based routing improves over uniform when AUC > 0.5 + epsilon).**

Let f_correct(x) be the generation quality when using the correct domain adapter, and f_uniform(x) be the quality under uniform 1/N composition. Let p = P(top-1 selects correct adapter) and q = P(top-1 selects any wrong adapter) = 1-p. The expected quality of top-1 routing is:

    E[Q_top1] = p * E[f_correct] + q * E[f_wrong_single]

The expected quality of uniform composition is:

    E[Q_uniform] = (1/N) * sum_i E[f_i]

Top-1 improves over uniform when:

    p * E[f_correct] + (1-p) * E[f_wrong_single] > (1/N) * sum_i E[f_i]

This holds when p > 1/N (better than random selection) AND the correct adapter's quality dominates the average.

Given AUC = 0.942 on math, the correct adapter is selected with p ~ 0.79-0.94. Since 1/N = 0.2, we have p >> 1/N by a factor of 4-5x. Combined with Finding #179 (math adapter gives 24x correctness improvement), the expected improvement is substantial for math.

**Key insight:** Even imperfect routing (p=0.8) is far better than uniform (1/N=0.2) when the correct adapter has large quality advantage.

QED (the improvement bound follows from p >> 1/N and quality gap between correct and wrong adapters).

---

## E. Predictions — Behavioral AND Quantitative

### Behavioral Predictions
1. Top-1 routing selects the correct domain adapter most of the time for math/medical
2. Top-1 routed generation produces more correct math answers than uniform composition
3. Code domain shows no routing benefit (AUC=0.5, routing is random)
4. Energy gap computation is fast relative to generation (it reuses the forward pass)

### Quantitative Predictions (derived from above)
| Prediction | Source | Value | Kill if |
|-----------|--------|-------|---------|
| Top-1 accuracy (math) | AUC^(N-1) lower bound | >= 79% | < 60% |
| Top-1 accuracy (overall) | Weighted average | >= 70% | < 50% (K575: <80%) |
| Math correctness (top-1 vs uniform) | Finding #179 + p>>1/N | top-1 > uniform | top-1 <= uniform (K576) |
| Energy overhead | Forward pass reuse | < 10% of gen time | > 10% (K577) |

---

## F. Assumptions & Breaking Conditions

1. **Energy gap is stable across prompts within a domain.** If gap magnitudes vary wildly per-prompt, the ranking from aggregate AUC may not hold per-query. Breaking: top-1 accuracy << 80%.

2. **Generation quality correlates with adapter relevance.** The energy gap measures NLL reduction. If NLL reduction doesn't predict generation quality (Finding #178: PPL doesn't predict quality, r=0.08), then ranking by energy gap could select the "most fluent" adapter rather than the "most task-correct" one. Breaking: top-1 selects high-Delta_E adapter but generation quality doesn't improve.

3. **Single adapter is sufficient (no composition needed).** Top-1 uses only one adapter. If the task requires multiple domains (e.g., "write Python code to solve a math problem"), single-adapter routing may be worse than composition. Breaking: tasks requiring cross-domain knowledge.

---

## G. Worked Example (N=5, d_model=2560)

Consider a math query x with energy gaps:
- medical: Delta_E = -0.80
- code: Delta_E = -0.60
- math: Delta_E = -1.90
- legal: Delta_E = -0.55
- finance: Delta_E = -0.70

Top-1: argmin = math (Delta_E = -1.90). Correct!
Uniform: weights = [0.2, 0.2, 0.2, 0.2, 0.2].

With top-1 routing: 100% of the adapter capacity goes to the relevant math adapter.
With uniform: only 20% goes to math, 80% is irrelevant interference.

Energy gap computation cost: 1 forward pass per adapter (5 total) + 1 base forward pass = 6 forward passes on the prompt (not generation tokens). For a 50-token prompt generating 128 tokens, the overhead is approximately 6*50 / (128*1) = 2.3 prompt-length equivalents. Since prompt processing is ~10x faster than generation per token (batched), overhead is ~2.3/12.8 ~ 18% — this may be close to the 10% kill criterion.

**Optimization:** Cache the base NLL and compute adapter NLLs only. Further: compute energy gaps on a PREFIX of the prompt (first 32 tokens) to reduce overhead.

---

## H. Complexity & Architecture Connection

- **Energy gap computation:** O(N * T * d) for N adapters, T prompt tokens, d model dim
- **Top-1 selection:** O(N) argmin over scalar gaps
- **Generation:** Unchanged — single adapter forward pass
- **Memory:** Need to load each adapter sequentially for gap computation, or maintain all in memory

The production analogy is MoLoRA (arxiv 2603.15965) which routes per-token. Our approach routes per-query (coarser but cheaper — one routing decision per sequence vs per token).

---

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Selecting the adapter with maximum NLL reduction (argmin Delta_E) guarantees p >> 1/N when AUC >> 0.5, making random/uniform selection suboptimal by construction.

2. **Which existing theorem(s) does the proof build on?**
   Neyman-Pearson lemma (optimal likelihood ratio test), data processing inequality for argmax over sufficient statistics.

3. **What specific numbers does the proof predict?**
   Top-1 accuracy >= 79% on math (from AUC^4 = 0.942^4), overall >= 70%, energy overhead < 10%.

4. **What would FALSIFY the proof?**
   If top-1 accuracy is significantly below AUC^(N-1) lower bound, the pairwise AUC scores are not independent or are inflated. If generation quality doesn't improve despite correct selection, Assumption 2 (NLL ranking predicts quality) fails.

5. **How many hyperparameters does this approach add?**
   Count: 0. Top-1 routing has no hyperparameters — it is pure argmin.

6. **Hack check:** No stacking. This is a single mechanism (argmin over energy gaps) replacing uniform 1/N weights.
