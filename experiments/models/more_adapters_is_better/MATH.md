# More Adapters Is Better: Mathematical Foundations

## 1. Mechanism Definition

We compose N ternary LoRA adapters via per-expert A_i, B_i pairs with uniform 1/N scaling:

    y = W_base * x + (alpha / N) * sum_{i=1}^{N} (x @ A_i) @ B_i^q

where:
- W_base in R^{d_out x d_in}: frozen ternary base weight (BitNet-2B-4T, d=2560)
- A_i in R^{d_in x r}: frozen Grassmannian-initialized projection (r=16)
- B_i^q in R^{r x d_out}: STE-quantized ternary B matrix
- alpha = 20.0: LoRA scaling factor
- N: number of adapters in the pool

Per-domain PPL under **routed** composition (top-k routing, k adapters selected):

    y_routed(x) = W_base * x + (alpha / k) * sum_{j in S(x)} (x @ A_j) @ B_j^q

where S(x) = {j : routing_head_j(h(x)) > 0.5}, |S(x)| = k (Gumbel-sigmoid selection).

**Key distinction from uniform composition:** Under routing, adding adapter N+1 only affects
tokens where routing_head_{N+1} fires. For tokens in domains 1..N, the selected subset S(x)
is unchanged, so PPL should be unchanged. This is the mechanism we test.

## 2. Why It Works (The Independence Argument)

**Claim:** Under ideal routing (accuracy ~1.0), adding adapter N+1 cannot degrade domains 1..N.

**Proof sketch:**
Let PPL_k(d) = perplexity on domain d when k adapters are in the pool.
Under perfect routing for domain d tokens: S(x) = {d} (only the correct adapter fires).

Then: PPL_k(d) = PPL_{k+1}(d) = exp(L_base - delta_d)

where delta_d is the loss reduction from adapter d, independent of other adapters in the pool.

**When routing is imperfect (accuracy < 1.0):**
- False positives: irrelevant adapter j fires for domain d tokens.
  Impact: dilution of correct signal by factor k/(k+1) per false positive.
- False negatives: correct adapter d fails to fire.
  Impact: domain d gets base-only prediction.

The Grassmannian skeleton guarantees A_i^T A_j approx 0 (mean |cos| = 0.024 at N=24).
When a wrong adapter fires, its contribution (x @ A_j) @ B_j^q is nearly orthogonal to
the correct adapter's contribution, so interference is bounded:

    ||interference|| <= (alpha/k) * ||x|| * ||A_j^T A_d|| * ||B_j|| * ||B_d|| / (||A_d|| * ||B_d||)

At mean |cos(A_i, A_j)| = 0.024, this is ~2.4% of the correct signal.

## 3. What Breaks It

**Routing degradation at scale:** As N grows, the negative class for each routing head
becomes more diverse. If new domains are similar to existing ones (e.g., psychology vs
sociology), routing accuracy drops, increasing false positive rates.

From existing data: routing accuracy ranges from 0.85 (sports, engineering) to 1.0 (math).
Mean routing accuracy = 0.927. At this accuracy, ~7.3% of tokens get misrouted.

**Kill condition K1:** If average system PPL (across all in-pool domains) stops improving
or degrades after N=10, the thesis fails. This would mean interference from misrouting
outweighs the benefit of new adapters.

**Kill condition K2:** If any domain 1..10 regresses >5% when domains 11..24 are added,
the independence argument fails in practice.

## 4. Assumptions

1. **Routing heads are adequate.** Justified: 24/24 heads trained with >85% accuracy.
   If wrong: false positives cause cross-domain interference.

2. **Grassmannian orthogonality holds.** Justified: mean |cos| = 0.024 at N=24, well
   below the Welch bound. Capacity N_max = (2560/16)^2 = 25,600 >> 24.
   If wrong: adapter interference grows with N.

3. **Domain-specific adapters help their domain.** Justified: 24/24 domains show
   specialization improvement (17-47% PPL reduction vs base).
   If wrong: adapters are noise, adding more adds nothing.

## 5. Complexity Analysis

- Per eval token: O(k * d_in * r + k * r * d_out) for k routed adapters
- Memory: O(N * L * P * r * (d_in + d_out)) for N adapters, L layers, P projections
  At N=24: 24 * 30 * 7 * 16 * (2560 + 2560) = ~1.3 GB adapter storage
- Routing: O(N * d_hidden * h_hidden) per token for N heads
  At N=24, d=2560, h=32: negligible

## 6. Worked Example (from real data)

At N=24 with uniform composition:
- Base avg PPL: 10.076
- Composed avg PPL: 6.902 (gamma = 6.902/10.076 = 0.685, 31.5% improvement)
- Individual (oracle) avg PPL: 6.293 (gap from uniform: 6.902/6.293 = 1.097)

Under routed composition (oracle = individual adapter per domain):
- Expected avg PPL at routing accuracy 0.93: ~6.293 * (1 + 0.07 * epsilon)
  where epsilon is the interference per misrouted token

The experiment measures whether avg PPL monotonically improves as N grows:
N=5: avg across 5 domains with 5-adapter pool
N=10: avg across 10 domains with 10-adapter pool
N=15: avg across 15 domains with 15-adapter pool
N=20: avg across 20 domains with 20-adapter pool
N=24: avg across 24 domains with 24-adapter pool

## 7. Connection to Architecture

This experiment validates the core SOLE (Sparse Orthogonal LoRA Experts) thesis:
the system improves monotonically with N because each adapter is independent.

Production analog: DeepSeek-V3 uses 256 routed experts with auxiliary-loss-free
load balancing. Our Grassmannian skeleton replaces their expert-level FFN with
LoRA adapters, achieving similar independence via geometric orthogonality rather
than learned routing. The key difference: DeepSeek experts share the same
parameter space and compete for capacity; our experts occupy orthogonal subspaces
and compose additively.
