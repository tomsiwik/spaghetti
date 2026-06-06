# Leave-One-Out Expert Ranking: Mathematical Foundations

## Variables and Notation

| Symbol | Shape / Type | Description |
|--------|-------------|-------------|
| W | (d, d') per layer | Base model frozen weight matrices |
| delta_i = B_i A_i | (d, d') per layer | LoRA expert i's weight delta (rank r) |
| N | scalar | Number of composed experts (N=50 for pilot) |
| r | scalar | LoRA rank (r=16 for pilot adapters) |
| d | scalar | Model hidden dimension (d=3584 for Qwen2.5-7B) |
| T | scalar | Number of calibration texts per set |
| x_t | (seq_len,) | Calibration text t (tokenized) |
| L_t | scalar | Sequence length of text t |
| PPL(M, S) | scalar | Perplexity of model M on calibration set S |
| PPL_N | scalar | PPL of model with all N experts composed |
| PPL_{-i} | scalar | PPL of model with expert i removed |
| Delta_i | scalar | Relative PPL change from removing expert i |
| sigma_Delta | scalar | Std deviation of {Delta_i} across i=1..N |
| tau | scalar | Kendall rank correlation between rankings on sets A vs B |
| cos_ij | scalar | Cosine similarity between delta_i and delta_j (flattened) |

## 1. Composition via Weight-Space Merge

The composed model at layer l has effective weights:

```
W_composed^l = W^l + sum_{i=1}^{N} B_i^l A_i^l
```

The leave-one-out model (without expert i) has:

```
W_{-i}^l = W^l + sum_{j != i} B_j^l A_j^l = W_composed^l - B_i^l A_i^l
```

This is exact: no approximation from the subtraction because LoRA composition
is linear in weight space.

## 2. Perplexity Measurement

For a calibration text x = (x_1, ..., x_L), the cross-entropy loss is:

```
CE(M, x) = -(1/L) sum_{t=1}^{L} log P_M(x_{t+1} | x_{1:t})
```

Perplexity is:

```
PPL(M, x) = exp(CE(M, x))
```

Over a calibration set S = {x_1, ..., x_T}:

```
PPL(M, S) = exp( (1 / sum_t L_t) * sum_t sum_{j=1}^{L_t} -log P_M(x_{t,j+1} | x_{t,1:j}) )
```

This is the standard token-weighted perplexity (each token contributes equally,
regardless of which text it belongs to).

## 3. Leave-One-Out Delta

The relative PPL change from removing expert i, evaluated on calibration set S:

```
Delta_i(S) = (PPL_{-i}(S) - PPL_N(S)) / PPL_N(S) * 100%
```

**Interpretation:**
- Delta_i > 0: removing expert i INCREASES PPL (expert is helpful)
- Delta_i < 0: removing expert i DECREASES PPL (expert is harmful)
- Delta_i ~ 0: expert i has negligible impact

**Crucial distinction from KL divergence:** PPL measures absolute model quality
on the calibration text. KL divergence (KILLED, rho=-0.7) measured distributional
distance from base. A strong expert that shifts distributions far from base has
high KL but may have GOOD PPL. PPL directly captures "does this expert make the
model better at predicting text?" without reference to the base model's distribution.

## 4. Ranking and Kill Criteria

### K1: Ranking Variance (sigma_Delta >= 0.1%)

Compute the standard deviation of all LOO deltas:

```
sigma_Delta = std({Delta_1(S_A), Delta_2(S_A), ..., Delta_N(S_A)})
```

If sigma_Delta < 0.1%, all experts contribute approximately equally and the
ranking is indistinguishable from noise.

**Expected behavior under SOLE orthogonality:**

Under perfect orthogonality (cos_ij = 0 for all i != j), each expert's
contribution is independent. The PPL change from removing expert i depends only
on delta_i's effect on the calibration text. Since pilot adapters were trained
on diverse domains (python, math, medical, SQL, bash, etc.), we expect:

- Experts trained on domains represented in the calibration text will have
  Delta_i > 0 (removing them hurts)
- Experts on unrepresented domains may have Delta_i ~ 0 or slightly negative
  (their off-domain perturbation adds noise)

The key prediction: sigma_Delta should reflect the DIVERSITY of expert domains
relative to the calibration set. With 50 diverse experts and balanced calibration
text, we expect sigma_Delta >> 0.1%.

**Failure mode:** If all 50 experts produce identical PPL deltas, it means either:
(a) the calibration text is pathological (equally sensitive to all domains), or
(b) all experts have identical impact magnitude (unlikely given 42% mean PPL
improvement with high variance across domains in pilot-50 benchmark).

### K2: Runtime (< 4 hours for N=50)

Total computation:

```
Cost = T_load_base + T_compose_all + T_eval(S_A) + T_eval(S_B)
     + N * (T_compose_{N-1} + T_eval(S_A) + T_eval(S_B))
```

**Estimated timings (A5000, 4-bit NF4 Qwen2.5-7B):**

| Operation | Time (est.) | Count | Total |
|-----------|------------|-------|-------|
| Load base model | ~20s | 1 (reuse) | 20s |
| Compose N adapters (CPU merge) | ~10s | 1 | 10s |
| Compose N-1 adapters (CPU merge) | ~10s | 50 | 500s |
| PPL eval on T=20 texts | ~5s | 102 (1 ref + 50 LOO, x2 sets) | 510s |
| Load merged adapter (PEFT) | ~3s | 51 | 153s |
| Total | | | ~1193s (~20 min) |

**Critical optimization: keep base model loaded.** The KL health experiment
was killed partly because it reloaded the base model for every LOO (45s each).
We must:
1. Load base model ONCE into GPU memory
2. For each LOO: merge N-1 adapters on CPU, save to disk, load as PEFT adapter
3. Evaluate both calibration sets
4. Unload PEFT adapter (keep base model)
5. Repeat

With this approach, K2 should pass easily (~20 min << 4 hrs).

**Alternative optimization: subtraction approach.** Since W_{-i} = W_composed - delta_i,
we could:
1. Merge all N adapters into base weights once (merge_and_unload)
2. For expert i, SUBTRACT delta_i from the merged weights
3. Evaluate
4. ADD delta_i back

This avoids all CPU merge + disk I/O, reducing to pure GPU operations.
Risk: floating-point accumulation errors from repeated add/subtract cycles.
Mitigate by re-merging from scratch every K iterations (e.g., K=10).

### K3: Ranking Stability (Kendall tau >= 0.5)

Given two disjoint calibration sets S_A and S_B, compute independent rankings:

```
rank_A = argsort({Delta_1(S_A), ..., Delta_N(S_A)})
rank_B = argsort({Delta_1(S_B), ..., Delta_N(S_B)})
```

Kendall tau-b:

```
tau = (concordant_pairs - discordant_pairs) / sqrt((n0 - n1)(n0 - n2))
```

where n0 = N*(N-1)/2 total pairs, n1 = ties in rank_A, n2 = ties in rank_B.

**Expected behavior:** If LOO-PPL captures genuine expert quality differences,
rankings should be stable across different calibration texts. The main risk is
that calibration set composition introduces domain bias (e.g., set A has more
code tokens, favoring code experts).

**Mitigation:** Use stratified calibration sets. Both S_A and S_B should have
the same domain mix (code, math, general, creative, QA). This is achieved by
hardcoding diverse texts and assigning alternating texts to each set.

**Statistical power:** For N=50, Kendall tau has good power. Under H0 (rankings
are independent), tau ~ N(0, sqrt(2(2N+5)/(9N(N-1)))). For N=50:

```
sigma_tau = sqrt(2*105 / (9*50*49)) = sqrt(210/22050) = sqrt(0.00952) = 0.0976
```

So tau >= 0.5 is ~5.1 sigma above the null, extremely unlikely by chance.
Even tau >= 0.3 would be ~3.1 sigma (p < 0.001).

## 5. Why LOO Works Under SOLE Orthogonality

### Shapley vs LOO

The Shapley value of expert i is:

```
phi_i = (1/N!) * sum_{pi} [V(S_pi^i union {i}) - V(S_pi^i)]
```

where the sum is over all permutations pi and S_pi^i is the set of experts
preceding i in permutation pi. V(S) is the value function (here, -PPL or 1/PPL).

LOO is the special case where S_pi^i = {1, ..., N} \ {i} for all permutations:

```
LOO_i = V({1, ..., N}) - V({1, ..., N} \ {i})
```

The gap between Shapley and LOO comes from interaction effects. For any pair
(i, j), the interaction index is:

```
I_ij = V({i,j}) - V({i}) - V({j}) + V({})
```

**Under SOLE orthogonality (cos_ij ~ 0):**

When expert weight deltas are orthogonal, their effects on the loss landscape
are approximately independent. The PPL contribution of expert i does not depend
on whether expert j is present. Formally:

```
V(S union {i}) - V(S) ~= V({1,...,N} union {i}) - V({1,...,N})   for all S
```

This means I_ij ~ 0 for all pairs, and LOO_i ~ phi_i. The Shapley value
collapses to the LOO value.

**Quantitative bound:** At SOLE production cosines (cos ~ 0.0002), the
interaction term is bounded by:

```
|I_ij| <= c * |cos_ij| * ||delta_i||_F * ||delta_j||_F
```

for some constant c depending on the loss curvature. With cos ~ 0.0002, the
interaction effects are O(10^-4) of the main effects.

**This is a testable prediction:** If LOO rankings are stable (high tau), it
confirms that orthogonality makes interactions negligible. If LOO rankings are
unstable, it suggests that despite low cosine similarity, there are significant
higher-order interactions (possibly mediated through the nonlinear model).

## 6. Calibration Set Design

### Requirements

The calibration set must satisfy:
1. **Domain-agnostic**: No text should strongly favor any single expert's domain
2. **Diverse**: Cover multiple text types to detect broad quality effects
3. **Sufficient tokens**: Enough for reliable PPL estimation
4. **Balanced between sets A and B**: Same domain mix in both sets

### Token Budget

For reliable PPL estimation, we need enough tokens to reduce variance.
The standard error of log-PPL (cross-entropy) scales as:

```
SE(CE) ~ sigma_token / sqrt(n_tokens)
```

where sigma_token is the per-token log-probability standard deviation
(typically ~2-4 nats for language models).

For detecting Delta_i differences of 0.1% in PPL:

```
Delta(CE) = log(1.001) ~ 0.001 nats
SE(CE) < 0.001 requires n_tokens > (3/0.001)^2 = 9,000,000
```

This is too many tokens for practical evaluation. However, we don't need to
detect 0.1% differences with statistical significance per-expert. We need the
RANKING to be stable, which requires only that the noise be smaller than the
RANGE of deltas (not the smallest delta). If sigma_Delta is 1-5%, we need:

```
SE(CE) < 0.5% * log(PPL) ~ 0.005 * 3 = 0.015 nats
n_tokens > (3/0.015)^2 = 40,000
```

At ~50 tokens per calibration text, T=20 texts gives ~1000 tokens per set.
This is marginal. Recommendation: T=40 texts per set (80 total), ~2000 tokens
per set. Or use longer texts (256-512 tokens each), giving 5K-10K tokens per set
with T=20.

**Decision: T=30 texts per set, truncated to 512 tokens each.** This gives
~15,000 tokens per set, adequate for detecting 0.5% PPL differences at 3 sigma.

## 7. Worked Example (Projected from Pilot-50 Data)

### Setup
- N=50 adapters, diverse domains
- Pilot-50 benchmark: 42.2% mean PPL improvement, range from ~5% to ~80%
- Reference PPL (all 50 composed): PPL_50 ~ 6.0 (typical for Qwen2.5-7B on general text)

### Expected LOO Deltas

Under the hypothesis that each expert contributes proportionally to its quality:

For the BEST expert (e.g., abstract_math, 80% PPL improvement on its domain):
- Removing it should have detectable impact on math-related calibration tokens
- Estimated Delta_best ~ +0.5% to +2% (removing it increases PPL)

For the WORST expert (e.g., a domain with ~5% improvement):
- Removing it should have minimal impact
- Estimated Delta_worst ~ -0.1% to +0.1% (near noise floor)

For a HARMFUL expert (if any -- unlikely given 98% win rate):
- Delta_harmful < 0 (removing it decreases PPL)

Expected sigma_Delta: 0.3% to 1.5% (well above K1 threshold of 0.1%).

### Expected Kendall Tau

If the ranking is driven by genuine expert quality differences:
- tau ~ 0.6-0.8 (moderate to strong agreement)

If the ranking is noise-dominated:
- tau ~ 0.0-0.2 (no agreement)

### Expected Runtime

Using the subtraction optimization:
- Base model load: 20s
- Full merge + eval: 15s
- 50 LOO iterations * (subtract + eval + add back) * 2 sets: 50 * 12s = 600s
- Total: ~635s (~10.5 min)

Conservative estimate (CPU merge approach): ~20-30 min.

## 8. Connection to Prior Results

### Why This Differs from KL Health (KILLED)

KL health measured KL(P_composed || P_base), which captures distributional
DISTANCE from base. This anti-correlates with quality because strong experts
shift distributions more (both on-domain and off-domain).

LOO-PPL measures PPL(M_{-i}, S) - PPL(M_N, S), which captures the QUALITY
IMPACT of removing expert i. A strong expert that shifts distributions far from
base will have POSITIVE Delta_i (removing it increases PPL) IF its contribution
is net positive. A harmful expert that shifts distributions far from base will
have NEGATIVE Delta_i (removing it decreases PPL).

PPL is an absolute quality measure. KL is a relative distance measure. This is
the fundamental distinction.

### Connection to Micro composition_weight_sensitivity

The micro experiment found zero expert specialization (0.00% individual
improvement). LOO deltas were at floating-point epsilon (~1e-11). This was
vacuous because micro experts had nothing to contribute.

At macro scale, experts have 42.2% mean PPL improvement. The LOO deltas should
be orders of magnitude above the noise floor. K1 (sigma_Delta >= 0.1%) should
pass easily unless the calibration text is pathologically insensitive to all
50 domains.

### Connection to Expert Removal Safety

Micro experiments proved: alpha=0.022 (residual+RMSNorm amplification ratio),
complete removal safety bound D = sum_eps * 0.022. For N=50 adapters, removing
one adapter changes the composed output by at most 0.022 * ||delta_i||.

This means LOO is SAFE: the model won't catastrophically degrade when one expert
is removed. The question is whether the PPL change is DETECTABLE and STABLE, not
whether it's safe.

## 9. Assumptions

1. **Weight-space merge is the composition method.** We test naive addition
   (W + sum delta_i), consistent with SOLE default. Not Frechet (killed) or
   GS reorthogonalization.

2. **Calibration texts are domain-agnostic.** Hardcoded texts covering code,
   math, general, creative, QA. No overlap with adapter training data.

3. **4-bit quantization preserves PPL ranking.** Both base and composed models
   use the same NF4 quantization. Absolute PPL values change, but relative
   rankings should be preserved.

4. **Pilot-50 adapters have sufficient quality variance.** The benchmark shows
   42.2% mean with wide range. If all 50 adapters have identical quality, K1
   fails -- but this contradicts the benchmark data.

5. **LOO approximates Shapley under orthogonality.** This is the theoretical
   prediction from Section 5. If wrong, LOO rankings would be unstable (K3
   fails) because the ranking depends on which OTHER experts are present.

6. **PPL on generic text reflects composition quality.** An expert that hurts
   composition should increase PPL on SOME generic text. If an expert is
   harmful only on its own domain and neutral elsewhere, LOO-PPL on generic
   text won't detect it. This is a known limitation: LOO-PPL detects
   COMPOSITION-LEVEL harm, not DOMAIN-LEVEL harm.
