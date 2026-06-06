# Peer Review: Speculative Expert Selection

## NotebookLM Findings

Skipped -- the experiment is self-killed and the analysis is straightforward enough to review directly.

## Mathematical Soundness

### Autocorrelation measurement (Phase 3): CORRECT

The core metric -- consecutive token pair match rate -- is implemented correctly.
`matches = (experts_np[1:] == experts_np[:-1])` counts exact expert index matches
between adjacent tokens. Verified from results.json: 43,152 / 68,199 = 0.6327.
Per-domain spot checks (medical 1687/1913 = 0.8819, psychology 4508/4729 = 0.9533)
are arithmetically correct.

### Router overhead measurement (Phase 5): SOUND WITH CAVEATS

The router-only timing (0.166ms/token) is measured correctly: warm-up, repeated trials,
per-token average. Two caveats:

1. **The 36ms denominator is hardcoded, not measured.** It comes from
   exp_molora_per_token_routing. This is acceptable for an order-of-magnitude
   argument (the conclusion holds whether the denominator is 30ms or 50ms), but
   should be stated explicitly. MATH.md says 0.21ms (prior experiment) while the
   actual measurement is 0.166ms -- consistent direction, minor discrepancy.

2. **The speculative timing loop (lines 577-603) is misleading.** It still routes
   every token (to verify the prediction), so it measures verification overhead,
   not actual speculative savings. The paper correctly focuses on the router-only
   timing for the ceiling argument, so this is a code clarity issue, not an
   analytical error.

The ceiling argument is mathematically airtight: speedup <= C_router / C_total.
At 0.166ms / 36ms = 0.46%, no hit rate can achieve >0.46% speedup. K2 FAIL is
definitive.

### Transition matrix (Phase 4): MISLABELED

**Bug in line 529 of run_experiment.py:** The output maps `ALL_DOMAINS[i]` to
`self_transition_probs[i]`, where `i` is an expert index (0-23), not a domain index.
Expert 0 is not "medical" -- there is no guaranteed correspondence between domain
ordering and expert indices. The transition matrix rows/columns are expert indices,
but the output dictionary labels them as domain names.

This means the "self_transition_probs" dictionary in results.json (e.g., "medical":
0.8593, "creative_writing": 0.1512) has **incorrect labels**. The values themselves
are real self-transition probabilities of experts 0-23, but attaching domain names
to them is wrong. The sticky_experts_top5 list (expert 11: 0.954, expert 10: 0.903)
is correctly labeled by expert index.

**Impact:** The transition matrix analysis is secondary to the main autocorrelation
measurement (Phase 3), which is correctly per-domain. The mislabeling does not affect
K1, K2, or any kill criterion. It would mislead anyone trying to interpret which
domains have which Markov structure, but the paper does not make claims that depend
on individual transition matrix entries by domain name.

### Cost model in MATH.md: CORRECT

The derivation `Speedup = C_R * p / (C_gen + C_R)` is correct. At p=0.80:
0.21 * 0.80 / 36.21 = 0.46%. The paper correctly notes the bound is
independent of hit rate for practical purposes.

## Novelty Assessment

This experiment is not claiming novelty -- it is a measurement study. The speculative
decoding reference (Leviathan et al.) is cited appropriately. The insight that
router overhead is negligible is genuinely useful for the project: it definitively
closes the "optimize the router" direction.

**Prior art check:** The finding that per-token routing oscillates (13-24 experts per
domain) while per-sequence quality is preserved was partially known from
exp_pointer_routing_no_merge (-0.46% PPL delta). This experiment adds the quantitative
decomposition: 6 domains are genuinely sticky, 14/24 are near-random. That
decomposition is new within the project.

## Experimental Design

### What it tests vs. what it claims: MOSTLY ALIGNED

The experiment claims to measure temporal autocorrelation of expert selection. It does.
The experiment claims to measure the speedup ceiling. It does (via arithmetic, which
is the correct approach -- you do not need to benchmark a <0.5% effect).

### Missing baseline: MODERATE CONCERN

**The 63.3% hit rate lacks a marginal baseline.** The correct null hypothesis for
autocorrelation is not random routing (1/24 = 4.2%) but the marginal distribution
baseline: if you shuffle all token-expert assignments within each domain (destroying
temporal structure but preserving the marginal frequency of each expert), what hit
rate do you get?

For a domain where 50% of tokens route to expert k, the expected match rate from
the marginal alone is sum(p_i^2) >= 0.25. The 6 high-autocorrelation domains likely
have concentrated marginal distributions (one dominant expert), which would inflate
the hit rate even without temporal structure.

Without this baseline, the paper cannot distinguish:
- (A) Genuine temporal autocorrelation (consecutive tokens are more likely to match
  than random draws from the same marginal distribution)
- (B) Non-uniform marginal expert preferences (some experts are just more popular,
  so random consecutive tokens match often by chance)

The run length statistics hint at (A) -- median run length of 1.0 for ALL domains
(including high-autocorrelation ones like psychology with avg run 19.7) suggests a
highly skewed distribution with many length-1 runs and a few very long runs. This
is consistent with temporal clustering, not just marginal concentration. But the
paper should compute and report sum(p_i^2) per domain as a baseline.

**Impact on K1:** If the marginal baseline is, say, 55%, then the 63.3% hit rate
represents only 8.3 percentage points of genuine temporal autocorrelation. K1 would
still technically PASS (63.3% >= 60%), but the scientific interpretation changes
substantially.

### Router quality concern: MINOR

The router is trained from scratch (500 steps, 74.7% accuracy). This is not the
same router from exp_softmax_router_scaling. If the router is poorly calibrated,
it might produce noisy expert assignments that reduce measured autocorrelation.
This would make the measurement conservative (true autocorrelation >= measured),
so it does not threaten K1 PASS.

### Bimodal distribution claim: FAIR

The gap between 82.7% (legal, rank 6) and 61.7% (sociology, rank 7) is 21
percentage points. The upper cluster (6 domains, 82.7-95.3%) and lower spread
(18 domains, 40.2-61.7%) are visually distinct. "Bimodal" slightly overstates it --
the lower group is a wide spread, not a tight cluster -- but the qualitative point
(some domains are highly autocorrelated, most are not) is valid.

## Contradiction with Prior Findings

The paper identifies an important contradiction: exp_pointer_routing_no_merge found
"per-sequence = per-token routing" (PPL equivalence), but this experiment shows the
router uses 13-24 different experts within single-domain text. The resolution --
within-cluster misrouting is quality-benign per exp_softmax_router_scaling -- is
correct and well-argued. This is the most scientifically valuable finding in the
experiment.

## Macro-Scale Risks (advisory)

Not applicable -- this experiment is effectively killed. The finding that router
overhead is negligible (<0.5% of inference) transfers directly to macro scale and
should be treated as settled. No further investigation of speculative routing is
warranted.

The per-domain autocorrelation data could inform cache-friendly serving strategies
(batch tokens by predicted domain to improve adapter weight locality), but that is
a different experiment with a different hypothesis.

## Verdict

**KILL**

The experiment correctly kills itself. Both the reasoning and the evidence are sound:

1. **K2 FAIL is mathematically definitive.** Router overhead is 0.46% of inference.
   No speculative strategy can yield >0.46% speedup. The 10% threshold is unreachable
   by approximately 20x.

2. **K1 PASS is technically correct but scientifically weak.** The 63.3% hit rate
   lacks a marginal baseline comparison. Much of it may be explained by non-uniform
   expert preferences rather than genuine temporal autocorrelation. The pass is
   marginal (63.3% vs 60% threshold) and 14/24 domains individually fail the
   threshold.

3. **The transition matrix has a labeling bug** (domain names mapped to expert indices)
   but this does not affect any kill criterion.

4. **Valuable side findings:**
   - Router overhead is negligible -- closes the "optimize routing" direction
   - Per-token routing oscillates widely within domains but is PPL-benign
   - 6/24 domains have genuinely sticky routing (>80%), correlated with distinctive
     vocabulary

**Recommended status: KILLED.** Record the router overhead finding and the
per-token oscillation finding in FINDINGS.md. Do not pursue speculative expert
selection further.
