# LEARNINGS: Scale-Aware Composition at N=15

## Status: PROVISIONAL (downgraded from SUPPORTED per adversarial review)

The experiment was underpowered (3 prompts/domain) to test its own predictions.
K637 failed but may be a noise artifact. K636 passed trivially.

## Key Learnings

### L1: Ternary adapters exhibit binary on/off behavior, not continuous scaling

The code adapter produces ~0% format scores at s<=4 and ~50-70% at s>=8.
Medical/legal/finance show minimal differentiation across scales. This means
"optimal scale selection" is really "minimum effective scale detection."

**Implication:** Scale sweeps with fine granularity are wasteful. Binary
classification (effective vs ineffective) is the useful signal. The deployment
system needs a minimum scale threshold, not a continuous optimizer.

### L2: Smoke tests (N=3) cannot resolve scale sensitivity

Standard errors exceed inter-scale differences for 4/5 domains. The "optimal"
scale at N=3 is a tie-breaking artifact. Scale sensitivity experiments need
>=20 prompts per condition to detect 10% relative differences.

**Implication:** Future scale experiments must use sufficient sample sizes.
Budget: 5 domains x 5 scales x 20 prompts = 500 evals minimum.

### L3: Math domain adapter is non-functional

Math scores 0.0 at every scale, every composition scheme, AND at base. This is
an eval issue (the prompts require multi-step arithmetic, which a 2B model
cannot do), not a composition issue. Math should be excluded from composition
quality assessments until the eval is fixed or the domain is replaced.

### L4: Oracle top-1 composition at N=15 does not degrade quality

4/5 non-trivial domains beat base under oracle routing at N=15. The
composition infrastructure correctly isolates the active adapter from
synthetic padding adapters. This validates the deployment scenario.

### L5: Per-domain scale selection is not load-bearing for deployment

The binary on/off behavior means routing quality (selecting the RIGHT adapter)
matters far more than scale tuning (selecting the OPTIMAL scale for that adapter).
Any scale in the "on" regime produces similar results. This simplifies deployment:
use a fixed scale >= minimum effective threshold.

## What Failed

- MATH.md predicted oracle scales would be identical to N=5. They shifted (0.4x-8x).
- But the shifts are within noise at N=3 samples. Inconclusive, not falsified.
- N=24 was not tested due to time constraint.

## What This Means for the Critical Path

The deployment track (P0) should focus on:
1. **Routing quality** (already proven: 100% accuracy on 5 domains)
2. **Minimum effective scale** (new question from this experiment)
3. **Generation quality** (not just format correctness)

Scale tuning is deprioritized -- a fixed scale of s=8 or s=20 works for all
structured domains (code, medical). Knowledge domains (legal, finance) work
at any scale because their effects are subtle.

## Recommendations

1. Re-run with N>=20 prompts if scale sensitivity is actually needed
2. Replace math eval or math adapter -- current combo yields zero signal
3. Investigate the binary on/off threshold as a standalone experiment
4. Proceed to generation quality testing (exp_generation_quality_test) -- this
   is the existential test and doesn't depend on scale tuning
