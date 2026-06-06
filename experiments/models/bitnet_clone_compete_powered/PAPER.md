# Clone-Compete Powered Tournament: Research Digest

## Hypothesis

Clone-compete evolution (warm-starting from the original adapter) produces
statistically significant per-sample improvement over the original at N=200+,
and the warm-start inheritance provides advantage beyond simply training on
more data.

## What This Experiment Does

This is the powered replication of `exp_bitnet_clone_compete`, which had N=38
samples and p=0.265 (inconclusive). This experiment adds two critical elements:

1. **N=500 tournament samples** (419 decisive) for statistical power
2. **Cold-start control arm** -- a fresh adapter trained from scratch on the
   SAME data as the clone, without inheriting the original's weights

The cold-start control disambiguates "warm-start inheritance helps" from
"more training data helps" (motivated by "The Appeal and Reality of Recycling
LoRAs" 2602.12323, which found random LoRAs merge equally well).

**3-arm design:**
- Arm A: original legal adapter (200 steps on law-stack-exchange)
- Arm B: clone v2 (warm-started from original, 200+200 steps on legalbench)
- Arm C: cold-start control (random init, 400 steps on same legalbench data)

## Key References

- Prior experiment: `micro/models/bitnet_clone_compete/` (N=38, p=0.265)
- "The Appeal and Reality of Recycling LoRAs" (2602.12323): random LoRAs
  merge equally well as carefully selected ones
- Population-Based Training (Jaderberg et al., 2017): clone-perturb-select
- Sakana AI evolutionary merging (2403.13187): evolutionary search over
  merging recipes

## Empirical Results

### Kill Criteria Assessment

| Criterion | Threshold | Result | Verdict |
|-----------|-----------|--------|---------|
| K1: clone win rate | >=55%, p<0.05 | 28.9%, p~0 | **KILLED** (clone significantly LOSES per-sample) |
| K2: tournament p-value | <0.05 | Wilcoxon p=0.001 | PASS (tournament is powered) |
| K3: cold-start != clone | cold-start significantly different | cold wins 71.3% per-sample, same aggregate PPL | **KILLED** (warm-start advantage is illusory) |

### Aggregate PPL (Legal Validation Data, 50 samples)

| Arm | PPL | vs Original | vs Base |
|-----|-----|-------------|---------|
| Base (no adapter) | 20.78 | -- | -- |
| Original (200 steps, law-stack-exchange) | 15.82 | -- | -23.9% |
| Clone v2 (warm-start, 400 steps, legalbench) | 13.04 | -17.6% | -37.2% |
| Cold-start (random init, 400 steps, legalbench) | 13.87 | -12.3% | -33.3% |

On validation data, clone v2 beats cold-start by 0.83 PPL (6.0%).

### Aggregate PPL (Tournament Data, 500 held-out samples)

| Arm | Mean PPL | vs Original |
|-----|----------|-------------|
| Original | 13.86 | -- |
| Clone v2 | 13.24 | -4.5% |
| Cold-start | 13.23 | -4.6% |

On held-out data, clone v2 and cold-start are indistinguishable (diff = 0.01 PPL).

### Per-Sample Tournament Results

**Clone v2 vs Original (N=419 decisive):**
- Clone win rate: 28.9% (121/419) -- **original wins 71.1%**
- Binomial p < 0.0001 (significantly below 50%)
- Wilcoxon p = 0.001 (aggregate difference is real)
- Cohen's d = 0.193 (small effect)
- Mean PPL diff: +0.63 [95% CI: 0.34, 0.91] (clone better on average)
- **Paradox:** clone loses 71% of samples but wins on aggregate PPL

**Cold-start vs Original (N=348 decisive):**
- Cold-start win rate: 47.4% (165/348) -- not significant (p=0.362)
- Wilcoxon p = 0.001 (aggregate difference is real despite equal win rate)
- Cohen's d = 0.267 (small-medium effect)
- Mean PPL diff: +0.63 [95% CI: 0.43, 0.84]

**Clone v2 vs Cold-start (N=373 decisive, THE CRITICAL COMPARISON):**
- Clone v2 win rate: 28.7% (107/373) -- **cold-start wins 71.3%**
- Binomial p < 0.0001 (cold-start significantly better per-sample)
- Wilcoxon p < 0.0001 (cold-start better by Wilcoxon)
- Cohen's d = -0.004 (negligible aggregate difference)
- Mean PPL diff: -0.007 [95% CI: -0.16, 0.15]
- t-test p = 0.934 (no aggregate difference)

### The Win-Rate vs Aggregate Paradox

Both clone and cold-start achieve the same aggregate improvement over original
(~0.63 PPL on tournament data). But their per-sample profiles differ dramatically:

- Cold-start wins many samples by small margins (broad, even improvement)
- Clone wins few samples by large margins (concentrated, heavy-tailed improvement)

The warm-start inheritance from the original creates a SPECIALIZATION BIAS:
the clone improves dramatically on the hardest samples (where the original
struggled most) but slightly degrades on easier samples. The cold-start,
lacking this bias, distributes its improvement more evenly.

### Composition Quality (1/N, 5 adapters)

| Domain | Original comp. | Clone comp. | Cold-start comp. |
|--------|---------------|-------------|-----------------|
| python | 2.74 | 2.74 (-0.10%) | 2.75 (+0.21%) |
| math | 4.42 | 4.41 (-0.10%) | 4.42 (+0.06%) |
| medical | 6.27 | 6.27 (-0.04%) | 6.29 (+0.26%) |
| legal | 19.28 | 19.11 (-0.88%) | 19.25 (-0.13%) |
| creative | 5.87 | 5.86 (-0.21%) | 5.88 (+0.15%) |

Clone shows slightly better composition quality than cold-start (all domains
improve slightly vs original). Cold-start shows marginal regression on 4/5
non-target domains (max 0.26%), well within noise.

### Warm-Start Advantage Decomposition

On validation data:
- Total improvement: 2.78 PPL (original 15.82 -> clone 13.04)
- Data component: 1.95 PPL (70.1%) -- from training on legalbench
- Warm-start component: 0.83 PPL (29.9%) -- from inheriting original weights

On held-out tournament data:
- Total improvement: 0.62 PPL
- Data component: 0.63 PPL (101.6%)
- Warm-start component: -0.01 PPL (-1.6%) -- NEGATIVE

The warm-start advantage exists only on data similar to the original's training
distribution. On novel legal text, it vanishes entirely.

## Interpretation

### What Died

1. **Per-sample clone advantage is illusory.** At N=38 (prior experiment),
   clone appeared to win 62.1% of samples. At N=500, clone loses 71.1%.
   The prior result was a small-sample artifact.

2. **Warm-start inheritance provides no generalizable advantage.** The
   cold-start control matches clone aggregate PPL on held-out data and
   BEATS it per-sample. The evolutionary "inheritance" mechanism adds nothing
   that couldn't be achieved by simply training from scratch.

3. **The win-rate paradox explains the prior result.** Clone v2's aggregate
   PPL improvement concentrates on a minority of very hard samples. At N=38,
   these hard samples were overrepresented, inflating the apparent win rate.

### What Survived

1. **Additional training data helps.** Both clone and cold-start significantly
   improve aggregate PPL over original (Wilcoxon p=0.001 for both). The
   Evolve phase works -- but the mechanism is "more data," not "inheritance."

2. **Composition regression remains negligible.** Max regression 0.26%
   (cold-start on medical). The 1/N scaling regression bound O(epsilon/N)
   holds across all arms.

3. **Clone has better composition profile.** The clone's warm-start, while
   not helping held-out PPL, produces slightly better composition behavior
   (all 5 domains improve vs original, compared to cold-start where 4/5
   non-legal domains marginally regress). This may reflect the warm-start
   keeping the clone closer to the original in parameter space, reducing
   the composition perturbation.

## Verdict: KILLED (K1 + K3)

**K1 KILLED:** Clone win rate 28.9% is far below 55% threshold. The clone
significantly LOSES per-sample despite winning on aggregate.

**K3 KILLED (reinterpreted):** Cold-start does not "match" clone in the
way K3 was written (they have different per-sample profiles), but it
BEATS clone per-sample while matching aggregate PPL. The warm-start
advantage is not just absent -- it produces a worse per-sample profile.

**K2 PASS:** The tournament is adequately powered (Wilcoxon p=0.001).

The evolutionary inheritance mechanism is killed. The Evolve phase should
be redesigned around "train on more data" rather than "clone and compete."

## Limitations

1. **Distribution shift confound.** Tournament data is from law-stack-exchange
   (samples 500+) and lex_glue ecthr_a (European Court of Human Rights).
   The original adapter was trained on law-stack-exchange (first 500), giving
   it distribution advantage on the law-stack-exchange portion of tournament
   data. The clone and cold-start were trained on legalbench (contract NLI
   tasks), a different legal subdomain. The per-sample win rates may partly
   reflect domain mismatch rather than warm-start effects.

2. **Training budget asymmetry.** Clone v2 inherits 200 steps on
   law-stack-exchange plus 400 steps on legalbench = 600 total steps of
   gradient exposure. Cold-start gets only 400 steps on legalbench. The
   fair comparison would give cold-start 600 steps, but on which data?
   The asymmetry is inherent to the evolutionary protocol.

3. **PPL-only evaluation.** Task eval was killed at 2B scale. PPL
   improvement does not guarantee task quality improvement.

4. **Single legal domain.** Results may not generalize to other domains
   where the original adapter is weaker or the fresh data is more similar
   to the original training data.

5. **Single seed.** All results from seed 42 (implicit). Prior experiments
   show CV=0.5% across seeds for this architecture.

6. **Tie definition.** Ties defined as |PPL_diff| < 1e-6 (bfloat16
   precision). Different tie thresholds would change decisive sample counts.

## What Would Kill This (Further)

- If the distribution shift confound is removed (same-distribution tournament
  data) and clone still loses per-sample, the kill is conclusive
- If the training budget is equalized (cold-start gets 600 steps) and clone
  still provides no advantage, the inheritance mechanism is definitively dead

## What This Means for the Architecture

The Evolve phase of the Living Composable Model does NOT need cloning or
tournament selection. Instead:

1. **Just retrain on more data.** When fresh domain data is available,
   train a new adapter from scratch on the combined (original + fresh)
   data. This is simpler, cheaper, and produces better per-sample profiles.

2. **Consider warm-start for composition.** The one surviving signal is that
   the warm-started clone has marginally better composition behavior (closer
   to original in parameter space). If composition regression is a concern,
   warm-starting may help -- but this is a minor effect (0.26% max regression
   either way).

3. **The PPL tournament is useful.** Even though clone-compete is killed,
   the per-sample PPL comparison methodology is validated. It can be used
   to compare retrained adapters or to evaluate composition quality.
