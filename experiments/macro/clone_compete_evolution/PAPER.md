# Clone-and-Compete Evolution: Research Digest

## Hypothesis

Cloning an expert, fine-tuning the clone with corrections (50 steps, ~60s),
and running a within-domain PPL tournament reliably identifies the better expert,
with >70% clone win rate, <50K queries to convergence, and <2% domain regression.

**Falsifiable:** If corrections don't help (win rate <= 70%), the tournament is
too slow (>50K queries), or competition causes domain regression (>2%), the
clone-and-compete mechanism is killed.

## What This Experiment Is

The core mechanism test for SOLE Phase 3 (Evolve). Without clone-and-compete,
the Living Composable Model cannot improve itself -- it can only add new experts
but never fix existing ones.

**Protocol:**
1. Select 5 diverse domain experts from pilot 50 (python, bash, math, medical, sql)
2. For each expert: identify its 50 highest-loss training examples (its weak points)
3. Clone the expert, fine-tune clone on these corrections (50 steps, lr=1e-4)
4. Score both original and clone on general domain queries (regression check)
   and correction queries (improvement check) using within-domain PPL
5. Determine winner, measure convergence speed

**Why within-domain PPL:** Answer-only PPL was killed at macro for cross-domain
comparison (r=-0.63). But clone-and-compete only ever compares two adapters on
the SAME domain. Different domains have different PPL baselines (medical text
is lower entropy than code), which explains the cross-domain failure. Within a
single domain, PPL should be a valid comparison signal because both adapters
see identical prompts and the same entropy baseline. This experiment implicitly
tests the exp_relative_ppl_within_domain hypothesis.

## Key References

- **Sakana AI, "Evolutionary Optimization of Model Merging Recipes" (2024):**
  Population-based evolutionary search over merging recipes. SOLE is simpler:
  binary tournament with directed corrections (not random mutation).

- **SPIN, "Self-Play Fine-Tuning" (2024):** Model improves by competing against
  its own previous version. Similar principle (improvement through competition)
  but SPIN uses self-play discrimination, SOLE uses correction + PPL comparison.

- **EvoMoE (2025):** Expert evolution via progressive initialization from a
  single expert. Addresses expert uniformity in MoE. SOLE evolves through
  post-training correction, not initialization.

- **PASs-MoE (2025):** Identifies "misaligned co-drift" where routers and
  experts drift apart during joint training. SOLE avoids this entirely: hash
  ring routing is non-differentiable (no router drift) and experts are frozen
  during competition (no expert drift on non-competing experts).

- **CompeteSMoE (2025):** Competition-based MoE training with statistical
  guarantees. Most directly relevant, but operates during training; SOLE
  operates post-training at inference time.

**Novelty confirmed:** No prior work implements clone-and-compete specifically
for LoRA adapters in a post-training, inference-time evolution loop.

## Empirical Results

**TO BE FILLED from results/clone_compete_evolution/clone_compete_results.json**

### Per-Domain Tournament Results

| Domain | Clone Wins? | Correction PPL Improvement | General Regression | Convergence | Effect Size |
|--------|:-----------:|:--------------------------:|:------------------:|:-----------:|:-----------:|
| python | - | - | - | - | - |
| bash | - | - | - | - | - |
| math | - | - | - | - | - |
| medical | - | - | - | - | - |
| sql | - | - | - | - | - |

### Kill Criteria Assessment

| Criterion | Value | Threshold | Verdict |
|-----------|-------|-----------|---------|
| K1: Clone win rate | -% | >70% | - |
| K2: Max convergence queries | - | <50K | - |
| K3: Max domain regression | -% | <2% | - |

### Fine-Tuning Efficiency

| Domain | FT Time (s) | Final Loss | Steps |
|--------|:-----------:|:----------:|:-----:|
| python | - | - | 50 |
| bash | - | - | 50 |
| math | - | - | 50 |
| medical | - | - | 50 |
| sql | - | - | 50 |

## Limitations

1. **5 domains is low statistical power.** A 4/5 win rate (80%) has p=0.19
   under the null hypothesis (corrections don't help). The result is directional,
   not definitive. A full validation would test 20+ domains.

2. **Training data as corrections is circular.** We use the expert's own
   high-loss training examples as ground truth corrections. This is a cleaner
   signal than teacher-generated corrections but does not test the actual
   automated correction pipeline (70B teacher judging 7B output).

3. **No real serving traffic.** The tournament is simulated with static data.
   Real clone-and-compete would operate on live hash ring traffic with shadow
   scoring. The experiment tests the mechanism, not the integration.

4. **PPL proxy for quality is untested within-domain at macro.** Cross-domain
   PPL correlation failed (r=-0.63). Within-domain is hypothesized to work
   but unvalidated. If within-domain PPL also fails, the scoring mechanism
   is fundamentally broken and task accuracy must replace it.

5. **Single seed.** No variance estimates across random seeds. The fine-tuning
   warm-start should reduce sensitivity to seed, but this is not verified.

## What Would Kill This

**At this scale (macro, 5 domains):**
- Clone win rate <= 70% (3 or fewer domains): corrections don't reliably help
- Any domain regression > 2%: fine-tuning is destructive
- PPL inversion: lower PPL on corrections but clearly worse actual output

**At production scale:**
- Tournament requires >10K real queries to resolve (too slow for evolution)
- Clone-and-compete does not produce monotonic quality improvement over 10 cycles
  (exp_evolution_convergence, downstream)
- Correction pipeline cost exceeds benefit (corrections cost more than retraining)
