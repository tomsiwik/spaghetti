# Concat+Calibrate N=5 Calibration Budget Sweep: Research Digest

## Hypothesis

The N=5 concat+calibrate failure (+5.07% vs joint at 100 calibration steps)
is caused by router underfitting rather than a fundamental mechanism failure.
Sweeping calibration steps {100, 200, 300, 500} will close the gap to <3%
vs joint and beat simple average (+3.33%).

**Falsifiable**: If 500-step calibration still exceeds +3% vs joint, or if no
budget beats simple average, the router-underfitting explanation is killed.

---

## What This Experiment Is

A parameter sweep of the calibration budget for concat+calibrate LoRA
composition at N=5 domains. No new model or mechanism -- this reuses the
exact RoutedDeltaGPT and calibration loop from the lora_merging_bakeoff
experiment, varying only the number of router calibration steps.

The sweep tests whether the N=5 failure identified in the merging bakeoff
(concat+cal at +5.07% vs joint, beaten by simple average at +3.33%) can be
resolved by giving the router more optimization steps.

**Setup**: 5 domains (a-e, f-j, k-o, p-t, u-z), rank-8 LoRA, top_k=2
routing, Adam lr=3e-3, 3 seeds (42, 123, 7).

---

## Lineage in the Arena

```
gpt
 `-- lora_gpt (LoRA adapters on MLP)
      `-- lora_merging_bakeoff (N=2/N=5 method comparison)
           `-- concat_cal_n5_calibration_budget (this experiment)
```

---

## Key References

- **lora_merging_bakeoff** (this project): established concat+cal wins at N=2
  (+1.14%) but loses at N=5 (+5.07%), simple average at +3.33%.
- **lora_procrustes** (this project): RoutedDeltaGPT implementation, confirmed
  LoRA delta orthogonality (cos ~ 0.014) and linear decomposability.

---

## Empirical Results

### Calibration Budget Sweep (N=5, 3-seed aggregate)

| Method | Mean Val Loss | Std | vs Joint | vs Simple Avg |
|--------|-------------|-----|----------|---------------|
| Joint training | 0.5011 | 0.0054 | baseline | - |
| **Simple average** | **0.5154** | **0.0068** | **+2.85%** | **ref** |
| concat_cal_100 | 0.5215 | 0.0054 | +4.06% | +1.18% |
| concat_cal_200 | 0.5220 | 0.0043 | +4.16% | +1.28% |
| concat_cal_300 | 0.5158 | 0.0037 | +2.94% | +0.08% |
| concat_cal_500 | 0.5226 | 0.0082 | +4.30% | +1.41% |

### Per-Seed Detail

| Method | Seed 42 | Seed 123 | Seed 7 |
|--------|---------|----------|--------|
| Joint | 0.4999 | 0.4964 | 0.5070 |
| Simple avg | 0.5222 | 0.5086 | 0.5154 |
| concat_cal_100 | 0.5252 | 0.5239 | 0.5153 |
| concat_cal_200 | 0.5249 | 0.5241 | 0.5170 |
| concat_cal_300 | 0.5195 | 0.5120 | 0.5160 |
| concat_cal_500 | 0.5197 | 0.5319 | 0.5162 |

### Kill Criteria Evaluation

| Criterion | Result | Threshold | Verdict |
|-----------|--------|-----------|---------|
| KC1: 500-step gap vs joint | +4.30% | <3% | **KILL** |
| KC2: any budget beats simple avg | Best: 300 steps (+0.08% worse) | Must beat | **KILL** |

---

## Analysis

### 1. The Non-Monotonic Surprise

The most informative finding is that the relationship between calibration
budget and quality is NOT monotonic:

```
100 steps: +4.06%  --|
200 steps: +4.16%    |-- flat/slightly worse
300 steps: +2.94%  --'-- local minimum (best)
500 steps: +4.30%  ---- WORSE than 300
```

This rules out the "router underfitting" hypothesis. If 100 steps were
insufficient for router convergence, we would expect monotonic improvement
with more steps. Instead, the router appears to hit a sweet spot around
300 steps and then DEGRADES with further training.

### 2. Router Optimization Is Unstable, Not Underfit

The high variance at 500 steps (std=0.0082, vs 0.0037 at 300 steps) and the
per-seed detail tell the story:

- Seed 123, 500 steps: +7.15% (catastrophically bad)
- Seed 7, 500 steps: +1.82% (excellent)
- Seed 42, 500 steps: +3.97% (mediocre)

The router optimization landscape for N=5 experts is unstable. Longer training
amplifies this instability rather than resolving it. The round-robin domain
cycling (optimizing for domain s%5 at step s) creates conflicting gradients
that accumulate over more steps.

### 3. 300 Steps Is Tantalizing But Not Reliable

At 300 steps, concat+cal achieves +2.94% vs joint, which technically passes
KC1 (<3%). But it still loses to simple average by +0.08%. And 300 steps is
a fragile sweet spot:

- Seed 123 at 300 steps: +3.14% (fails KC1)
- Seed 42 at 300 steps: +3.91% (fails KC1)
- Only seed 7 at 300 steps: +1.77% (passes convincingly)

The aggregate passes by a hair, but it is not robust across seeds.

### 4. Simple Average Is Simpler and Better

The merging bakeoff noted simple average at +3.33% vs joint. This sweep
re-measured it at +2.85% (within noise of the original). Simple average:

- Requires zero calibration data
- Requires zero calibration compute
- Has zero hyperparameters
- Achieves better or equal quality at N=5

The only budget where concat+cal approaches simple average is 300 steps,
and even there it loses by 0.08%.

### 5. N=2 Success Was Degenerate

At N=2 with top_k=2, the router selects ALL experts every time. Routing
degenerates to a learned weighted average. This is strictly more expressive
than uniform averaging (simple average) and explains the N=2 win (+1.14%).

At N=5 with top_k=2, the router must make genuine routing decisions (select
2 of 5), and the optimization difficulty scales combinatorially. The N=2
result does not generalize.

### 6. Updated Comparison Table (N=5)

Combining results from both the merging bakeoff and this sweep:

| Method | vs Joint | Compute | Data Needed |
|--------|----------|---------|-------------|
| Simple average | +2.85% | ~0.1ms | None |
| DARE (p=0.3) | +3.36% | ~0.4ms | None |
| concat_cal_300 | +2.94% | ~2.2s | Mixed-domain |
| concat_cal_100 | +4.06% | ~0.9s | Mixed-domain |
| concat_cal_500 | +4.30% | ~4.1s | Mixed-domain |
| TIES | +20.68% | ~0.3ms | None |

Simple average dominates the Pareto frontier at N=5.

---

## Micro-Scale Limitations

1. **Character-level toy domains** may not reflect real domain diversity.
   With genuinely distinct domains (code vs prose), routing may be easier
   because hidden states are more domain-discriminative.

2. **Rank-8 LoRA limits expert differentiation**. With rank 64+, experts
   may be sufficiently distinct that routing adds value over averaging.

3. **Fixed learning rate across budgets**. A learning rate schedule
   (warmup + decay) might stabilize longer training runs. However, the
   router has only 1,280 parameters -- sophisticated schedules seem
   unlikely to fundamentally change the picture.

4. **top_k=2 is fixed**. Different k values might interact with budget,
   though the sparse_router experiment showed k=2 is optimal at micro scale.

5. **3 seeds provides directional evidence**, not statistical significance.

---

## What Would Kill This

### Already Killed

Both kill criteria are triggered:

- **KC1**: 500-step calibration achieves +4.30% vs joint, not <3%.
- **KC2**: No calibration budget beats simple average. Best (300 steps)
  is +0.08% worse.

### At Macro Scale

The hypothesis could revive at macro scale if:
- Domain-discriminative hidden states make routing easier (contrastive
  router experiment suggested this may be the case at d=256+)
- Higher LoRA rank creates more differentiated experts
- Better optimization (learning rate schedule, gradient accumulation)
  stabilizes router training

---

## Key Takeaways

1. **The N=5 concat+cal failure is NOT caused by router underfitting.**
   More calibration steps do not monotonically improve quality. The
   relationship is non-monotonic with a fragile optimum around 300 steps.

2. **Router optimization is unstable at N=5.** The combination of 5
   competing domain gradients and a non-convex routing landscape creates
   high variance and seed-dependent outcomes.

3. **Simple average is the N>=5 default.** At N=5, simple averaging
   (+2.85% vs joint) beats all concat+cal budgets tested, requires no
   calibration data or compute, and has zero hyperparameters.

4. **The contribution model scales differently by N.** At N=2,
   concat+cal adds value (+1.14% vs joint). At N=5, it subtracts value.
   The crossover suggests concat+cal is beneficial only when top_k equals
   or approximates N (i.e., soft averaging, not genuine routing).

5. **Status: KILLED.** Both kill criteria triggered. Concat+calibrate is
   not the right composition protocol for N>=5 at micro scale. Simple
   averaging is the default.

---

## Artifacts

- `micro/models/concat_cal_n5_calibration_budget/` -- code, MATH.md, PAPER.md
- `micro/models/concat_cal_n5_calibration_budget/test_calibration_budget.py` -- full sweep
- Parent experiment: `micro/models/lora_merging_bakeoff/`
- Total experiment time: ~84 seconds (3 seeds x 4 budgets)
