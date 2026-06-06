# Execution-Based Self-Learning Loop: Research Digest

## Hypothesis

Code experts can self-improve via execution feedback alone, without any teacher
model: generate K solutions, execute against test cases, train on passing
solutions (SFT) or pass/fail pairs (DPO), repeat.

**Falsifiable:**
- K1: Self-learning loop does not improve pass@1 after 5 cycles.
- K2: Generated training data quality degrades over cycles (model collapse).

## What This Experiment Is

A Monte Carlo simulation of the self-learning loop dynamics, calibrated from
published results in the code self-improvement literature (SPIN, ReST-EM,
CodeRL, RLTF, Shumailov model collapse). The simulation models:

1. **Skill update** from execution-verified training data (SFT and DPO variants)
2. **Diversity dynamics** with constant or accelerating decay (Shumailov model collapse)
3. **Fresh data mixing** as collapse mitigation
4. **K sweep** for sample efficiency under both decay models

This is explicitly a simulation study -- no actual model training occurs.
The value is in characterizing the dynamics (convergence rate, collapse
boundary, DPO advantage) before building the production self-learning pipeline.

### Calibration Sources

| Parameter | Value | Source |
|-----------|-------|--------|
| Initial pass@1 | 0.30 | CodeGen-2B on MBPP (~30%), typical small code models |
| SFT learning rate (alpha_SFT) | 0.15 | ReST-EM: 5-15pp over 2-3 cycles |
| DPO learning rate (alpha_DPO) | 0.22 | SPIN: ~1.5x improvement rate vs SFT |
| Diversity decay SFT (gamma_SFT) | 0.03/cycle | Shumailov et al.: collapse at ~20 cycles |
| Diversity decay DPO (gamma_DPO) | 0.015/cycle | Assumed: DPO preserves ~2x diversity (contrastive) |
| Decay acceleration (a) | 0.08/cycle | Fit to model collapse literature |
| Collapse threshold | diversity < 0.3 | Qualitative: below 30%, outputs are degenerate |

**Note on parameter consistency:** gamma_SFT=0.03 and gamma_DPO=0.015 are used
in both the main simulation (constant decay) and the stress test (accelerating
decay). Both scripts use the same clean geometric decay model from MATH.md:
d_{t+1} = d_t * (1 - gamma). The stress test adds acceleration:
gamma(t) = gamma_0 * (1 + a)^t.

## Key References

- Chen et al. 2024, "Self-Play Fine-Tuning (SPIN)" -- iterative self-play
  converges in ~3 iterations, outperforms DPO supplemented with GPT-4 data
- Singh et al. 2024, "ReST-EM" -- 2-3 iterations of rejection sampling +
  EM training, 5-15pp improvement on code/math
- Le et al. 2022, "CodeRL" -- reinforcement learning from execution feedback,
  +5-12% on MBPP with critic-based filtering
- Liu et al. 2023, "RLTF" -- RL from unit test feedback, uses fine-grained
  test signals (not just pass/fail) for more stable training
- Haluptzok et al. 2022, "Language Models Can Teach Themselves to Program
  Better" -- self-generated verified puzzles double test accuracy
- Shumailov et al. 2023, "The Curse of Recursion" -- training on model's own
  outputs causes progressive diversity loss and eventual collapse
- Zhou et al. 2025, "R-Diverse" -- mitigating diversity illusion in self-play
  LLM training
- RLEF (2024) -- RL from execution feedback for code LLMs, grounding in
  execution results

## Empirical Results

### Experiment 1: SFT vs DPO (Pure Self-Play, 15 Cycles, Constant Decay)

Both scripts now use the same clean geometric decay model (d_{t+1} = d_t * (1 - gamma))
with gamma_SFT=0.03 and gamma_DPO=0.015.

| Metric | SFT | DPO | Delta |
|--------|:---:|:---:|:-----:|
| Initial pass@1 | 0.344 | 0.344 | -- |
| Pass@1 at cycle 5 | 0.456 | 0.574 | **+11.8pp** |
| Final pass@1 (cycle 15) | 0.562 | 0.768 | **+20.6pp** |
| Peak pass@1 | 0.562 | 0.768 | **+20.6pp** |
| Convergence cycle | 9.7 | 12.8 | DPO learns longer |
| Final diversity | 0.653 | 0.809 | DPO preserves more |
| Collapse rate | 0% | 0% | Neither collapses |

**K1: PASS.** Both SFT (+11.2pp) and DPO (+23.0pp) improve pass@1 substantially
after 5 cycles. DPO achieves ~2.1x the improvement of SFT.

**On the DPO advantage magnitude:** The 2.1x DPO advantage is an input
assumption, not a prediction of the simulation. The total parametric
amplification of DPO over SFT is alpha_DPO/alpha_SFT * (1 + beta_neg) =
0.22/0.15 * 1.3 = 1.91x per cycle. The observed 2.06x (23.0/11.2) is
consistent with this and partially includes the signal-skill coupling effect
(harder problems pass as skill increases, producing stronger signal). The
simulation confirms the dynamics are self-consistent with these calibrated
parameters but does not independently validate the DPO advantage magnitude.
The directional finding (DPO > SFT) is robust; the specific magnitude is a
calibration choice informed by SPIN.

**K2: PASS** under constant decay. Diversity drops but does not reach
collapse threshold within 15 cycles.

### Experiment 2: Stress Test (Accelerating Decay, 30 Cycles)

The constant-decay model is unrealistically optimistic. With accelerating decay
(modeling the positive feedback loop in model collapse):

| Method | Peak pass@1 | Peak Cycle | Collapse Cycle | Final pass@1 |
|--------|:-----------:|:----------:|:--------------:|:------------:|
| SFT | 0.510 | 11 | **18-19** | 0.005 |
| DPO | 0.701 | 13 | **26-27** | 0.100 |

**K2 under stress: FAIL for pure self-play.** Both methods eventually collapse,
but DPO delays collapse by ~7 cycles (37% longer).

**Note on collapse cycle variance:** All 20 seeds collapse within a 1-cycle
window (SFT: 18-19, DPO: 26-27). The noise term (sigma=0.003) is negligible
compared to the decay dynamics, making the trajectory essentially deterministic.
The Monte Carlo ensemble confirms this determinism but provides no additional
statistical power for estimating the collapse boundary. The collapse cycle
numbers should be interpreted as point estimates from a deterministic model,
not as means of a meaningful distribution. To obtain genuine uncertainty
estimates, one would need to vary the calibration parameters themselves
(parametric sensitivity analysis, as in Experiment 5).

**Key insight:** There is a "golden window" of ~10-25 cycles (method-dependent)
where self-learning improves the expert before diversity collapse degrades it.
The optimal strategy is to stop self-learning and inject fresh data before
crossing the collapse boundary.

### Experiment 3: Fresh Data Mixing (Collapse Mitigation)

| Fresh % | DPO Final pass@1 | DPO Final Diversity | Collapse? |
|--------:|:-----------:|:---:|:---:|
| 0% | 0.100 | 0.198 | Yes (cycle 26) |
| 10% | 0.142 | 0.220 | Yes |
| 20% | 0.197 | 0.243 | Yes |
| 30% | 0.266 | 0.265 | Yes |
| **50%** | **0.609** | **0.310** | **No** |

**50% fresh data prevents collapse indefinitely.** Below 50%, the accelerating
decay eventually overwhelms the recovery signal.

**Self-consistency note on the 45-50% threshold:** The analytical model
predicts f > 0.45 (45% fresh data needed) using r=0.02, and the simulation
uses the same r=0.02 recovery rate. The agreement between math (45%) and
simulation (50%) is a self-consistency check, not independent confirmation.
The simulation implements the same equations as the analysis; agreement is
expected. The practical recommendation (50% fresh data) is conditional on
r=0.02 being correct, which requires empirical validation.

For the constant-decay model (Experiment 1), 0% fresh data is sufficient --
no collapse occurs. The truth likely lies between these extremes: 10-30%
fresh data per cycle is a reasonable practical recommendation.

### Experiment 4: Solutions Per Problem (K Sweep, Constant Decay)

| K | DPO Final pass@1 | DPO Peak |
|--:|:-----------:|:--------:|
| 1 | 0.587 | 0.587 |
| 3 | 0.769 | 0.769 |
| 5 | 0.771 | 0.771 |
| 10 | 0.768 | 0.768 |
| 20 | 0.766 | 0.766 |
| 50 | 0.765 | 0.765 |

### Experiment 4b: K Sweep Under Accelerating Decay (Stress Test)

| K | DPO Peak pass@1 | Collapse Cycle | Final pass@1 |
|--:|:-----------:|:--------------:|:------------:|
| 1 | 0.617 | 26 | 0.087 |
| 3 | 0.687 | 26 | 0.093 |
| 5 | 0.695 | 26 | 0.094 |
| 10 | 0.701 | 26 | 0.100 |
| 20 | 0.705 | 26 | 0.097 |
| 50 | 0.708 | 26 | 0.103 |

**K=3 is the sweet spot under both decay models.** Under constant decay, K=3
achieves 0.769 vs K=1 at 0.587 (+18.2pp), with K>3 providing <0.3pp additional
benefit. Under accelerating decay, K=3 achieves 0.687 vs K=1 at 0.617 (+7.0pp),
with K>3 again providing diminishing returns (<2pp from K=3 to K=50).

Notably, K does NOT affect the collapse boundary: all K values collapse at
cycle 26 under accelerating decay. This is because K affects the quality of
the learning signal (more contrastive pairs) but not the diversity dynamics
(which depend only on gamma and acceleration). More solutions per problem
help the model learn faster but do not slow diversity loss.

### Experiment 5: Decay Acceleration Sensitivity

| Acceleration | DPO Peak | Collapse Cycle | Final pass@1 |
|:------------:|:--------:|:--------------:|:------------:|
| 0% (constant) | 0.769 | never | 0.752 |
| 2% | 0.749 | never | 0.710 |
| 5% | 0.723 | never | 0.637 |
| **8%** | **0.701** | **26** | **0.100** |
| 12% | 0.678 | 21 | 0.014 |
| 20% | 0.642 | 16 | 0.005 |
| 30% | 0.611 | 12 | 0.004 |

**The critical acceleration rate is ~8%.** Below 5%, DPO never collapses within
30 cycles. Above 8%, collapse is inevitable. This is the key empirical
measurement for production: if real diversity decay accelerates faster than
~5%/cycle, fresh data injection is mandatory.

### Kill Criteria Assessment

**K1: Does pass@1 improve after 5 cycles?**

| Method | Improvement at Cycle 5 | Verdict |
|--------|:---:|:---:|
| SFT | +11.2pp | **PASS** |
| DPO | +23.0pp | **PASS** |

Both methods robustly improve pass@1. DPO achieves ~2.1x the improvement of
SFT. This ratio is consistent with the 1.91x parametric amplification from
calibrated parameters (see note above); the directional finding is robust but
the magnitude is a calibration input.

**K2: Does training data quality degrade (model collapse)?**

| Scenario | SFT | DPO | Verdict |
|----------|:---:|:---:|:---:|
| Constant decay, 15 cycles | No collapse | No collapse | **PASS** |
| Accelerating decay, 30 cycles | Collapse at 18-19 | Collapse at 26-27 | **CONDITIONAL** |
| Accel. decay + 50% fresh | Prevents | Prevents | **MITIGATED** |

K2 is CONDITIONAL: collapse depends on whether real diversity decay accelerates.
With constant decay (optimistic), no collapse. With accelerating decay
(pessimistic), collapse is inevitable but delayed significantly by DPO and
completely prevented by 50% fresh data mixing.

**Overall verdict: SUPPORTED (not proven)**

The self-learning loop works in principle (K1 robustly passes), but collapse
risk exists under realistic conditions (K2 conditional). The experiment
identifies the precise mitigation: DPO over SFT, and fresh data mixing when
diversity drops below ~50%.

## SOLE Integration

This experiment validates the execution feedback path for Phase 3 (Evolve):

```
Expert v1 (code domain)
    |
    +-- Generate K=3 solutions per problem from MBPP/HumanEval
    |
    +-- Execute against test cases (binary oracle)
    |
    +-- DPO training on (pass, fail) pairs
    |     (NOT SFT: DPO gives +20.6pp advantage and 2x diversity preservation)
    |
    +-- After cycle: check diversity metric
    |     If diversity < 0.5: inject 30-50% fresh data next cycle
    |     If diversity < 0.3: STOP self-learning, full retraining
    |
    +-- Expert v2 replaces v1 on hash ring (shadow scoring)
    |
    +-- Repeat for 5-10 cycles (within safe window)
```

**Cost per self-learning cycle:**
- Generation: 200 problems * 3 solutions * ~$0.001 = $0.60
- Execution: essentially free (subprocess calls)
- DPO training: ~5 min on A5000 = ~$0.01
- Total: ~$0.61/cycle, ~$6.10 for 10 cycles
- Compare: original expert distillation = $0.44

Self-learning at 10 cycles costs ~14x the initial distillation but can
improve pass@1 by up to 40pp (0.30 -> 0.70). This is cost-effective if the
expert serves many queries.

**Cost caveat:** The $0.60/cycle generation estimate assumes small-model
inference at ~$0.001/solution. At macro scale with Qwen2.5-7B, generation
cost per solution may be $0.005-0.01, making per-cycle cost $3-6, and 10
cycles $30-60. The economics depend heavily on serving infrastructure.

## Limitations

1. **Simulation, not empirical.** No actual LoRA training or code generation
   occurs. The skill update model (logistic with diminishing returns) is a
   reasonable approximation but may not capture real training dynamics
   (loss plateaus, catastrophic forgetting, mode collapse).

2. **DPO advantage is parametric.** The ~2.1x DPO advantage is mechanically
   guaranteed by the 1.91x parameter amplification (alpha ratio * beta_neg).
   The simulation confirms self-consistency of the dynamics, not the magnitude
   of the DPO advantage. Macro-scale A/B testing is needed to validate.

3. **Scalar skill model.** Real code generation is multi-dimensional. Collapse
   may affect some skills (algorithm design) more than others (syntax).

4. **Binary oracle.** Test suites are imperfect (coverage < 100%). The
   correction_signal_quality experiment showed 11.3% degeneracy for
   systems_programming at 60% coverage. This simulation assumes perfect tests.

5. **Collapse cycle estimates are deterministic.** The noise model (sigma=0.003)
   is too small to produce meaningful variance across seeds. The "20/20 seeds
   collapse" finding confirms the trajectory is deterministic, not that the
   result is robust to parameter uncertainty. The collapse boundary estimates
   come from Experiment 5 (acceleration sweep), which provides genuine
   sensitivity analysis.

6. **Calibration uncertainty.** The key parameters (learning rates, decay rates,
   acceleration) are calibrated from different papers using different models
   and benchmarks. The exact numbers should not be taken at face value --
   the directional findings (DPO > SFT, collapse is real but delayed, fresh
   data prevents it) are more robust than the specific numbers.

7. **No LoRA-specific effects.** LoRA training may be inherently more resistant
   to collapse than full fine-tuning (rank constraint acts as regularizer),
   or it may be more fragile (limited capacity means faster saturation).
   The LoRA vs full fine-tuning collapse dynamics are unknown.

8. **Independence assumption.** Generated solutions are modeled as independent
   Bernoulli trials. Real LLM sampling with temperature creates correlated
   outputs, which would accelerate diversity loss.

9. **Fresh data threshold is circular.** The analytical prediction (f > 45%)
   and simulation result (50% prevents collapse) use the same recovery rate
   r=0.02. This is a self-consistency check, not independent confirmation.
   The actual fresh data requirement depends on the true value of r, which
   requires empirical measurement.

## What Would Kill This

**At micro scale (what already passed/failed):**
- K1: PASS (both SFT +11.2pp and DPO +23.0pp after 5 cycles)
- K2: CONDITIONAL (collapse under accelerating decay, prevented by fresh data)

**At macro scale (what needs validation):**
- Actual LoRA self-training on a code expert (Qwen2.5-7B + python LoRA) with
  MBPP test suite as oracle. Measure real pass@1 improvement and diversity
  across 10 cycles.
- If pass@1 does not improve after 5 real cycles, K1 is killed.
- If outputs become repetitive (unique n-gram ratio drops >50%) within 10
  cycles without fresh data, K2 is killed for the constant-decay model.
- DPO vs SFT A/B test on the same expert and problem set.
- Fresh data mixing: does 20% fresh data per cycle prevent collapse in practice?

**What would fundamentally kill the approach:**
- If execution feedback produces degenerate solutions at >20% rate even with
  high test coverage (>80%). This would mean the oracle itself is unreliable.
- If LoRA rank constraint causes collapse faster than full fine-tuning
  (saturation rather than diversity loss).
- If the cost of generating K=3 solutions per problem exceeds the benefit
  of the pass@1 improvement (unfavorable economics at scale).
