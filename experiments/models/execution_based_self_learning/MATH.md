# Execution-Based Self-Learning: Mathematical Foundations

## Notation

| Symbol | Shape / Domain | Definition |
|--------|---------------|------------|
| N | scalar, positive int | Number of problems in evaluation set |
| K | scalar, positive int | Solutions generated per problem per cycle |
| t | scalar, non-neg int | Self-learning cycle index (0-indexed) |
| s_t | scalar in (0, 1) | Skill level at cycle t (abstract pass probability) |
| d_t | scalar in (0, 1] | Diversity at cycle t (1.0 = maximum, 0.05 = floor) |
| delta_i | scalar in (0, 1) | Difficulty of problem i |
| p_i(t) | scalar in (0, 1) | Probability expert passes problem i at cycle t |
| alpha_SFT | scalar, positive | SFT learning rate |
| alpha_DPO | scalar, positive | DPO learning rate |
| beta_neg | scalar in [0, 1] | DPO negative example weight |
| gamma | scalar, non-neg | Diversity decay base rate per cycle |
| a | scalar, non-neg | Diversity decay acceleration rate |
| f | scalar in [0, 1] | Fresh data fraction mixed per cycle |

## Pass Probability Model

Each problem i has difficulty delta_i drawn from a truncated normal:

    delta_i ~ TruncNormal(mu_d, sigma_d, [0.01, 0.99])

The probability of generating a correct solution depends on skill, difficulty,
and diversity:

    p_i(t) = sigmoid(logit(s_eff(t)) - logit(delta_i))

where the effective skill accounts for diversity:

    s_eff(t) = s_t * (0.5 + 0.5 * d_t)

**Interpretation:** At full diversity (d=1), effective skill equals nominal skill.
At collapse threshold (d=0.3), effective skill is 65% of nominal. The
0.5 + 0.5*d formula ensures that even a collapsed model retains some capability
(it can still generate solutions, just fewer unique ones).

### Pass@1 and Pass@k

Expected pass@1:

    pass@1(t) = (1/N) * sum_i p_i(t)

Unbiased pass@k (Chen et al., 2021):

    pass@k = E[1 - C(K-c, k) / C(K, k)]

where c is the number of correct solutions out of K attempts.

## Skill Update Rules

### SFT Update

Signal comes from passing solutions only. Stronger signal from harder problems:

    sigma_SFT(t) = mean(delta_i : solutions_i has at least 1 pass) * d_t

Skill update with diminishing returns:

    s_{t+1} = s_t + alpha_SFT * sigma_SFT(t) * (1 - s_t)

The (1 - s_t) term ensures skill asymptotes toward 1.0, matching the
diminishing returns observed in ReST-EM and SPIN.

### DPO Update

Contrastive signal from problems with both passing AND failing solutions:

    sigma_DPO(t) = mean(delta_i : problem i has both pass and fail) * (1 + beta_neg) * d_t

    s_{t+1} = s_t + alpha_DPO * sigma_DPO(t) * (1 - s_t)

**Why DPO is stronger:** The contrastive signal (1 + beta_neg) > 1 amplifies
learning because the model learns both what to do and what to avoid. The
fraction of problems with contrastive pairs depends on skill level:

- Low skill (s ~ 0.3): most problems have both pass and fail -> strong DPO signal
- High skill (s ~ 0.9): most problems all-pass -> DPO degrades to SFT

This creates a natural curriculum: DPO is strongest early (when most needed)
and weakens as the model improves (when it matters less).

**Important note on DPO advantage magnitude:** The total DPO amplification over
SFT is approximately alpha_DPO/alpha_SFT * (1 + beta_neg) = 0.22/0.15 * 1.3 =
1.91x per cycle. The observed ~2.1x advantage in simulation is consistent with
(and largely explained by) this parametric amplification. The DPO advantage
magnitude is an input assumption calibrated from SPIN, not a prediction of the
simulation. The simulation confirms the dynamics are consistent with this
assumption, but does not independently validate the magnitude.

### Fixed-Point Analysis

At the fixed point s* where s_{t+1} = s_t:

    alpha * sigma(t) * (1 - s*) = 0

This holds when s* = 1 (perfect) or sigma(t) = 0 (no signal). In practice,
sigma(t) -> 0 as diversity drops, creating an effective fixed point at:

    s* = s_0 + alpha * integral(sigma(tau) * (1 - s_tau), tau=0..infinity)

## Diversity Dynamics

### Constant Decay Model

    d_{t+1} = d_t * (1 - gamma)

After T cycles: d_T = d_0 * (1 - gamma)^T

Time to collapse (d_T = d_collapse):

    T_collapse = log(d_collapse / d_0) / log(1 - gamma)

With gamma_SFT = 0.03, d_collapse = 0.3: T_collapse = 39.5 cycles (SFT).
With gamma_DPO = 0.015, d_collapse = 0.3: T_collapse = 79.6 cycles (DPO).

Both the main simulation (Experiment 1) and the stress test use this same
clean geometric decay as the base model. The stress test adds acceleration
on top (see below). Neither script uses data-dependent decay weighting.

### Accelerating Decay Model (Shumailov et al.)

The key insight from model collapse literature: training on own outputs
narrows the distribution, and a narrower distribution produces more similar
outputs, which narrows it further. This positive feedback loop creates
accelerating decay:

    gamma(t) = gamma_0 * (1 + a)^t

    d_{t+1} = d_t * (1 - gamma(t))

After T cycles:

    d_T = d_0 * prod_{t=0}^{T-1} (1 - gamma_0 * (1 + a)^t)

Taking logs:

    log(d_T / d_0) = sum_{t=0}^{T-1} log(1 - gamma_0 * (1 + a)^t)
                   ~ -gamma_0 * sum_{t=0}^{T-1} (1 + a)^t     (for small gamma)
                   = -gamma_0 * ((1+a)^T - 1) / a

Time to collapse:

    T_collapse ~ log(1 - a * log(d_collapse/d_0) / gamma_0) / log(1 + a)

**Numerical examples:**

| gamma_0 | a    | T_collapse (SFT) | T_collapse (DPO) |
|---------|------|:-----------:|:-----------:|
| 0.03    | 0.00 | 39 cycles   | > 40 cycles |
| 0.03    | 0.05 | 28 cycles   | > 30 cycles |
| 0.03    | 0.08 | 19 cycles   | 26 cycles   |
| 0.03    | 0.12 | 14 cycles   | 21 cycles   |
| 0.03    | 0.20 | 10 cycles   | 16 cycles   |

### Fresh Data Recovery

With fresh data fraction f:

    d_{t+1} = d_t * (1 - gamma(t)) + f * r

where r is the recovery rate per unit fresh data. The steady-state diversity
(setting d_{t+1} = d_t) for constant gamma:

    d_steady = f * r / gamma

For collapse prevention, we need d_steady > d_collapse:

    f > gamma * d_collapse / r

With gamma=0.03, d_collapse=0.3, r=0.02:

    f > 0.45  (45% fresh data needed)

**Self-consistency note:** The simulation uses the same r=0.02 recovery rate,
so the simulation result (50% fresh data prevents collapse) is a self-consistency
check of the model, not an independent confirmation. The math and simulation
implement the same equations; agreement is expected and does not validate
the choice of r=0.02 against empirical data.

## Convergence Rate Analysis

The skill improvement per cycle:

    Delta_s(t) = alpha * sigma(t) * (1 - s_t)

This is a product of three decreasing terms:
1. sigma(t) decreases as diversity drops
2. (1 - s_t) decreases as skill increases
3. Constant alpha

The maximum improvement occurs at the cycle where:

    d/dt[sigma(t) * (1 - s_t)] = 0

For the constant-decay model, this is approximately:

    t_peak ~ log(alpha * sigma_0) / gamma

In practice:
- SFT peak improvement: cycle 4-6, then diminishing returns
- DPO peak improvement: cycle 3-5, faster initial learning

## Worked Example (Micro Scale)

Parameters: N=200, K=10, s_0=0.30, gamma_SFT=0.03, gamma_DPO=0.015, a=0.08

**Cycle 0:**
- s_eff = 0.30 * (0.5 + 0.5*1.0) = 0.30
- Mean pass@1 = sigmoid(logit(0.30) - logit(0.50)) = sigmoid(-0.847) = 0.300
- Generate 10 solutions per problem: ~70% of problems have >= 1 pass
- SFT signal: mean(difficulty | pass) * 1.0 ~ 0.45
- SFT update: 0.30 + 0.15 * 0.45 * 0.70 = 0.347

**Cycle 5 (SFT):**
- diversity ~ 1.0 * (1-0.03*1.08^t) product ~ 0.87
- skill ~ 0.45
- pass@1 ~ 0.45

**Cycle 5 (DPO):**
- diversity ~ 1.0 * (1-0.015*1.08^t) product ~ 0.93
- skill ~ 0.57
- pass@1 ~ 0.57

**DPO advantage at cycle 5: ~12pp**, growing to ~20pp at peak.

## Calibration Sources

| Parameter | Value | Source |
|-----------|-------|--------|
| s_0 = 0.30 | Initial pass@1 | Small code models on MBPP (CodeGen-2B ~30%) |
| alpha_SFT = 0.15 | SFT learning rate | ReST-EM: ~5-15pp over 2-3 cycles |
| alpha_DPO = 0.22 | DPO learning rate | SPIN: ~1.5x SFT improvement rate |
| gamma_SFT = 0.03 | SFT diversity decay | Model collapse at ~20 cycles matches Shumailov. Used in both main sim and stress test. |
| gamma_DPO = 0.015 | DPO diversity decay | DPO preserves ~2x more diversity (contrastive). Used in both main sim and stress test. |
| a = 0.08 | Decay acceleration | Fit to Shumailov curve: collapse at ~20 cycles |
| d_collapse = 0.3 | Collapse threshold | Qualitative: <30% of original diversity = degenerate |

## Computational Cost

Per self-learning cycle:
- Generation: N * K forward passes (N=200, K=10 -> 2000 forward passes)
- Execution: N * K test suite runs (~2000 subprocess calls, ~10s on MBPP)
- Training (SFT): ~100-300 steps on filtered data (~2 min on A5000)
- Training (DPO): ~200-500 steps on preference pairs (~4 min on A5000)
- Total per cycle: ~5-10 minutes
- 15 cycles: ~1.5-2.5 hours

This fits comfortably within the SOLE expert training budget (~15 min base
training + 15 cycles * 5 min = ~90 min total).

## Assumptions

1. **Independence:** Solutions are generated independently. In practice, LLM
   sampling with temperature creates correlated samples. This underestimates
   diversity loss.

2. **Binary oracle:** Test suites provide perfect pass/fail signals. In
   practice, test coverage < 100% means some "passing" solutions are wrong
   (degeneracy). See correction_signal_quality for coverage analysis.

3. **Instant learning:** Skill update is instantaneous. Real LoRA fine-tuning
   may not fully absorb the signal in 100-300 steps.

4. **Monotonic difficulty:** Problem difficulties are fixed across cycles.
   In practice, the expert may "overfit" to certain difficulty patterns.

5. **Scalar skill:** We model skill as a single scalar. Real code ability is
   multi-dimensional (syntax, algorithms, design patterns, etc.). Collapse
   may affect dimensions differently.
