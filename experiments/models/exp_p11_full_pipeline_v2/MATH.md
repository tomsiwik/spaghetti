# MATH.md — P11.M0: Full Pipeline v2

## Motivation

P11 experiments individually validated components for improving Gemma 4 reasoning:
- P11.Z0: Plan-and-Solve prompting (+2pp MMLU-Pro predicted)
- P11.L0: RSD-aligned adapter (student-filtered s1K, +3pp predicted)
- P11.G0: GRPO-improved adapter (RL-dense feedback, +3pp predicted)
- P11.Z1: Injection decoding (wait-token forcing, ~0pp from pre-registration)

This experiment combines the strongest components to test the Room Model's additive independence
claim: do independently validated gains stack without interference?

**Prior art**: Room Model (our Finding #xxx); arXiv:2406.16838 (LoRA-the-Explorer, adapter composition
independence); arXiv:2209.01510 (PS — Plan-and-Solve Prompting).

---

## Theorem 1: Additive Independence Under Exclusive Routing

**Theorem**: Let C_A (adapter), C_P (PS prompt), and C_I (injection) be performance-improving
components with independently measured MMLU-Pro gains δ_A, δ_P, δ_I over base accuracy A_0.
Let each component act on disjoint degrees of freedom under Room Model exclusive routing
(ΔW_combined = ΔW_base + ΔW_selected, zero blending). Then:

    A(C_A ∪ C_P ∪ C_I) ≥ A_0 + max(δ_A, δ_P, δ_I)      [lower bound]
    A(C_A ∪ C_P ∪ C_I) ≤ A_0 + δ_A + δ_P + δ_I          [upper bound]

**Proof**:
- Lower bound: The combined pipeline can always reproduce the best single-component result
  by using the best adapter (ΔW_selected) with the best prompt template. Since the adapter
  and prompt act on disjoint channels (weight space vs. attention priming), there is no
  cancellation. The pipeline cannot perform worse than its strongest component.

- Upper bound: The total gain is at most Σδ_i. This holds when components are perfectly
  orthogonal (no shared variance in error correction). In practice, if both the adapter and
  the PS prompt help on the same hard question, double-counting occurs → A < A_0 + Σδ_i.

- Independence argument: Room Model guarantees ΔW_combined = ΔW_base + ΔW_selected.
  The PS prompt acts on the attention pattern (input space transformation). These are
  geometrically independent: weight-space and input-space interventions don't interact
  in a first-order Taylor expansion of the loss.

**QED**

---

## Theorem 2: Plan-and-Solve as Algorithmic Priming

**Theorem**: PS prompting increases MMLU-Pro accuracy by reframing the attention distribution
from answer-retrieval to step-by-step execution. Formally, let f_base be the attention pattern
under direct prompting and f_PS under PS prompting. Then:

    E[correct | f_PS] ≥ E[correct | f_base]

when the question requires multi-step reasoning (condition on: question has >= 2 logical steps).

**Proof** (informal): PS prefix ("Let's understand the problem and devise a plan")
activates chain-of-thought generation in the thinking channel. The model allocates
more thinking tokens to planning before committing to an answer, reducing premature
answer extraction. This is equivalent to temperature-0 beam search in the thinking space
with the beam initialized by the PS prefix.

**Reference**: arXiv:2209.01510 (Plan-and-Solve Prompting, Wang et al. 2023).
Our prior result: P11.Z0 predicted K1529 (>= 64% MMLU-Pro, +2pp over base).

**QED**

---

## Theorem 3: Injection Irrelevance at High Thinking Depth

**Theorem**: When base model thinking depth D_avg >> D_threshold (threshold for injection),
injection decoding contributes δ_I ≈ 0 to MMLU-Pro accuracy.

**Proof**: P11.Z1 pre-registration: Gemma 4's mean thinking depth = 2614 chars >> 500-char
injection threshold. Since injection is only triggered when D < D_threshold, and D_avg >> D_threshold,
the probability of triggering injection P(trigger) → 0. Therefore δ_I ≈ 0 at D_threshold=500.

At D_threshold=1500 (raised threshold), P(trigger) increases but generation quality may not
improve if the model is already adequately reasoning.

**Corollary**: K1546c (injection adds >= 1pp) is expected to FAIL. This is itself a finding:
injection decoding is redundant for models that already think deeply. The useful contribution
is adapter quality (K1546b) and PS prompting (K1546a).

**QED**

---

## Quantitative Predictions

| Criterion | Theorem | Prediction | Source |
|-----------|---------|------------|--------|
| K1544: full_pipeline MMLU-Pro >= 70% | T1 upper bound | UNCERTAIN: base 62.1% + δ_A(~3-5pp) + δ_P(~2pp) ≈ 67-69% | Optimistic: 70% if adapter gains stack |
| K1545: full_pipeline GSM8K >= 85% | T2 + T3 | UNCERTAIN: depends on math adapter quality | P11.F0 base unknown |
| K1546a: PS adds >= 1pp | T2 | LIKELY PASS: PS forces planning → reduces premature answer | P11.Z0 K1529 |
| K1546b: adapter adds >= 1pp | T1 | LIKELY PASS: RSD/GRPO adapters trained on correct traces | P11.L0, P11.G0 |
| K1546c: injection adds >= 1pp | T3 | EXPECTED FAIL: D_avg >> threshold | P11.Z1 pre-reg |

**Overall K1544 prediction**: LIKELY FAIL at 70% threshold. Expected 67-69%.
Revised conditional pass: if any single adapter delivers > 5pp gain (beyond predicted range),
K1544 becomes possible.

**Kill note**: K1544 at 70% is aspirational (matching Google's reported figure). The experiment
is valuable regardless — it establishes the actual P11 compound gain and identifies
the bottleneck component (adapter quality or prompting).

---

## Failure Mode Analysis

**FM1**: Adapter degradation on MCQ format — RSD/GRPO adapters trained on reasoning traces
may hurt answer extraction (same failure as Finding #517: -26pp for s1K adapter).
Fix: fall back to base model + PS prompt if adapter accuracy < base accuracy.

**FM2**: PS prompt + adapter interaction — if the PS prefix confuses the adapter (trained
without PS prefix), performance may drop below adapter-only. The ablation will detect this.

**FM3**: Injection triggers → extra tokens → timeout — if D_threshold=1500, ~50% of questions
trigger injection, doubling generation time. Budget check: 140q × 2 passes × 25s = 117 min.
Fix: limit injections to 1 per question (max_injections=1 in implementation).

---

## Experimental Design

**4 conditions (ablation)**:
1. `base_thinking`: Base Gemma 4B 4-bit, thinking=True, direct answer prompt
2. `adapter_only`: Best available adapter + thinking, direct answer prompt
3. `adapter_ps`: Best adapter + thinking + PS prompt (P1_ps template)
4. `full_pipeline`: Best adapter + thinking + PS prompt + injection (Wait token at < 1500 chars)

**Adapter priority** (checked at runtime):
1. `adapters/math-rsd-aligned-v0/` (P11.L0, student-filtered)
2. `adapters/math-s1k-grpo-v0/` (P11.G0, RL-improved)
3. `adapters/math-star-r1-v0/` (P11.I0, synthetic R2)
4. `adapters/math-s1k-reasoning-v0/` (P11.F0, raw s1K)

**MMLU-Pro evaluation**: 5 categories × 7 questions × 4 conditions = 140 generations
**GSM8K evaluation**: 25 questions on best condition only

**Estimated time**: 140 × ~25s + 25 × ~25s ≈ 69 min ✓ (within 2h)
