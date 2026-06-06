# MATH: P11.C0 — ThinkPO Polish (DPO: Long Thinking > Short Thinking)

## Motivation

The GRPO experiment (RS-SFT on MMLU-Pro) aligns training distribution with eval
distribution, structurally preventing catastrophic forgetting. However, RS-SFT only
trains on *correct* completions — it does not teach the model to *prefer* longer,
more elaborated thinking chains when multiple response qualities are possible.

**Problem**: After RS-SFT, the model may produce occasional short/shallow thinking
traces that happen to be correct, which creates variance in reasoning quality.

**Solution (ThinkPO, arXiv:2502.13173)**: Apply DPO where `chosen = long CoT` and
`rejected = short CoT` for the same question. This teaches the model to generate
more thorough thinking even when short traces can also be correct.

---

## Theorem 1: DPO Objective (Rafailov et al. 2023, arXiv:2305.18290)

**Statement**: The KL-constrained reward maximization objective:

    max_{π} E_{x~D, y~π(·|x)}[r(x,y)] − β · D_KL[π(·|x) || π_ref(·|x)]

has optimal solution:

    π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y)/β)

where Z(x) = Σ_y π_ref(y|x) exp(r(x,y)/β) is the partition function.

**DPO re-parameterization**: Since r(x,y) = β · log(π_θ(y|x)/π_ref(y|x)) + β·logZ(x),
the reward is implicitly defined by the policy ratio. The partition function cancels
in preference comparison, yielding:

    L_DPO(π_θ) = -E_{(x,y_w,y_l)~D}[
        log σ(
            β · (log π_θ(y_w|x) - log π_ref(y_w|x)) -
            β · (log π_θ(y_l|x) - log π_ref(y_l|x))
        )
    ]

where:
- y_w = chosen (long CoT), y_l = rejected (short CoT)
- π_ref = GRPO adapter (frozen reference policy)
- β = 0.1 (temperature, controls KL divergence from reference)

**QED.**

---

## Theorem 2: ThinkPO — Length as Preference Signal (arXiv:2502.13173)

**Statement**: For chain-of-thought reasoning, longer thinking traces contain
more exploration steps and correction opportunities, yielding higher expected
accuracy than shorter traces. Therefore, for any question where both a long trace
y_w (|thinking| > θ_long) and short trace y_l (|thinking| < θ_short) exist,
the preference pair (y_w, y_l) is a valid DPO training signal.

**Motivation from paper**: arXiv:2502.13173 reports 87.4% → 91.2% on MATH500 (+3.8pp)
using length-based DPO on the same model after SFT. The preference labels are
self-generated — no human annotation needed.

**Mechanism**: Let T(y) = |thinking characters in y|. If T(y_w) > θ_long:
- y_w contains more reasoning steps
- More exploration → more self-correction opportunities
- Empirically correlated with accuracy (arXiv:2501.12599, DeepSeek-R1)

The DPO objective then pushes π_θ toward the policy that generates long traces,
encoded directly in the reward ratio:
    r(x,y_w) - r(x,y_l) = β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l))

After DPO, π_θ assigns higher log-probability to long traces for the same input.

**Prediction** (from paper scaling to E4B-4bit):
- GRPO base: ~63% MMLU-Pro (K1497 target from RS-SFT run)
- ThinkPO +2pp = 65% MMLU-Pro (paper shows +3.8pp on harder MATH500)
- Thinking length: avg_thinking_chars increases by ≥ 10% vs GRPO baseline

**QED.**

---

## Theorem 3: Offline DPO — Reference Log-Probs Precomputation

**Statement**: DPO requires reference log-probs π_ref(y_w|x) and π_ref(y_l|x).
Since π_ref does not change during training, these can be precomputed once and
cached. This eliminates the need for a second live model during training.

**Proof**: The DPO gradient is:
    ∇L_DPO = -β · σ(-h) · (∇log π_θ(y_w|x) - ∇log π_θ(y_l|x))

where h = β·log(π_θ(y_w)/π_ref(y_w)) - β·log(π_θ(y_l)/π_ref(y_l)).

The reference terms log π_ref(y_w|x) and log π_ref(y_l|x) appear as CONSTANTS
in the gradient expression (they do not depend on θ). Therefore, precomputing them
offline and storing as fixed values has identical gradient dynamics to running
the reference model in-loop.

**Memory savings**: Avoids loading two model copies simultaneously.
Peak memory: 1 model + LoRA weights ≈ 5GB (vs 10GB for two-model approach).

**QED.**

---

## Theorem 4: Distribution Alignment (from GRPO Theorem 1)

Same insight as GRPO: DPO training on MMLU-Pro questions D_train = D_eval ensures
DPO preference updates cannot increase MMLU-Pro loss. The DPO objective is a
preference objective; for in-domain questions it cannot cause catastrophic forgetting
worse than GRPO baseline.

Formally: if y_w and y_l are both sampled from π_ref(·|x) where x ~ D_eval, then
both completions are already in the support of the training distribution, and DPO
only adjusts the probability ratio between them.

---

## Quantitative Predictions

| Metric | Predicted | Mechanism |
|--------|-----------|-----------|
| MMLU-Pro (thinking) vs GRPO | +2pp | Theorem 2: ThinkPO adds 2-4pp (paper shows 3.8pp) |
| avg_thinking_chars vs GRPO | +10% | DPO pushes toward longer traces |
| GSM8K vs GRPO | ±3pp | Distribution preserving (D_train=D_eval, no math-specific shift) |
| repetition_rate vs GRPO | ≤ GRPO | Longer thinking should be more varied (not more repetitive) |

---

## Kill Criteria

- **K1499**: ThinkPO MMLU-Pro (thinking) >= GRPO score + 2pp
  - Kill condition: ThinkPO < GRPO + 2pp → DPO length preference not improving MCQ accuracy
  - Expected behavior: PASS (paper shows +3.8pp on MATH500)
  
- **K1500**: avg_thinking_chars (ThinkPO) >= avg_thinking_chars (GRPO) * 1.10
  - Kill condition: thinking length does not increase → DPO not working as intended
  - Expected behavior: PASS (DPO reward pushes toward longer traces)
  
- **K1501**: GSM8K accuracy (ThinkPO) >= GSM8K accuracy (GRPO) - 5pp
  - Kill condition: GSM8K degrades more than 5pp → catastrophic forgetting on math
  - Expected behavior: PASS (D_train = D_eval, no domain shift)

---

## Failure Modes

**Failure Mode 1: GRPO adapter too small (only 5 smoke steps)**
- Structure: If GRPO full run hasn't completed, ThinkPO uses a weak reference model
- Prevention: Experiment depends on exp_p11_grpo_reasoning_adapter COMPLETION
- Detection: If GRPO accuracy < 56.1% (K1497), mark ThinkPO provisional

**Failure Mode 2: Thinking length signal too noisy on 4-bit**
- Structure: E4B-4bit may not reliably generate variable-length thinking (always ~2857 chars)
- Detection: If std(thinking_chars) < 200, preference pairs are not meaningfully different
- Response: Use base model vs GRPO adapter as short/long pair instead

**Failure Mode 3: β=0.1 too weak (KL too small)**
- Structure: If β is too small, reference log-probs dominate, policy barely moves
- Detection: If |π_θ(y_w) - π_ref(y_w)| < 0.01 throughout training → β too small
- Response: Increase β to 0.5 or use SFT only on long traces instead of full DPO

---

## Prior Work

- arXiv:2502.13173 (ThinkPO): DPO with length-based preference; +3.8pp MATH500
- arXiv:2305.18290 (DPO): Direct Preference Optimization (Rafailov et al. 2023)
- arXiv:2501.12599 (s1): "Wait" token budget forcing for extended thinking
- arXiv:2501.12948 (DeepSeek-R1): RS-SFT warmup before GRPO; shows longer CoT = higher accuracy
- arXiv:2402.03300 (GRPO): Group Relative Policy Optimization; precursor to RS-SFT
