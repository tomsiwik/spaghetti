# LEARNINGS: LoRA Scale Sweep Generation Quality

## 1. What We Learned

LoRA scale is domain-dependent, not universal. Three domain categories emerge:
- **Learnable-task** (math): +700% at s=20 -- SFT teaches chain-of-thought format that unlocks latent reasoning
- **Structured-output** (code, medical): +17-36% at s=20 -- format helps but less dramatically
- **Knowledge-dependent** (legal, finance): -14% to -32% at s=20, optimal at s=1-4 -- SFT overwrites what little domain knowledge the 2B base has

Code adapter dominance (Finding #208) was a scale artifact. At per-domain optimal scales, domain adapters significantly win 2/5 (medical, math), tie 3/5. The "universal code adapter" was really "code adapter at a scale that only suits code/math."

**Critical rho measurement:** Even at s=20, the Weyl perturbation ratio rho=0.034 (3.4% of base spectral norm). The model is ALWAYS in augmentation regime. Legal/finance degradation cannot be spectral overwrite -- it must operate through output distribution shift amplified by softmax/attention. This disproves the "overwrite regime" hypothesis from the original MATH.md.

## 2. How It Connects to Prior Work

| Finding | Connection |
|---------|-----------|
| #209: Domain adapters degrade at s=20 | **Refined**: only knowledge-dependent domains degrade. 3/5 domains peak at s=20 |
| #208: Code adapter universal best | **Resolved**: scale artifact. Domain adapters win at correct scales |
| #212: Code adapter destroys GSM8K at s=20 | **Consistent**: alignment tax (2405.13432) applies differently per domain type |
| #215: Scale=2 preserves PPL | **Extended**: scale=2 preserves base but is suboptimal for learnable/structured domains |
| LIMA (2305.11206) | **Confirmed**: math SFT is pure format teaching (+700%). Knowledge domains need base knowledge intact |

## 3. What It Means for the Architecture

**Per-domain scale selection must be part of the routing/composition mechanism.** A single lora_scale for all adapters is fundamentally wrong. The routing system needs to:
1. Apply high scale (s=20) for format-learning domains (math, code)
2. Apply low scale (s=1-4) for knowledge-dependent domains (legal, finance)
3. This is equivalent to per-adapter learned coefficients -- exactly what LoRAuter, ACM, and RDLC do automatically

**The rho finding changes the theoretical framing.** We cannot use Weyl's inequality to predict quality transitions because rho << 1 everywhere. The mechanism is not spectral -- it's distributional. Small weight perturbations are amplified nonlinearly through softmax. This connects to 2603.02224 (subspace geometry governs forgetting via principal angles, not norms).

## 4. What Surprised Us

1. **rho(20) = 0.034** -- We expected the "overwrite regime" at high scale. The perturbation is tiny even at s=20. The entire augmentation/overwrite framing was wrong.
2. **Math +700%** -- The strongest specialization of any experiment. 10% base to 80% at s=20. SFT format teaching is enormously powerful for chain-of-thought.
3. **3/5 domains peak at s=20** -- Finding #209 suggested s=20 was universally bad. It's not -- it's bad for exactly the 2 domains where the base model lacks knowledge.
4. **Finance scores bimodal** -- 5 prompts score ~0.3-0.5, 5 score ~0.0. The mean is misleading. Some finance tasks are answerable, others are impossible for the 2B base.

## 5. What We'd Do Differently

1. **Measure rho FIRST, not as a review fix.** The perturbation ratio should be computed before any quality experiments to set expectations.
2. **Use larger n.** n=10 is borderline for statistical significance, especially for binary metrics (math) and bimodal distributions (finance). n=30+ would give reliable confidence intervals.
3. **Include standardized benchmarks alongside behavioral eval.** Running MMLU/GSM8K at each scale point would ground the findings in established metrics.
4. **Test rank-scale interaction.** We only tested rank=16. AdaLoRA (2303.10512) and rsLoRA (2312.03732) show rank and scale interact -- lower rank with higher scale may differ fundamentally.

## 6. NotebookLM Consultation

**Confirming evidence:**
- SOLE architecture already uses energy_rank_99 heuristic for per-domain rank selection -- same principle as per-domain scale
- Logit-scale mismatch confirmed: continuous LoRA weights at different magnitudes cause catastrophic failure in composition
- DR-LoRA, LoRA-LEGO, ACM all solve this problem automatically (our sweep is manual calibration)

**Contradicting evidence:**
- Single high-magnitude adapter is "roughly neutral on MMLU" -- catastrophic degradation only happens during multi-adapter composition, not single adapter
- Ternary base models may inherently limit nonlinear logit explosions (sign/zero masking)
- 4-bit models tolerate higher adapter influence than 2-bit (bit-width dependent)

**Alternatives to manual scale sweeping:**
- ACM (2410.02906): activation-based mutual information for automatic per-layer coefficients
- RDLC: hypernetwork generates continuous, token-dependent coefficients
- LoRAuter: semantic similarity-based automatic fusion weights
- LoraHub: black-box optimization of adapter weights on validation set

**New references added:**
- #308: rsLoRA (2312.03732) -- rank-stabilized scaling factor
- #309: Subspace Geometry (2603.02224) -- forgetting governed by gradient subspace angles, not norms

## 7. Recommended Follow-ups

1. **exp_generation_quality_test (P0)** -- THE existential test: does routed composition produce better text than base alone? Use per-domain optimal scales from this experiment.
2. **exp_task_accuracy_real_benchmarks (P0)** -- MMLU/GSM8K/HumanEval with composition at per-domain scales. Ground behavioral findings in standardized metrics.
3. **exp_automatic_scale_selection** -- Implement ACM or similar to auto-determine per-adapter scale. Manual sweeping doesn't scale to N=25 adapters.
4. **Increase generation n to 30+** for statistically robust confidence intervals on behavioral metrics.
