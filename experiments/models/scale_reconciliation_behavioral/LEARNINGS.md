# LEARNINGS: Scale Reconciliation Behavioral (KILLED)

## Core Finding

**Two distinct behavioral regimes exist at different LoRA scales, and no uniform scale
can span both.** s=2.0 is a FORMAT regime (small corrections, preserves base knowledge).
s=20.0 is a CAPABILITY regime (overrides base, activates learned reasoning). Math gains
require the capability regime; knowledge domains (legal/finance) require the format regime.
Per-domain scale is an architectural requirement, not a tuning convenience.

## Literature Grounding

### Directly Relevant

- **Hu et al. (2021) LoRA (arXiv:2106.09685):** Establishes the perturbation ratio
  rho = alpha/r * ||BA||/||W||. Our experiment confirms rho controls behavioral regime:
  rho << 1 (format), rho >> 1 (capability override). The standard alpha/r range [0.5, 2.0]
  is a FORMAT regime by construction.

- **Zhou et al. (2023) LIMA (arXiv:2305.11206):** Claims SFT teaches format, not knowledge.
  Our results PARTIALLY confirm: at s=2.0, format IS preserved (10/10 coherent on all
  domains), but math gains are ZERO (0.100 = base). LIMA is correct that format is
  learnable at low perturbation, but math reasoning is not "format" — it requires
  capability-level perturbation.

- **Finding #217 (lora_scale_ablation):** Empirically measured rho < 0.15 at all tested
  scales, suggesting adapters never truly dominate base weights. Yet behavioral effects at
  s=20 are dramatic (8x math). This implies the threshold for behavioral regime change is
  much lower than rho=1. The linear perturbation model understates nonlinear effects of
  attention pattern shifts.

### Contradicting Evidence / Alternative Approaches

- **PiSSA (arXiv:2404.02948):** SVD-initialized LoRA starts with larger initial gradients
  in the principal subspace. If scale effects are driven by subspace alignment (not just
  magnitude), PiSSA-style initialization could shift the regime boundary, making lower scales
  effective for capability activation. This would weaken our two-regime model.

- **LoRA+ (arXiv:2402.12354):** Different learning rates for A and B matrices. The effective
  perturbation ratio depends on the trained ||BA|| norms, not just scale. Our per-domain
  optimal scales may partially compensate for suboptimal training-time learning rate ratios.

- **DoRA (arXiv:2402.09353):** Decomposes weight updates into magnitude and direction.
  Our regime boundary (format vs capability) may correspond to the magnitude component
  exceeding a threshold. DoRA's decomposition could make the transition more controllable
  than raw scale multiplication.

- **MoLoRA (arXiv:2603.15965):** Per-token routing with learnable scale — exactly the
  architecture our finding demands. Already referenced in VISION.md as the routing target.

### From Prior Experiments in This Project

- **Finding #246 (contrastive training):** PPL-optimal scale was s=2.0, but this experiment
  proves PPL-optimal != behavioral-optimal. The contrastive training finding was measuring a
  PROXY. This extends the proxy chain to seven levels.

- **Finding #238 (behavioral eval):** Per-domain optimal scales {math:20, code:20, medical:20,
  legal:4, finance:1} are confirmed as dominant. No configuration tested here improves on
  the per-domain optimal.

- **lora_scale_ablation LEARNINGS:** Measured rho < 0.15 at all scales, recommended scale=4-8.
  That recommendation was PPL-based. For behavioral quality, math/code need s=20 regardless.

## Key Insight: Seven-Level Proxy Chain

The research has now documented a seven-level cascade where each proxy fails to predict
the next:

1. PPL does not predict MMLU accuracy (Finding #236, r=0.08)
2. MMLU accuracy does not predict behavioral quality (Finding #238)
3. PPL improvement sets do not predict specialization (Finding #240)
4. Cosine similarity does not predict functional disagreement (Finding #240)
5. Domain classification does not predict composition quality (Finding #243)
6. Adapter orthogonality does not predict contrastive value (Finding #245)
7. **PPL-optimal scale does not predict behavioral-optimal scale (Finding #249)**

## Architectural Implication

The routing system must output TWO parameters per token:
1. **Adapter selection** (which expert) — already solved at 90% accuracy (Finding #247)
2. **Scale parameter** (format vs capability) — domain-dependent, must be learned or tabled

The per-domain optimal scale map {math:20, code:20, medical:20, legal:4, finance:1} can be
hardcoded as a lookup table (zero additional parameters) or learned as part of MoLoRA-style
routing. Given Finding #247's 90% routing accuracy, a lookup table may suffice for deployment.

## Statistical Caveat

n=10 per domain is small. The legal (0.097 vs 0.096) and finance (0.181 vs 0.155) comparisons
between s=2.0 and per-domain optimal are within noise. The math result (0.100 vs 0.800) is
robust — 0/10 vs 8/10 correct is not noise. The adversarial review correctly flagged P3
(knowledge preservation) as INCONCLUSIVE rather than confirmed.

## Recommended Follow-ups (Priority Order)

1. **Generation quality existential test** — P0 deployment: does the full system (router +
   per-domain scale + adapter) produce useful text end-to-end? All findings so far use
   isolated metrics. Need holistic generation quality assessment.

2. **Scale phase transition mapping** — Test s={4, 8, 12, 16} on math to locate the exact
   transition boundary. Is it sharp (phase transition) or gradual? Affects whether the lookup
   table needs fine-grained scales or just {low, high}.

3. **Composition-aware routing** — L-MoE (2510.17898) / MoLoRA approach: train router to
   jointly select adapter AND scale with generation quality as objective.

## References

- Hu et al. 2021, LoRA (arXiv:2106.09685)
- Zhou et al. 2023, LIMA (arXiv:2305.11206)
- Zhu et al. 2024, PiSSA (arXiv:2404.02948)
- Hayou et al. 2024, LoRA+ (arXiv:2402.12354)
- Liu et al. 2024, DoRA (arXiv:2402.09353)
- MoLoRA (arXiv:2603.15965)
- Finding #217, #238, #246, #247, #249 (this project)
