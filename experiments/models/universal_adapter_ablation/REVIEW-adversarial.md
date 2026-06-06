# Peer Review: Universal Adapter Ablation

## Experiment Type
Guided exploration (Type 2).

The proven framework is SFT adapter composition with Grassmannian A-matrices.
The unknown is whether routing between adapters provides measurable behavioral
benefit over the single best adapter. This is a well-framed ablation that asks
a legitimate architectural question.

## Hack Detector
- Fix count: 0 (pure ablation, no new mechanisms)
- Is MATH.md a proof or a description? Description with quantitative framework -- NOT a proof, but appropriate for Type 2
- Metric used as evidence: Composite of execution-based metrics (syntax_valid, answer_correct, factual_overlap, response_quality)
- Kill criteria source: Derived from the coverage-ratio framework (alpha = q_code/q_best). K608 threshold of 50% is generous by design. K609 directly tests specialization.

## Self-Test Audit
1. One-sentence impossibility property: Correctly states there is no impossibility guarantee -- this tests WHETHER the failure mode obtains. Honest. PASS.
2. Cited theorems: Multi-armed bandits (Robbins 1952), MoE specialization (Shazeer 2017). These are real and relevant, though loosely applied (no formal theorem invocation with precondition checks). Acceptable for Type 2. PASS.
3. Predicted numbers: P1 (code >= 80% on structured), P2a/P2b (competing hypotheses for prose), P3 (V < 0.1 vs V > 0.3). These are specific and falsifiable. PASS.
4. Falsification condition: alpha < 0.3 on prose domains would refute universal adapter hypothesis. Clear. PASS.
5. Hyperparameter count: 0 (ablation). PASS.
6. Hack check: Not adding fixes; questioning whether a core component is needed. PASS.

## Mathematical Soundness

MATH.md presents a coverage-ratio framework, not a formal proof. For a Type 2
guided exploration, this is acceptable -- the framework is well-defined and
the unknown (alpha values on prose domains) is precisely identified.

The alpha = q_code(d) / q_best(d) framework is straightforward and correctly
applied. The routing value formula V = 1 - sum(q_code)/sum(q_best) is
algebraically correct when alpha is defined per-domain.

**One issue:** The framework assumes q_best(d) = max_i q_i(d), but the
experiment reveals cases where q_code > q_domain AND q_code > q_base but
q_code < q_base on some metrics (e.g., response_quality on code domain:
base=0.74, code=0.50). The composite score hides metric-level tradeoffs.
The paper acknowledges this in Finding #5 but does not revise the framework
to account for multi-objective quality. This is a limitation, not a fatal flaw.

**Routing value calculation discrepancy:** PAPER.md states V = 1 - 2.787/2.562
= -8.8%. But V should be 1 - sum(q_code)/sum(q_best). Here q_code total =
2.787 and q_best (oracle) total = 2.562. Since q_code > q_best, this means
alpha > 1 on average and routing is actively harmful. The formula gives
V = 1 - 2.787/2.562 = 1 - 1.088 = -0.088 = -8.8%. Correct.

However, this is a striking result: the "oracle" domain-specific routing is
WORSE than universal code adapter. This could mean (a) the domain adapters are
genuinely bad, or (b) the evaluation metrics favor code-style responses. The
paper argues (a) with supporting evidence (legal/finance adapters worse than
base). Plausible but worth noting.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table. Assessment:

| Prediction | Match | Quality |
|-----------|-------|---------|
| P1: Code >= 80% on code, math | YES (alpha=1.0, 1.20) | Strong -- exceeded prediction |
| P2a vs P2b: Coverage on prose | H2a confirmed (alpha 0.97-1.25) | Clear resolution of unknown |
| P3: V < 0.1 if H2a | V = -8.8% (negative) | Exceeded prediction |

The predictions were directionally correct and the measurements clearly
resolve the H2a/H2b unknown. This is a well-executed guided exploration.

## NotebookLM Findings

Skipped -- the experiment is straightforward enough that a deep review is
not needed. The mathematical framework is simple (coverage ratios), the
experimental design is clean (4 configs x 5 domains x 10 queries), and the
results are unambiguous.

## Novelty Assessment

This is an ablation, not a novel method. Its value is architectural: it
establishes that current SFT adapters lack sufficient specialization to
justify routing. This is an important negative result that prevents wasted
effort on routing mechanisms for under-specialized adapters.

The connection to MoE literature (Shazeer 2017, DES-MoE) is appropriate.
The contrast with DES-MoE's 43-76% wrong-routing penalty vs this experiment's
0% or negative penalty is a genuinely informative comparison.

## Concerns (non-blocking)

1. **Composite metric masks tradeoffs.** Medical domain: domain adapter has
   higher factual_overlap (0.535 vs 0.478) but lower response_quality (0.689
   vs 0.734). The 50/50 composite hides that a user might prefer factual
   accuracy over response fluency. The paper should state that the composite
   weighting is arbitrary and results could change with different weights.

2. **response_quality metric is suspicious.** Base model achieves the highest
   response_quality on ALL 5 domains (0.69-0.78). This suggests the metric
   may reward fluent but uninformative responses. If response_quality is
   essentially measuring "does it look like nice prose," it penalizes all
   adapters equally and inflates base model scores. This weakens the claim
   that code adapter is "better" on prose domains -- it may just be "less
   bad" at format disruption.

3. **n=10 per domain.** The paper acknowledges this. The medical result
   (alpha=0.97, difference of 0.016) is clearly not significant. Legal
   (alpha=1.25, difference of 0.075) and finance (alpha=1.11, difference
   of 0.036) are borderline. Only math (alpha=1.20, difference of 0.131)
   is likely significant at n=10.

4. **lora_scale=20 is extreme and not ablated.** At scale 20, adapter
   influence is amplified 20x. Domain-specific adapters trained on harder
   data (legal loss 2.84 vs code loss 1.28) may be disproportionately
   harmed by this scaling. A sweep over lora_scale (1, 5, 10, 20) could
   reveal that domain adapters improve at lower scales while code adapter
   degrades. This is acknowledged in limitations but deserves emphasis.

5. **Findings #208 and #209 status SUPPORTED is appropriate** given the
   guided exploration type and the clear resolution of the H2a/H2b unknown.
   Not over-claimed.

## Macro-Scale Risks (advisory)

1. The code adapter's universality may be an artifact of (a) a small base
   model with limited domain knowledge, (b) short 300-step training, or
   (c) lora_scale=20 overwhelming subtle domain signals. At macro scale
   with longer training and properly tuned scale, domain specialization
   may emerge and routing may become valuable.

2. The finding that "SFT mainly teaches format compliance" may not hold for
   larger models where SFT can genuinely inject domain knowledge.

3. This result should NOT be used to permanently remove routing from the
   architecture. It should gate routing behind a prerequisite: "demonstrate
   adapter specialization first."

## Verdict

**PROCEED**

This is a well-designed guided exploration that cleanly resolves its stated
unknown (does routing add value for current SFT adapters?). The answer is
clearly "no, and it actively hurts." The mathematical framework is simple
but appropriate. Predictions are specific, falsifiable, and matched by
measurements. Kill criteria are sensible. The experiment correctly identifies
that the root cause is under-specialized adapters, not broken routing.

The non-blocking concerns (composite metric masking, response_quality
suspicion, n=10, lora_scale not ablated) are real limitations that the paper
mostly acknowledges. They do not invalidate the core finding that current
domain adapters are not specialized enough to justify routing.

The finding status of SUPPORTED is correctly calibrated -- not over-claimed
as conclusive, appropriately hedged with the caveat that better adapters
could change the picture.
