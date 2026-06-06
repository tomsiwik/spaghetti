# Peer Review: Cross-Domain Scale Phase Transition

## Experiment Type
Guided exploration. MATH.md correctly identifies the proven framework (Finding #250: math phase transition at s*=[4,6]) and the unknown (whether s* is universal or domain-dependent). The exploration successfully narrowed the unknown: the transition is math-specific.

## Hack Detector
- Fix count: 0. Pure measurement experiment, no mechanisms added.
- Is MATH.md a proof or a description? Description of exploration framework -- appropriate for guided exploration type. No Theorem/Proof/QED, which is correct for Type 2.
- Metric used as evidence: Per-domain accuracy at varying LoRA scales. The metric is directly tied to the behavioral question (does capability activate at a critical scale?).
- Kill criteria source: Derived from predictions. K1 threshold (0.3 jump) is half the observed math jump (0.60), which is a reasonable relaxation. K2 and K3 also follow from the prediction table.

## Self-Test Audit
1. One-sentence impossibility property: "Whether the critical LoRA scale s* is universal across domains or domain-dependent." This is one property. PASS.
2. Cited theorems: Hu et al. (2106.09685) is real. Finding #250 is internal but documented. The "softmax threshold theory" in Section C is informal reasoning, not a cited theorem -- but this is acceptable for guided exploration framing. PASS with note.
3. Specific numbers: H1 predicts jump >= 0.3 at s=[4,6]. H2 predicts |s*_code - s*_math| > 4. Prediction table with 8 scales x 5 columns. PASS.
4. Falsification: K1 FAIL means phase transitions are math-specific. This targets the hypothesis, not just the experiment. PASS.
5. Hyperparameters: 0 added. Sweeping existing parameter. PASS.
6. Hack check: No. Measurement experiment. PASS.

Self-test is complete and honest.

## Mathematical Soundness

MATH.md is a well-structured exploration framework, not a proof. For guided exploration, the relevant question is: does the framework correctly identify the unknown and design measurements to narrow it?

**What holds:**
- The LoRA perturbation scaling argument (delta_y = s * B^T A^T x) is correct and well-cited.
- The three competing hypotheses (H1 universal, H2 domain-dependent, H3 metric-dependent) span the plausible outcome space reasonably well.
- The prediction table makes specific, falsifiable predictions under each hypothesis.

**Weaknesses:**
- The "softmax threshold theory" (Section C, paragraph 4) is hand-waving, not a cited result. The claim that "tasks requiring subtle attention shifts have smaller margins" is plausible but unproven. This does not invalidate the exploration but should not be treated as a foundation.
- Assumption A3 acknowledges n=10 is marginal (p=0.18 Fisher exact for a 0.3 jump). The experiment proceeds anyway, which is acceptable for guided exploration but means null results have low statistical power. The code domain's non-monotonic behavior could partly be noise from small n.

## Prediction vs Measurement

PAPER.md contains the prediction-vs-measurement table. Results are clear:

| Prediction | Predicted | Measured | Match |
|-----------|-----------|----------|-------|
| P1: Code jump >= 0.3 | >= 0.3 | 0.154 | NO |
| P2: Medical jump >= 0.3 | >= 0.3 | 0.024 | NO |
| P3: |s*_code - s*_math| > 4 | > 4 | 1.0 (NO) |
| P4: Code plateau >= 0.5 | >= 0.5 | 0.624 | YES |
| P5: Medical plateau >= 0.5 | >= 0.5 | 0.291 | NO |

All three kill criteria fail. The researchers correctly identify this as a KILL with an informative negative result.

**One concern with the analysis:** The code data (0.35, 0.50, 0.57, 0.36, 0.50, 0.50, 0.49, 0.62) is extremely noisy. The drop from 0.574 at s=4 to 0.357 at s=6 is a 0.217 swing with only 10 prompts. The bimodal per-prompt analysis (Group A disrupted, Group B activated) is the most interesting finding but is presented as observation rather than tested against a hypothesis. With n=10, the bimodal claim rests on 5 prompts per group -- too few to be confident.

**The key insight is sound:** The math "phase transition" is likely an evaluation artifact (binary exact-match metric amplifying format activation) rather than a fundamental LoRA property. This is a genuinely useful negative result that correctly downgrades Finding #250.

## NotebookLM Findings

Skipping -- the experiment is already killed and the analysis is straightforward. NotebookLM review would not add value to an already-clear negative result.

## Novelty Assessment

This is an internal replication/extension experiment testing the generalizability of Finding #250. No external novelty claim is made. The finding that phase transitions are evaluation artifacts rather than LoRA properties is useful for the project's internal knowledge base, particularly because it prevents building routing logic around a non-universal phenomenon.

No prior art concern -- this is testing an internal finding's generalizability, which is the right thing to do before building on it.

## Macro-Scale Risks (advisory)

Not applicable -- the experiment killed the hypothesis. The macro implication is positive: do not build per-domain scale routing based on a phase transition that does not generalize. This simplifies the architecture.

## Verdict

**PROCEED** (as a killed finding)

The experiment is correctly killed by the researchers. This is a well-executed guided exploration that:

1. Had clear, falsifiable predictions derived from the framework.
2. Collected measurements that cleanly refute the universality hypothesis.
3. Produced a genuinely useful negative finding: the math phase transition is an evaluation artifact, not a fundamental LoRA property.
4. Correctly identifies what would overturn the kill (better adapter, execution-based eval, larger n).

The MATH.md framework, self-test, and PAPER.md analysis are all at the right level for guided exploration. The prediction table is complete and the kill criteria are well-derived.

**Minor recommendations (not blocking):**
1. The bimodal code behavior (Group A disrupted vs Group B activated) deserves a follow-up exploration if code routing matters. With n=5 per group, it is suggestive but not conclusive.
2. Finding #250 should be annotated to note that its phase transition is evaluation-method-dependent, not a fundamental LoRA property. This downgrades it from "supported" to "provisional" or adds a caveat.
3. The medical adapter's near-zero effect (max delta +0.028) means the medical domain contributed no useful signal to this experiment. Future cross-domain tests should verify adapter quality on held-out data before sweeping scales.
