# Peer Review: Text Classification Routing (contrastive_routing_n5)

## Re-Review Status

This is a re-review after 5 doc-only fixes from the original REVISE verdict.

| Fix | Required | Applied? | Assessment |
|-----|----------|----------|------------|
| 1. Rename to "Text Classification Routing" | YES | YES | Title changed in both MATH.md and PAPER.md. Naming note added explaining the contrastive origin. Directory still named contrastive_routing_n5 but this is acceptable for traceability. |
| 2. Downgrade theorems to design principles | YES | YES | MATH.md Section D now uses "Design Principle 1" and "Design Principle 2". No Theorem/Proof/QED blocks remain. Honest framing as hypotheses to test empirically. |
| 3. Integration discussion with softmax router | YES | YES | PAPER.md adds "Integration with Existing Routing" section explaining: softmax router handles NTP adapters at N=24; this TF-IDF classifier handles SFT adapters at N=5. Different settings, complementary methods. |
| 4. Narrow claim from "routing solved" to scoped | YES | YES | Critical finding header reads "works for 5 Well-Separated Domains". No overreaching claims. Broader problems (fuzzy boundaries, N>5) explicitly called open. |
| 5. K607 marked inconclusive | YES | YES | PAPER.md states K607 result is "inconclusive" due to unreliable keyword density metric (Finding #179). Decomposition section properly caveats: "could reflect genuine adapter quality issues, metric noise, or both." |

All 5 fixes properly applied.

## Experiment Type

Guided exploration. MATH.md declares this explicitly: the proven framework is input-based
adapter retrieval (LoraRetriever, Zhao et al. 2024), and the unknown is whether 5 SFT
domains on a micro-scale model produce sufficiently distinct vocabulary distributions for
reliable TF-IDF classification routing. The experiment narrows this unknown: yes, they do
(90% accuracy), with legal-finance confusion as the weak point (70-80%).

This is a legitimate guided exploration -- it operates within the established principle
of input-based routing and discovers the empirical separability of these specific 5 domains.

## Hack Detector

- Fix count: 1 (replace energy gap routing with TF-IDF classifier). Clean single change.
- Is MATH.md a proof or a description? Description with honest "Design Principle" framing. No longer pretends to be a proof. This is appropriate for guided exploration.
- Metric used as evidence: routing accuracy (well-defined, directly measurable) for K605; math correctness (behavioral) for K606; keyword density (acknowledged unreliable) for K607.
- Kill criteria source: K605 threshold (70%) is a reasonable operational floor. K606 threshold (60%) is derived from the prior Finding #204 baseline. K607 is marked inconclusive. Mixed provenance but transparent.

## Self-Test Audit

1. **One-sentence impossibility property:** "Routing depends only on input text features, which are independent of adapter NLL, so NLL-dominance of one adapter cannot corrupt routing decisions." -- This is a genuine and singular property (function signature independence). PASS.

2. **Cited work:** LoraRetriever (Zhao 2024), InfoNCE (van den Oord 2018), Supervised Contrastive (Khosla 2020) are real papers. MATH.md now honestly notes these "motivated the principle of input-based routing but are not directly applied in this experiment." The naming note at the top makes the gap explicit. PASS (improved from original).

3. **Specific numbers predicted:** Routing >= 90%, math correctness >= 60%, 0/5 domains degraded. Specific and falsifiable. PASS.

4. **Falsification condition:** "If domains are NOT separable in TF-IDF space (accuracy < 50%), the assumption of vocabulary distinctness is wrong." Targets the core hypothesis. PASS.

5. **Hyperparameters added:** 0 for the classifier (scikit-learn defaults). lora_scale=20.0 inherited. Honest. PASS.

6. **Hack check:** Single mechanism replacement. Not a hack. PASS.

## Mathematical Soundness

No formal theorems claimed (correctly, for a guided exploration). The design principles are:

**DP1 (Domain Separability Hypothesis):** Our 5 domains have distinct enough vocabulary for TF-IDF linear separability. This is framed as a hypothesis to test, not a theorem. The rationale gives intuition (domain-specific keywords) and acknowledges the empirical question: "whether our specific 5 domains are sufficiently separated is the empirical question this experiment answers." Honest. The legal-finance overlap is explicitly flagged as a risk.

**DP2 (NLL-independence):** A classifier on input text is independent of adapter NLL by construction. Now correctly framed as "a property of the function signature" and "a design choice, not a mathematical discovery." This was the main dishonesty in the original submission and is now fixed.

No mathematical soundness issues remain because no mathematical claims are made beyond what is justified.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table:

| Prediction | Measured | Match |
|---|---|---|
| Routing >= 90% | 90% (45/50) | YES |
| Energy gap ~36% | 36% | YES (baseline) |
| Per-domain medical >= 70% | 100% | YES |
| Per-domain code >= 70% | 100% | YES |
| Per-domain math >= 70% | 100% | YES |
| Per-domain legal >= 70% | 80% | YES |
| Per-domain finance >= 70% | 70% | YES (at threshold) |
| Math correctness >= 60% | 80% | YES |
| 0/5 domains degraded | 3/5 worse | NO (inconclusive) |

The routing prediction of exactly 90% against a >= 90% threshold remains suspicious (possible post-hoc calibration), but the kill criterion K605 uses 70%, which was clearly set beforehand and is cleanly passed. The 90% prediction in MATH.md D' is a stretch target, not the kill criterion, so even if post-hoc, the actual kill criterion is met.

K607 failure is now properly marked inconclusive. PAPER.md does not draw conclusions about adapter quality from the unreliable keyword density metric. This is the correct handling.

## Remaining Concerns (Non-Blocking)

1. **n=10 per domain remains extremely thin.** 95% CI for 70% at n=10 is approximately [35%, 93%] (binomial). Finance at 70% could plausibly be 40% or 90%. This is acknowledged in Limitations but is worth emphasizing: these numbers are directional only.

2. **Validation/test contamination risk** (Limitation #5) is still unresolved. If validation and test prompts come from the same valid.jsonl, the 93.6% val accuracy may be inflated.

3. **The 90% prediction hitting exactly 90%** still suggests possible post-hoc threshold calibration. Not blocking because the actual kill criterion (70%) was clearly pre-set and cleanly exceeded.

4. **Routed generation is 5x slower** (644s vs 125s). results.json shows this. Not discussed in PAPER.md. At production scale this matters, though at micro scale it is expected overhead from loading adapters sequentially.

## Novelty Assessment

Low novelty (unchanged from original review). TF-IDF + logistic regression for text classification is textbook. The contribution is the diagnostic finding: routing was not the bottleneck -- adapter quality for prose domains is. This decomposition (routing works, adapters may not) is the genuine value.

## Macro-Scale Risks (advisory)

- TF-IDF will fail with fuzzy domain boundaries (acknowledged).
- At N>5, vocabulary overlap increases and linear separation degrades.
- VISION.md softmax router handles N=24 for NTP; the integration story for SFT at scale remains open.
- The TF-IDF approach is diagnostic, not production. A learned router (BERT-tiny or the existing softmax router) is the path forward.

## Verdict

**PROCEED**

All 5 REVISE fixes have been properly applied:
- The experiment is honestly framed as guided exploration with design principles (not theorems).
- Citations are properly scoped as motivation, not direct foundations.
- Integration with the softmax router is discussed.
- Claims are narrowed to "5 well-separated domains."
- K607 is marked inconclusive with the metric unreliability caveat.

The finding should be recorded as `supported` with caveats:
- Routing accuracy is directional (n=10 per domain).
- Generation quality assessment is inconclusive (unreliable metric).
- The real finding is the decomposition: input-based routing works for SFT adapters where NLL-based routing fails, and adapter quality (not routing) is the remaining bottleneck.
