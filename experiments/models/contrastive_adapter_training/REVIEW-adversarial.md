# Peer Review: Contrastive Adapter Training (RE-REVIEW)

## Experiment Type
Guided exploration (Type 2) -- operates within the framework of NeuroLoRA COL
(2603.12378), LoRACLR (2412.09622), and LIMA (2305.11206). The unknown being
explored: does contrastive weight decorrelation during training produce
domain-specialized (not format-specialized) adapters?

## Re-Review: Were All 6 Original Issues Fixed?

### Issue 1: Downgrade "Theorem 1" to Conjecture -- FIXED
MATH.md Section D now reads "Conjecture 1 (Contrastive Decorrelation of Format
Component)" and explicitly states "This is a mechanism description, not a formal
proof." The dimensionality error (||Delta_F||^4) is acknowledged and corrected.
Three previously unstated assumptions are now listed (multi-dimensional format,
correlated domain content, no convergence guarantee). The self-test answer #1
honestly says "no impossibility property is proven."

### Issue 2: Critical ablation -- FIXED
PAPER.md now documents that the baseline condition IS lambda=0 at scale=2.0,
meaning the ablation was already embedded in the experimental design. A comparison
table quantifies the delta. This is the correct resolution -- no re-run was needed
because the experiment already controlled for scale.

### Issue 3: "Math is new generalist" claim -- FIXED
PAPER.md Key Observation #2 now includes a caveat: "This could simply replicate the
code-universality pattern (Finding #208) with a different data source. Without
behavioral benchmarks (MMLU, GSM8K, HumanEval), we cannot distinguish..." This is
appropriately scoped.

### Issue 4: Finding #214 as primary -- FIXED
The Findings section now leads with "PRIMARY Finding: Weight orthogonality !=
behavioral specialization (confirms 2510.03262)" and frames it as confirmatory, not
novel. The 99.6% vs 0.3-5.7% gap is the headline. Correct framing.

### Issue 5: K617 assessment -- FIXED
PAPER.md now states "K617: FAIL (KILL triggered)" without equivocation. The
discussion explains why the kill criterion was triggered (5/5 domains have alpha > 0.9)
and provides a constructive lesson for future kill criteria design (separate code
dominance from low differentiation). The results.json overall_verdict remains "KILL."
This is honest and correct.

### Issue 6: Implementation-theory gap -- FIXED
MATH.md Section G2 documents four specific divergences: round-robin vs joint
optimization, stale contrastive gradients, frequency (every 5 steps), and partial
parameters (lora_b only). PAPER.md Limitation #5 references this section. This is
thorough.

**Summary: All 6 issues are resolved.**

## Hack Detector
- Fix count: 2 (contrastive loss + low LoRA scale). Not flagged. Scale is controlled
  for (both conditions use scale=2.0).
- Is MATH.md a proof or a description? **Mechanism description, honestly labeled as
  such.** "Conjecture 1" not "Theorem 1." Self-test admits no impossibility property.
  This is appropriate for Type 2 guided exploration.
- Metric used as evidence: PPL cross-evaluation. PAPER.md Limitation #1 acknowledges
  PPL is a weak proxy (r=0.08 with task quality). The primary finding (weight orth !=
  behavioral spec) does not depend on PPL being a good metric -- it uses the PPL
  weakness as evidence for the finding itself.
- Kill criteria source: K617 from Finding #208 (reasonable provenance). K618/K619 are
  convergence/regression checks. The 15% hypothesis threshold is acknowledged as
  "too aggressive" -- not derived from theory.

## Self-Test Audit

1. **One-sentence impossibility property:** Honest answer: "no impossibility property
   is proven." This is correct for a Type 2 guided exploration. No evasion.

2. **Cited theorems:** NeuroLoRA COL, LoRACLR, LIMA. These are real papers providing
   real mechanisms. MATH.md correctly identifies them as techniques/hypotheses, not
   as theorems with conditions to verify. The addition of 2510.03262 (Rethinking
   Orthogonality) as a critical caveat is appropriate.

3. **Predicted numbers:** Inter-adapter cosine < 0.1, domain beats code by >= 15%,
   training loss < 2x. The 15% is not derived (acknowledged). The cosine < 0.1 is
   reasonable from the mechanism but not bounded. Acceptable for Type 2.

4. **Falsification condition:** "If standard SFT adapters already have low inter-adapter
   cosine, the format-sharing hypothesis is wrong." This targets the assumption, not
   just the experiment. Good. The experiment confirmed format-sharing (baseline cos = 0.97).

5. **Hyperparameter count:** 2 (lambda, LoRA scale). Honestly stated with rationale
   for each choice.

6. **Hack check:** Clean. Single mechanism, not stacking fixes. Correctly states it
   replaces (not augments) independent SFT.

**Self-test: PASS. No blanks, no evasions.**

## Mathematical Soundness

MATH.md correctly identifies itself as a mechanism description, not a formal proof.
For a Type 2 guided exploration, this is acceptable IF the proven framework is clearly
stated and the unknown is precisely identified.

**Proven framework:** NeuroLoRA COL achieves weight decorrelation. LIMA hypothesis:
SFT teaches format. Both are established. The combination (use COL to suppress format
component) is the exploration.

**Unknown being explored:** Does weight decorrelation produce behavioral domain
specialization? Answer from experiment: no, or at least not at the scale measured
(0.3-5.7% differentiation despite 99.6% weight decorrelation).

**Remaining issues (not blocking):**
- The worked example in Section F is illustrative but hand-wavy. The claim
  "cos(delta_code, delta_math) approx 0.0 (naturally decorrelated domain content)"
  assumes the conclusion. If domain content components share structure, the contrastive
  loss would also suppress beneficial transfer. This is listed as unstated assumption #2
  but remains unverified.
- Section G describes joint training ("all N adapters are updated each step") while
  Section G2 documents that implementation uses round-robin. G and G2 are now clearly
  separated but G could note "this describes the idealized scheme; see G2 for actual
  implementation."

**These are advisory, not blocking.**

## Prediction vs Measurement

PAPER.md contains the required table. Assessment:

| Prediction | Measured | Match? | Comment |
|-----------|----------|--------|---------|
| Inter-adapter cos < 0.1 | 0.0036 | YES | 27x overshoot; prediction not tight but directionally correct |
| Baseline cos > 0.3 | 0.9726 | YES | Confirms format dominance hypothesis |
| Code NOT universal best | 0/5 | YES | Mechanism works as intended |
| Training loss < 2x | 1.0-1.22x | YES | Trivially met |
| All domains improve vs base | 5/5 | YES | Positive but low bar |
| Domain beats code >= 15% | max 5.7% | NO | Core hypothesis not supported |

The critical emergent finding (99.6% weight decorrelation -> only 0.3-5.7% PPL
differentiation) was NOT predicted but is correctly identified as the primary result.
For a Type 2 exploration, discovering unexpected structure in the unknown is a valid
outcome.

## Did Fixes Introduce New Problems?

No. The revisions are conservative:
- Relabeling (Theorem -> Conjecture, Theorem -> Observation) adds honesty without
  changing content.
- The ablation documentation clarifies what was already in the data.
- Caveats narrow claims rather than introducing new ones.
- Section G2 adds transparency about implementation.
- K617 FAIL acceptance is consistent with results.json.

One minor inconsistency: MATH.md Section D header says "D. Mechanism Description
(Not a Formal Proof)" but Section D (Predictions) also starts with "D." This is a
formatting issue, not a substantive problem.

## Experiment Type Verification

**Type 2 (guided exploration) requirements:**
1. State the PROVEN framework: YES. NeuroLoRA COL (decorrelation mechanism) + LIMA
   (format hypothesis) are established.
2. Identify the UNKNOWN precisely: YES. "Does forcing weight-space decorrelation
   during training produce genuinely domain-specialized adapters (not just
   format-specialized ones)?"
3. Narrow the unknown: YES. The experiment narrows the unknown by showing that weight
   decorrelation alone does NOT produce behavioral specialization (0.3-5.7%
   differentiation). This is a genuine narrowing -- it rules out a plausible mechanism.

**Type 2 requirements: MET.**

## Novelty Assessment

The experiment applies known techniques (NeuroLoRA COL, LoRACLR) to a specific setting.
The application is not novel. The primary contribution -- confirming 2510.03262's result
that weight orthogonality is a weak predictor of behavioral specialization -- is
confirmatory. The quantification (99.6% vs 0.3-5.7%) in the specific BitNet/LoRA
setting is a useful data point for the research program.

The ablation table (lambda=0 vs lambda=1 both at scale=2.0) is a clean contribution:
contrastive loss fixes medical adapter, breaks code universality, but only modestly
improves differentiation. The scale finding (2.0 vs 20.0) from Finding #212 is
confirmed as independently important.

## Macro-Scale Risks (advisory)

1. Joint training of all N adapters breaks independently-trainable property (VISION.md).
   At scale, this is impractical.
2. Grassmannian skeleton already provides orthogonality guarantees (17x decorrelation).
   Contrastive training during SFT may be redundant.
3. The primary finding (weight orth != behavioral spec) suggests that the Grassmannian
   approach should be evaluated on BEHAVIORAL metrics, not just cosine similarity.

## Verdict

**PROCEED**

All 6 original REVISE issues are resolved. The fixes are honest, conservative, and
do not introduce new problems. The experiment is correctly classified as Type 2 guided
exploration and meets all requirements for that type.

The experiment's own verdict is KILL (K617 triggered), which is accepted without
equivocation. Despite the KILL on the original hypothesis, the experiment produced
a valuable finding: weight-space orthogonality (99.6% decorrelation) does not
translate to behavioral specialization (0.3-5.7% PPL differentiation), confirming
2510.03262. This is a legitimate Type 2 outcome -- the exploration narrowed the
unknown by ruling out a plausible mechanism.

The findings are appropriately scoped:
- Primary finding (weight orth != behavioral spec) is framed as confirmatory, not novel
- Scale finding is noted as potentially more important than contrastive training
- Math-as-generalist claim is properly caveated
- Kill criteria failure is accepted and analyzed constructively

This experiment is ready to be recorded as a supported finding with the K617 KILL
acknowledged. No further revisions needed.
