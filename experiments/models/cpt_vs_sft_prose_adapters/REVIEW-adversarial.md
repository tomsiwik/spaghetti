# Peer Review: CPT vs SFT Prose Adapters

## Experiment Type
Guided exploration (Type 2)

## Hack Detector
- Fix count: 1 (single variable change: training data format). CLEAN.
- Is MATH.md a proof or a description? **Description dressed in equations.** The "information-theoretic framing" in Step C is not a proof. It defines I_format and I_knowledge, asserts CPT sets I_format=0, and concludes CPT allocates all capacity to knowledge. There is no derivation, no bound proven, no QED. The decomposition I = I_format + I_knowledge is asserted without proving (a) the decomposition is valid (format and knowledge may share mutual information), (b) that CPT truly has I_format = 0 (raw legal text has implicit format -- paragraph structure, citation patterns, etc.), or (c) that the capacity bound C(r,d) is tight enough that the format/knowledge split actually matters at rank-16.
- Metric used as evidence: Factual recall score (token overlap). Not proven to predict behavioral outcomes. A score of 0.097 vs 0.056 could reflect generation style differences rather than genuine knowledge differences.
- Kill criteria source: K1 and K2 are reasonable behavioral thresholds. K3 is derived from prediction P3 (convergence). These are fair for guided exploration.

## Self-Test Audit
1. **One mathematical property:** "CPT allocates full adapter capacity to domain knowledge." This is a single property. PASS.
2. **Cited theorems:** Gururangan 2020 (2004.10964) is real but is an empirical study, not a theorem. Calling it "Theorem via empirical demonstration" is misleading -- it is an observation, not a proved result. LIMA (2305.11206) is real but is also an observation, not a theorem. Neither provides the proven mathematical framework required for Type 2 exploration. FLAG.
3. **Specific numbers:** P1: >=15% improvement. P2: >=15%. P4: >=80% coherence. These are specific and falsifiable. PASS.
4. **Falsification condition:** "CPT adapters score WORSE than SFT on factual recall for BOTH domains." This is reasonable. PASS.
5. **Hyperparameters added:** 0 new hyperparameters. PASS.
6. **Hack check:** No stacked fixes. Single variable changed. PASS.

## Mathematical Soundness

**BLOCKING ISSUE: No proven framework for Type 2.**

MATH.md claims this is Type 2 (guided exploration within a proven framework). The cited framework is "the two-regime model (Finding #249)." But Finding #249 is itself an empirical observation from a micro-experiment, not a formally proven theorem. A finding from a prior micro-experiment does not constitute "proven math" in the sense required by the proof-first methodology.

The information-theoretic argument has three specific problems:

1. **The I_format = 0 claim for CPT is false.** Raw legal text has format: paragraph breaks, citation conventions, discourse structure, sentence length distributions. CPT on raw legal text absolutely encodes format information about legal prose. The claim that "raw text has no separate format component" conflates "instruction-response format" with "format" generally. The argument should be: CPT encodes domain-natural format while SFT encodes instruction-response format. This is a weaker claim.

2. **The capacity decomposition is not additive in general.** I(delta_theta; D) = I_format + I_knowledge assumes format and knowledge are independent information components. They are not. Legal terminology appears in specific syntactic patterns -- the format carries knowledge. The mutual information I(format; knowledge) could be substantial, making the decomposition I = I_format + I_knowledge - I(format; knowledge) + higher-order terms.

3. **No bound on how much knowledge rank-16 can encode.** The argument says CPT uses "all" capacity for knowledge, but never estimates whether rank-16 capacity is sufficient for legal domain knowledge. The experiment actually showed it is NOT sufficient (CPT legal = base model, meaning zero knowledge injected). This is predictable: 10.9M parameters encoding 80K tokens of legal text is ~137 bytes per token of training data, but the adapter must generalize, not memorize.

**What the math should have predicted:** Given 80K tokens of training data and rank-16 LoRA (~10.9M ternary params), the effective information capacity is bounded. A back-of-envelope calculation: at ~1 trit per parameter, the adapter can encode ~17M trits = ~10.7M bits. The training corpus has H(D) * 80K entropy, where H(D) for legal English is roughly 4-6 bits/token, giving ~400K bits of information. The capacity exceeds the data, so the bottleneck is not capacity but rather whether 200 iterations of SGD on 80K tokens is sufficient to find the relevant subspace. This would have predicted that CPT at this data scale would barely move the adapter -- which is exactly what happened.

## Prediction vs Measurement

The prediction-vs-measurement table exists and is well-structured. Results:

| Prediction | Measured | Verdict |
|-----------|----------|---------|
| P1: CPT >= 15% better on legal | +74.2% (but CPT = base, SFT degraded) | Technically YES but misleading |
| P2: CPT >= 15% better on medical | -0.2% (tie) | NO |
| P3: Convergence | Legal: NO, Medical: YES | PARTIAL |
| P4: Coherence >= 80% | 100% both | YES |
| P5: < 2 hours | 127 seconds | YES |

**Critical observation the paper correctly identifies:** P1 is misleading. CPT did not improve over base (0.097 vs 0.098 = -1%). It "beat" SFT only because SFT degraded from base. The prediction framework asked the wrong question: it predicted CPT > SFT, when the interesting measurement is CPT > base. CPT failed to inject any knowledge whatsoever.

**The paper is honest about this.** Section "Why Legal CPT Won (It Did Not Actually Win)" directly states the CPT adapter learned nothing. This intellectual honesty is commendable and the interpretation section is the strongest part of the paper.

## NotebookLM Findings
Skipped -- the experiment is already killed, the paper is honest about results, and the issues are clear from direct reading.

## Novelty Assessment
- Gururangan et al. 2020 already showed DAPT works but used millions of tokens, not 80K. The scale gap is acknowledged.
- The finding that "CPT at 80K tokens / rank-16 / 200 iters is a no-op" is not novel -- it follows directly from the data scale limitation.
- The finding that "SFT damages legal prose at scale=20" was already known from Finding #209.
- The genuinely useful new information is the causal mechanism: SFT degrades legal because format patterns overwrite base model knowledge. This narrows the unknown in a useful way.

## Macro-Scale Risks (advisory)
- The entire argument depends on the claim that CPT would work with more data. This is plausible (Gururangan showed it at scale) but untested. At macro scale, the data requirement for meaningful CPT may be prohibitive for a runtime composition system.
- The real question for the architecture is whether domain knowledge should come from adapters at all, or whether it should come from the base model with adapters only for routing/style. LIMA already suggests the latter.

## Verdict

**KILL (confirmed)**

The experiment was correctly killed. The review confirms the kill is justified and identifies additional issues:

1. **No proven mathematical framework.** The "information-theoretic framing" is a description, not a proof. The cited "theorems" are empirical observations. For a Type 2 experiment, the framework it operates within must be proven math, not prior micro-experiment findings. This is a BLOCKING issue that would have required REVISE if the experiment were still active.

2. **The core prediction was wrong in an informative way.** CPT was predicted to inject knowledge (+15% over SFT). Instead, CPT injected nothing (= base model). The paper correctly identifies this and pivots the interpretation, which is good scientific practice.

3. **The actually useful finding is already known.** "SFT damages legal at scale=20" was Finding #209. The new information (causal mechanism via format overwriting) is valuable but incremental.

4. **The experiment answered a question nobody should have asked.** Given LIMA's finding that knowledge comes from pre-training, and given 80K tokens is orders of magnitude below DAPT data requirements, the prediction that 80K tokens of CPT would inject knowledge into a 2B model was implausible from the start. The math, had it been done rigorously, would have predicted this failure.

**For future work:** If revisiting CPT, first prove a lower bound on the data required for measurable knowledge injection at rank-16 in a 2B model. The experiment should test whether that bound is achievable, not whether CPT "works" in the abstract.
