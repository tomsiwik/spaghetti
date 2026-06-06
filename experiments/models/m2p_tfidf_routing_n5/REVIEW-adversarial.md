# Peer Review (Re-review): exp_m2p_tfidf_routing_n5

## Experiment Type
Verification (Type 1) -- proof precedes experiment, measurements verify predictions.

## Re-review Context

Three blocking issues were raised in the first review. This re-review verifies the fixes
and checks for new issues introduced during revision.

---

## Fix Verification

### Fix 1: Lemma C.4 false prefixes -- FIXED (with residual)

**Original issue:** Lemma C.4 claimed "sort:" and "reverse:" prefixes existed in the data
and cited nonexistent trigrams "so>" and "re>".

**What was done:** Lemma C.4 (MATH.md lines 92-152) was rewritten from scratch. The five
domain formats are now correctly described: sort as `{input}>{output}` with ascending-sorted
output, reverse as `{input}>{output}` with reversed output, no prefix labels. The separation
argument for sort vs. reverse now correctly relies on character-ordering statistics in output
bigrams, not nonexistent format prefixes. The ~10-15% confusion prediction for short sequences
is stated and argued from the weak bigram signal at length 2.

**Residual issue (non-blocking):** Theorem C.2 (MATH.md lines 79-80) still contains the
parenthetical example `(e.g., "sort:", "reverse:", "*" triggers)`. The `"sort:"` and
`"reverse:"` n-grams do not exist in the actual data. This is in a parenthetical illustration
of Fisher's theorem, not in the proof itself, so it does not invalidate any derivation. The
`run_experiment.py` docstring (line 17) also still references `"sort:"`. Both should be cleaned
up but are not blocking.

**Verdict: FIXED.** The core proof mechanism is now correct. Residual cosmetic references
to nonexistent prefixes do not affect any theorem or derivation.

### Fix 2: Theorem 2 QED downgrade -- FIXED

**Original issue:** QED claimed "format-string argument is exact" which was false for
sort/reverse.

**What was done:** QED (line 250-252) now reads: "QED (directional; format-string argument
exact for arithmetic, repeat, parity; character-ordering argument statistical for sort,
reverse -- predicting ~10-15% confusion for short sequences)."

**Verdict: FIXED.** The qualification is honest and matches the proof's actual strength.

### Fix 3: Confidence intervals -- FIXED

**Original issue:** No confidence intervals on measurements (30 samples/domain is thin).

**What was done:** PAPER.md now includes a full "Confidence Intervals and Statistical
Reliability" section (lines 57-96). Wilson score intervals at 95% confidence are computed
for all routing accuracy measurements. The overall routing CI is [92.8%, 96.6%], with
the lower bound 22.8pp above the K867 kill threshold.

For quality ratios, the paper honestly states that per-sequence losses were not stored,
so exact CIs cannot be computed post-hoc. A qualitative reliability assessment is provided
(margins of 22.2pp and 12.2pp above thresholds) and future runs are recommended to store
per-sequence losses for exact intervals.

**Independent verification of Wilson CIs:**
- Overall (p=0.95, n=500): lower bound = 0.9273. Claimed: 92.8%. Correct.
- Sort (p=0.89, n=100): lower bound = 0.8137. Claimed: 81.2%. Correct.
- Reverse (p=0.86, n=100): lower bound = 0.7787. Claimed: 78.0%. Correct.

**Verdict: FIXED.** CIs are correctly computed and honestly qualified for quality ratios.

---

## Hack Detector
- Fix count: 1 (clean replacement of MLP router with TF-IDF+LR; no stacking)
- Is MATH.md a proof or a description? Theorem 1 is a genuine proof (trivial tautology, validly argued). Theorem 2 is a proof sketch with a correct mechanism argument for sort/reverse. Lemma C.4 is now a correct description of the data format and separation mechanism.
- Metric used as evidence: Routing accuracy (directly measures routing), quality ratio (proxy for composition quality, defined via loss ratios). Quality ratio is a reasonable proxy but not proven to predict behavioral outcomes.
- Kill criteria source: Derived from proof/prior findings (K867 from Theorem 2 + Finding #207; K868 from K867 x Finding #351; K869 from Finding #351 reproducibility). Derivation chain is sound.

## Self-Test Audit

1. **One-sentence impossibility property:** PASS. Clear, singular, correct.

2. **Cited theorems:** PASS (with note). LoraRetriever, Fisher (1936), Cover (1965) are all real. Fisher's shared-covariance assumption is not verified for TF-IDF features, but the theorem is applied directionally (non-zero Fisher ratio implies better-than-chance separation), which is valid.

3. **Predicted numbers:** PASS. Three specific, falsifiable predictions. All match measurements.

4. **Falsification condition:** PASS. Correctly distinguishes falsifying Theorem 1 (impossible -- tautology) from falsifying Theorem 2 (routing accuracy < 70% on this data).

5. **Hyperparameter count:** PASS. 2 hyperparameters acknowledged, borrowed from Finding #207.

6. **Hack check:** PASS. Clean replacement, not additive fix.

## Mathematical Soundness

### Theorem 1 (Distribution-Agnostic Routing) -- VALID

The proof is correct and trivial: a function that does not depend on model parameters is
invariant to model parameter changes. Calling this a "theorem" elevates a design observation,
but the proof structure is sound and the QED warranted.

### Lemma C.4 (Sequence-level disambiguation) -- VALID (revised)

The five domain formats are now correctly described. The syntactic separation argument for
arithmetic/parity/repeat is correct: `+` and digit presence for arithmetic, `even`/`odd`
for parity, `*` for repeat.

Minor inaccuracy: arithmetic is claimed to use `{+, -, =}` (line 96) but the data generator
only produces addition (`f"{a}+{b}={a+b}"`), never subtraction. The `-` token does not
appear in the data. This does not affect the separation argument since `+` alone is sufficient.

The character-ordering argument for sort vs. reverse is now correct. The analysis of length-2
ambiguity is reasonable. My independent calculation: with input lengths uniform over {2,3,4},
~1/3 are length-2. Of length-2 sort/reverse pairs, ~56% produce identical strings (when
input is same-char or descending). This yields ~(1/3)*0.56*0.5 = 9.3% baseline confusion from
length-2 alone, consistent with the observed 11-14% and the predicted 10-15%.

### Theorem 2 (Sequence-Level Disambiguation) -- VALID (directional)

The proof sketch is now internally consistent. The claimed mechanism (character-ordering
statistics) matches the actual data format. The QED is appropriately qualified.

One discrepancy: line 149 says "TF-IDF over n-grams (1,2)" but the code uses
`ngram_range=(1, 3)` (trigrams). The proof argues from bigram statistics but the experiment
has access to trigrams, making the experimental setup strictly more powerful than the proof
assumes. This makes the proof conservative -- it argues from a weaker feature set than
actually used. Not a flaw, but the discrepancy should be noted.

### Corollary (Kill Criteria Derivation) -- VALID

The formula `q_composed >= alpha_route * q_m2p` is correct under stated assumptions.
Kill thresholds follow logically from Finding #207 and Finding #351.

### Residual issue in Theorem C.2 (non-blocking)

Lines 79-80 still cite `"sort:"` and `"reverse:"` as examples of n-grams with different
distributions. These strings do not exist in the data. This is in a parenthetical
illustration of Fisher's theorem, not in any derivation, so it does not invalidate
the proof. Should be cleaned up in a future pass.

## Prediction vs Measurement

The prediction-vs-measurement table in PAPER.md (lines 25-31) is complete and well-structured.

| Prediction | Predicted | Measured | Match? |
|---|---|---|---|
| Routing accuracy >= 70% (expected ~90%) | >= 70% | 95.0% [92.8%, 96.6%] | YES |
| Composition quality >= 70% of SFT | >= 70% | 92.2% | YES, strong margin |
| Oracle quality >= 80% (~93.3% expected) | >= 80% | 92.2% | YES, within 1.1pp |
| Sort/reverse confusion ~10-15% | 10-15% | 11-14% | YES, dead center |

All predictions match. The sort/reverse confusion prediction is particularly well-calibrated:
10-15% predicted, 11-14% measured. This demonstrates the revised proof mechanism (character-
ordering statistics) correctly identifies the separation mechanism.

The TF-IDF quality = oracle quality coincidence (both 92.2%) is adequately explained in
PAPER.md by the structural similarity of sort/reverse adapters.

## NotebookLM Findings

NotebookLM was not used for this re-review. The review was conducted through direct document
and code inspection with independent mathematical verification.

## Novelty Assessment

**Prior art:** LoraRetriever (He et al., 2402.09997) for text-based routing decoupled from
model internals. Finding #207 established TF-IDF+LR routing at 90% on SFT domains.

**Delta:** Modest but real. Confirms that text-based routing eliminates the covariate-shift
failure mode in M2P composition, which is a necessary building block. The sort/reverse
confusion analysis (character-ordering statistics, length-dependent ambiguity) adds a small
but genuine insight about the limits of surface-level text routing.

## New Issues Found in Revision

1. **(Non-blocking)** MATH.md Theorem C.2 (line 79-80) still references `"sort:"` and
   `"reverse:"` as example n-grams. Cosmetic residual from pre-revision text.

2. **(Non-blocking)** MATH.md Lemma C.4 line 96 claims arithmetic uses `{+, -, =}` but
   the data generator only produces `+` and `=`, never `-`.

3. **(Non-blocking)** Lemma C.4 line 98-99 describes parity as "binary tokens followed by
   the label token" but omits the `>` separator that parity shares with sort/reverse. The
   separation argument is unaffected (parity has unique `even`/`odd` tokens) but the data
   format description is incomplete.

4. **(Non-blocking)** MATH.md line 149 says "n-grams (1,2)" but code uses ngram_range=(1,3).
   Proof is conservative relative to experiment.

5. **(Non-blocking)** run_experiment.py docstring line 17 still references `"sort:"`.

None of these are blocking. They are cosmetic inaccuracies that do not affect any theorem,
derivation, or kill criterion evaluation.

## Macro-Scale Risks (advisory)

1. **TF-IDF routing fails on semantically overlapping domains.** The toy domains are
   separable by surface-level format tokens and character ordering. Real domains (legal,
   medical, code) share vocabulary extensively. Character-level TF-IDF will not suffice;
   semantic routing (embeddings, classifiers on sentence representations) will be needed.
   This is acknowledged implicitly by the experiment's scope.

2. **Hard routing (argmax) discards uncertainty.** When domains overlap at scale, soft
   routing (weighted adapter combination) is more robust. Current architecture selects one
   adapter entirely.

3. **Sort/reverse confusion is a prototype of real-world failure.** The 11-14% confusion
   between structurally similar domains is the most informative signal. At macro scale,
   many domain pairs will be this similar or worse.

4. **Parity domain reveals adapter seed-specificity.** M2P loss on parity (2.68) is 4.6x
   worse than SFT (0.58), despite the base model already handling parity well (0.59). This
   suggests M2P adapters may not transfer across base model random seeds, which has
   implications for the broader pipeline.

## Status Assessment

The finding status of "supported" is appropriate. Rationale:

- Theorem 1 is a tautology (verified by construction, not by measurement).
- Theorem 2's predictions match measurements (95% routing, 11-14% sort/reverse confusion).
- The proof mechanism for sort/reverse is now correct but remains a "proof sketch" -- it
  argues from Fisher ratio being non-zero but does not compute the actual Fisher ratio or
  provide a tight bound on expected accuracy.
- Quality measurements lack exact CIs (acknowledged, with honest margin analysis).

"Supported" correctly captures: proof is directionally verified, predictions match, but the
proof sketch is not tight enough for "conclusive." Upgrading to "conclusive" would require:
(a) computing the actual Fisher ratio for sort/reverse character-ordering bigrams, and
(b) exact CIs on quality ratios from per-sequence loss data.

## Verdict

**PROCEED**

All three blocking issues from the first review have been fixed. The core proof mechanism
is now correct: Lemma C.4 argues from character-ordering statistics (not nonexistent format
prefixes), Theorem 2's QED is appropriately qualified, and Wilson CIs are correctly computed.

Five non-blocking cosmetic issues were identified (residual "sort:" reference in Theorem C.2,
missing `-` in arithmetic, omitted `>` in parity description, n-gram range mismatch, docstring
residual). None affect any theorem, derivation, or kill criterion.

The experiment successfully demonstrates that text-based TF-IDF routing eliminates the
covariate-shift failure mode in M2P composition, achieving 95.0% routing accuracy vs the
previous 36.6% MLP baseline. Finding status "supported" is appropriate.
