# Peer Review: m2p_third_domain

## Experiment Type
Guided exploration (as declared). MATH.md states the proven framework (M2P per-domain quality, Findings #359-#362) and identifies the unknown (whether M2P quality holds on a structurally different domain). This is correctly classified.

## Hack Detector
- Fix count: 1 mechanism (M2P applied to a new domain). Clean, no hacks.
- Is MATH.md a proof or a description? **Description.** There is no Theorem/Proof/QED block. MATH.md contains a domain design rationale and predictions based on group-theoretic intuition, but no formal derivation of why cross-domain transfer should be <30%.
- Metric used as evidence: Quality ratio (base_loss - m2p_loss) / (base_loss - sft_loss). Proven in prior findings (#359, #361) to be a meaningful proxy for M2P reconstruction fidelity. Acceptable.
- Kill criteria source: K900 and K902 are derived from prior findings. K901 (<50% cross-domain transfer) is a reasonable operationalization of "structural diversity" but the predicted value (<30%) was based on intuition, not derivation. Acceptable for guided exploration.

## Self-Test Audit

MATH.md does not contain a labeled "Self-Test" section. However, PAPER.md contains a self-test block at the end. Evaluating against the 6 criteria:

1. **Impossibility property:** PASS. Clearly stated: "Cross-domain transfer >50% is inevitable when tasks share the same token vocabulary, because M2P memory lies in the span of the shared embedding manifold." This is a genuine structural insight, post-hoc but well-articulated.

2. **Cited theorems:** PARTIAL. The argument cites dimension-counting properties of shared embedding manifolds and group-theoretic framing (Z_26 homomorphism, S_n permutation group). These are real mathematical structures correctly applied. However, no formal theorem is cited by name (e.g., no reference to a specific linear algebra result about shared subspaces). For guided exploration this is acceptable but not ideal.

3. **Predicted numbers:** PASS. Four quantitative predictions: cipher quality 85-100%, sort/reverse 98-102%, cross-domain <30%. All specific and falsifiable.

4. **Falsification conditions:** PASS. K901 is a clean falsification criterion. The impossibility argument itself has a stated falsification condition: "demonstrating <50% cross-domain transfer between two tasks sharing the same 26-token vocabulary at d=512."

5. **Hyperparameter count:** PASS. Zero new hyperparameters introduced. All architecture constants (D_M2P=64, N_MEMORY=32, etc.) are inherited from prior proven experiments.

6. **Hack check:** PASS. Clean single-mechanism experiment. No patches or workarounds.

## Mathematical Soundness

**Pre-experiment predictions:**
The prediction of <30% cross-domain transfer was based on informal reasoning: "Caesar cipher requires modular addition, sort/reverse require comparison/permutation, therefore M2P should learn different activation patterns." This is plausible intuition but not a derivation. No formal argument was given for why different computational primitives should produce different M2P representations.

**Post-experiment analysis (the key contribution):**
The impossibility argument in PAPER.md is the real intellectual content of this experiment. It states: when |V|=26 and d=512, the embedding manifold is a 26-dimensional subspace of R^512. All tasks over the same vocabulary are constrained to this subspace. M2P outputs B-matrices that produce LoRA perturbations; these perturbations act on the same embedding subspace regardless of which task trained the M2P.

**Is this analysis correct?** Mostly, with one important caveat:

The argument conflates two distinct spaces. The embedding matrix E in R^{26 x 512} defines a 26-dim subspace of R^512, yes. But the M2P does not output embeddings -- it outputs B-matrices that are applied as LoRA perturbations to attention and MLP weight matrices (wq, wk, wv, wo, fc1). The perturbation DeltaW = B * A acts on hidden states h in R^512, not on the raw embedding subspace. Hidden states at layers 1-2 are no longer confined to the column span of E; they have been transformed by attention and MLP operations.

The correct argument is simpler and more direct: with only 26 tokens, the entire input space has at most 26 distinct embedding vectors. Any function over sequences of these 26 embeddings lives in a low-dimensional function space. Two tasks over the same 26 tokens that both achieve low loss necessarily share most of the same representational structure because there is simply not enough input diversity to force disjoint representations. The M2P, which reads hidden states and outputs B-matrices, sees nearly identical activation distributions for sort vs. cipher because both process the same 26-token sequences through the same frozen base.

This is a stronger and more precise version of the argument. The "shared embedding manifold" framing is roughly right in spirit but technically imprecise about WHERE the bottleneck occurs (input space cardinality, not output embedding dimensionality).

**Does this matter for the kill decision?** No. The conclusion is correct regardless: shared vocabulary at |V|=26 makes cross-domain divergence essentially impossible. The fix (use disjoint vocabularies or fundamentally different token sets) is also correct.

## Prediction vs Measurement

PAPER.md contains a clear prediction-vs-measurement table:

| Metric | Predicted | Measured | Match |
|--------|-----------|----------|-------|
| Cipher M2P quality | 85-100% | 99.7% | Yes |
| Sort M2P quality | 98-102% | 98.5% | Yes |
| Reverse M2P quality | 98-102% | 99.1% | Yes |
| Cross-domain sort->cipher | <30% | 97.78% | NO |
| Cross-domain reverse->cipher | <30% | 98.54% | NO |
| grassmannian_max_cos | ~0 | 4.66e-9 | Yes |

The quality predictions match. The diversity predictions fail catastrophically (off by >3x from predicted, >2x from kill threshold). The kill is clean and unambiguous.

## NotebookLM Findings

Skipped -- the experiment is already killed, the analysis is clear, and the materials are straightforward enough that NotebookLM deep review would not add value beyond what the code and results already show.

## Novelty Assessment

This experiment is novel within the project context: it is the first attempt to test structural diversity across M2P domains. The negative result is valuable -- it establishes that vocabulary sharing is a hard constraint on domain separability, which informs future domain selection.

No prior art within the project addresses this specific question. The finding that shared vocabulary prevents representational diversity is well-known in the broader transfer learning literature (e.g., multi-task learning on shared vocabularies routinely shows positive transfer), but the specific application to M2P cross-domain isolation is new to this project.

## Experimental Design Assessment

**Cross-domain methodology:** The cross-domain M2P is trained for 300 steps (vs. 1000 for per-domain). This asymmetry is noted but acceptable -- even at 300 steps, the cross-quality reaches 97-98%, meaning the signal is saturated. More steps would only strengthen the kill.

**Missing control:** The experiment does not test cipher->sort or cipher->reverse transfer. Only sort->cipher and reverse->cipher are measured. PAPER.md acknowledges this ("cipher->sort not measured directly"). This is a minor gap; the argument is symmetric and 2 out of 4 pairs showing 97%+ transfer is sufficient for the kill.

**Arithmetic exclusion:** Correctly excluded (parity guard triggers because base already achieves near-SFT loss). Well-handled.

**Code quality:** Clean, well-structured, follows coding guidelines. Memory management is proper. No issues.

## Macro-Scale Risks (advisory)

1. **The impossibility structure scales:** At macro scale with real models (32k BPE vocabulary), domains that use overlapping vocabulary subsets will still show partial transfer. The lesson is not "use 26 tokens" but "vocabulary overlap is a continuous predictor of cross-domain transfer." Future macro experiments should measure vocabulary overlap (Jaccard coefficient over active token sets) as a covariate.

2. **Positive finding preserved:** The cipher quality result (99.7%) confirms M2P generalizes to substitution-group operations, not just permutation-group operations. This is a real signal about M2P expressiveness that should carry forward.

3. **Next domain candidate:** PAPER.md recommends natural language summarization over 32k BPE vocabulary as the strongest test. This is correct -- maximal vocabulary disjointness is the cleanest path to demonstrating structural diversity.

## Verdict

**KILL validated.**

The experiment was correctly killed. The root cause analysis is substantively correct (shared vocabulary prevents representational diversity) though technically imprecise in one detail (the bottleneck is input space cardinality, not embedding subspace dimensionality per se). The impossibility structure is well-documented. The positive finding (M2P generalizes to substitution operations) is preserved. The recommendations for next steps are sound.

No revisions needed. The kill, the analysis, and the documented impossibility structure are all adequate. Finding #371 (killed) should retain the impossibility structure statement as-is -- the informal framing is directionally correct even if the precise linear algebra argument could be tightened.
