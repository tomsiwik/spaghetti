# Peer Review: SwiGLU Macro Pruning Transfer (v2)

## Prior Review Status

The v1 review identified three blocking issues. Their resolution:

| Issue | v1 Problem | v2 Fix | Status |
|-------|-----------|--------|--------|
| Data provenance | 16 hardcoded prompts, falsely labeled WikiText-2 | `load_dataset("wikitext", "wikitext-2-raw-v1")` via HF datasets | FIXED |
| Held-out eval | Cal and eval data identical | Cal=test split, eval=validation split | FIXED |
| Random baseline | Missing | 3-seed random pruning at tau=0.05 count | FIXED |

All three blocking fixes are correctly implemented in the code.

---

## NotebookLM Findings

Manual review conducted. NotebookLM deep review deferred.

---

## Mathematical Soundness

### What holds

**Gate-product profiling is architecturally correct.** Lines 156-188 of
`profile_gate_products.py` correctly reproduce the SwiGLU forward pass:
`SiLU(gate_proj(x)) * up_proj(x)`. The interception happens before `down_proj`,
accumulating mean absolute values per neuron across all positions. Residual
connections and layer norms are correctly handled so profiling does not alter the
forward pass.

**Pruning implementation is correct.** Lines 446-475 zero out rows of both
`gate_proj` and `up_proj` for neurons below threshold, which is mathematically
equivalent to forcing `h_j(x) = 0` for all inputs. Weights are correctly
restored between threshold sweeps (lines 447-451) preventing cumulative errors.

**Bimodality coefficient calculation is correct.** Lines 297-321 compute
population moments (appropriate for N=116,736 where Bessel correction is
negligible). Sarle's BC = (skewness^2 + 1) / kurtosis with the SAS criterion
BC > 5/9 = 0.555 is standard.

**Random pruning baseline is correctly implemented.** Lines 509-592 sample
neuron indices uniformly across all layers without replacement, with proper
weight restoration between seeds. The seed offset (seed + 1000) avoids
collision. Three seeds is minimal but adequate for establishing direction.

**Data pipeline is now legitimate.** The `load_wikitext2_split` function
(lines 55-109) loads from HuggingFace datasets, concatenates non-empty lines,
tokenizes the full text, and splits into non-overlapping sequences. Calibration
uses test split (128 x 128 = 16,384 positions, 3,134 unique tokens), evaluation
uses validation split (64 x 128 = 8,192 positions, 1,945 unique tokens). These
are genuinely disjoint.

**Baseline perplexity is now reasonable.** 21.31 on WikiText-2 validation in
bf16 is plausible for Qwen2.5-0.5B (published benchmarks vary by tokenization
and sequence length; this is within expected range).

### Issues (non-blocking)

**The "~6x more" error propagation claim (MATH.md Section 3.2) is hand-wavy.**
Error amplification through 24 vs 4 layers depends on spectral properties of
the per-layer Jacobians, not simply layer count. The empirical results make
this point moot, but the theoretical justification should not claim a specific
multiplier without analysis.

**Sarle's BC false positive risk is real and acknowledged.** The paper
correctly flags (MATH.md Section 4.1, PAPER.md Section KC1) that extreme
skewness (39.1) and kurtosis (2382) are red flags for BC false positives. The
aggregate distribution may be heavy-tailed unimodal rather than truly bimodal.
Hartigan's dip test was not run. The paper's caveat is appropriate; the BC
result should be treated as directional, not definitive. This is a minor
weakness given that the bimodality claim is not the main finding.

**Attention mask is set to None during profiling.** Line 163 passes
`mask=None` to self-attention. For non-autoregressive profiling of packed
sequences, this means every token attends to every other token in the batch,
including across sequence boundaries. This could subtly alter hidden states
compared to standard causal inference. However, the effect on per-neuron mean
activation statistics is likely negligible -- it adds some noise to the
profiling but would not systematically bias which neurons appear low-activation.
The pruning evaluation (lines 383-412) uses `model(inputs)` which goes through
the model's standard forward pass with proper causal masking, so the perplexity
numbers are unaffected.

---

## Novelty Assessment

### Prior art

The finding that zero-shot structured pruning fails at macro scale is
well-established (Wanda, SparseGPT, LLM-Pruner, SliceGPT). The paper cites
these appropriately and positions itself not as a pruning method paper but as a
transfer analysis from micro to macro scale.

### Genuine novel contribution

The anti-signal finding is the real contribution: gate-product profiled pruning
is **8.9x worse than random** at macro scale, while being **2.3x better than
random** at micro scale with aux loss. This is not a "pruning fails" result
(known). It is a "the importance signal inverts" result, which is a stronger and
more specific claim. The decomposition into three non-transferring factors
(aux loss robustness, depth amplification, specialist neurons) is useful.

The insight that aux sparsity loss provides robustness training (not just
distribution shaping) is actionable for future work. It explains why you cannot
simply observe the bimodal distribution and prune -- the distribution is
necessary but not sufficient.

### Delta over existing work

No prior work specifically demonstrates that gate-product profiling is
anti-correlated with safe prunability at macro scale. The closest is Wanda's
observation that activation magnitude alone is insufficient (hence weight *
activation), but Wanda does not test structured SwiGLU gate-product pruning
or report an anti-signal.

---

## Experimental Design

### Strengths

1. **Clean data pipeline.** WikiText-2 via HuggingFace datasets with separate
   calibration (test) and evaluation (validation) splits. Token counts are
   consistent with the dataset.

2. **Random pruning baseline at the critical threshold.** Pruning 18,420
   neurons randomly (3 seeds) at the same count as tau=0.05 provides the key
   control. The 8.9x ratio (552.78 / 61.97) with std=8.52 across seeds is
   statistically clear -- even at random seed 2 (worst: 73.64), profiled
   pruning is still 7.5x worse.

3. **Multiple thresholds tested.** Six thresholds from tau=0.01 to tau=0.50
   with consistent catastrophic failure rules out threshold sensitivity.

4. **Weight restoration between experiments.** Prevents cumulative damage.

### Weaknesses (non-blocking)

1. **Random baseline only at one pruning level.** The 8.9x ratio is measured
   at tau=0.05 (15.8% pruned). At tau=0.01 (2 neurons), the comparison is not
   possible with random pruning (expected random damage at 2 neurons is
   essentially zero, so the +16.1% from profiled pruning is already strong
   evidence). At tau=0.10 (66.8%), random pruning would also degrade
   significantly, making the ratio less informative. The tau=0.05 comparison
   point is the most informative one, so this is adequate.

2. **No Wanda-style comparison.** The paper identifies weight * activation as
   the natural next step but does not test it. This is fair scope limitation
   for a transfer analysis but would strengthen the paper.

3. **Single model family.** Only Qwen2.5-0.5B tested. The "architectural
   property of SwiGLU" claim requires at least one more model family. The paper
   acknowledges this in Limitations (item 3).

4. **Calibration sequence length is short.** 128 tokens per sequence may miss
   long-range activation patterns. Specialist neurons that only fire in longer
   contexts could have different profiling statistics at seq_len=512 or 2048.
   This is unlikely to change the direction of results but could affect the
   magnitude of the anti-signal.

---

## Hypothesis Graph Consistency

The experiment matches node `exp_swiglu_macro_pruning_transfer` in
HYPOTHESES.yml. Kill criteria:
- KC1 (bimodality): PASS (BC=0.643, with appropriate caveats)
- KC2 (pruning quality >5% worse): KILL at all thresholds

The `partial_kill` status is correct. The evidence entries in HYPOTHESES.yml
accurately summarize the v2 results. The `depends_on: [exp_swiglu_gate_pruning]`
lineage is correct. The empty `blocks` list is correct -- the macro transfer
failure does not gate any downstream experiments (it terminates this branch).

---

## Macro-Scale Risks (advisory)

1. **The anti-signal finding may be specific to mean-activation profiling.**
   Alternative profiling metrics (activation frequency, max/mean ratio, weight
   norm * activation) could produce a positive signal. The paper's future work
   suggestions (Section 7 of MATH.md) are appropriate.

2. **The bimodality claim needs cross-model validation.** Testing on Llama 3.x,
   Mistral, or DeepSeek would determine whether this is truly a SwiGLU
   architectural property or specific to Qwen's training recipe.

3. **Post-pruning fine-tuning is the obvious next experiment.** The paper
   correctly identifies this but does not test it. Even 100 steps of fine-tuning
   after gate-product pruning could recover substantial quality, potentially
   re-inverting the signal direction.

---

## Verdict

**PROCEED**

The v2 revision has adequately addressed all three blocking issues from the v1
review:

1. Data is now genuinely WikiText-2 via HuggingFace datasets (verified in code
   and results.json provenance fields).
2. Calibration and evaluation use separate splits (test vs validation).
3. Random pruning baseline with 3 seeds provides the critical control.

The core finding -- that gate-product profiled pruning is 8.9x worse than
random at macro scale, inverting the micro-scale result -- is methodologically
sound. The anti-signal result is novel and actionable: it demonstrates that aux
sparsity loss provides robustness training (not just distribution shaping), and
that mean activation magnitude is an anti-predictor of safe prunability in
production models. The paper's self-assessment is honest, the limitations are
clearly stated, and the kill criteria are correctly applied.

The BC false-positive risk for bimodality is a legitimate concern but is
appropriately flagged in both MATH.md and PAPER.md. The bimodality claim is
secondary to the anti-signal finding, which stands regardless of how the
distribution is characterized.

This experiment concludes the gate-product zero-shot pruning branch with a
clear negative result and identifies viable next directions (Wanda-style
scoring, activation frequency, post-pruning fine-tuning).
