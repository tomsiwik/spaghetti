# Peer Review: Splay-Tree Adaptive Routing

## NotebookLM Findings

Skipped -- the researcher already killed the experiment with clear evidence. A deep NotebookLM review would not change the verdict. The review below covers all necessary dimensions.

## Mathematical Soundness

The math is correct and clearly presented. Specific verification:

1. **Log-odds bias formulation (Section 1.4)**: Adding `alpha * log(F_left / F_right)` to the logit before sigmoid is the correct way to multiplicatively reweight the output probability. For sigmoid, `sigma(z + log(p/q)) = sigma(z) * (p/q) / (1 + sigma(z) * (p/q - 1))` -- not exactly multiplicative as MATH.md claims ("approximately multiplies the output probability by p/q"), but the direction is correct and the approximation is reasonable for small corrections. The "approximately" qualifier saves this.

2. **EMA convergence (Section 5.1)**: Half-life calculation `log(0.5) / log(0.95) = 13.5 steps` is correct.

3. **Worked example (Section 6)**: Verified the numerical calculations. `beta_0 = log(0.520/0.480) = 0.0800` checks out. The convergence example at step 10 is also correct: `log(0.7/0.3) = 0.847`.

4. **Subtree precomputation (code)**: The `_precompute_subtrees` method correctly traces each leaf's path through the binary tree using bit decomposition. The left child = `2*node+1`, right = `2*node+2` indexing is standard for array-based binary trees.

5. **One concern -- gradient interaction not analyzed**: MATH.md Section 7.4 correctly identifies that splay biases change the loss landscape, potentially conflicting with gradient updates. This is acknowledged as an assumption but never analyzed. The splay bias is applied inside the forward pass (line 52 of splay_routing.py: `logit = self.proj(x) + self._splay_bias`), which means gradients flow through the sigmoid with the bias included. The gradient `dL/dw_i` is computed at a different operating point than without splay. This is not a bug per se, but the MATH.md claim that "the two channels operate at different timescales" (Section 5.2) understates the coupling. They operate on the SAME sigmoid output simultaneously.

6. **Leaf frequency update timing**: The `_update_leaf_frequencies` call happens AFTER the forward pass (line 214), using probabilities from the current forward pass. This means the splay bias applied during forward pass N reflects frequencies from steps 1 through N-1, not N. This is the correct design (no future information leakage), and matches the MATH.md description.

**Verdict: Math is sound. No errors found. One acknowledged-but-underanalyzed interaction (splay-gradient coupling).**

## Novelty Assessment

The splay-tree-to-MoE analogy is genuinely novel. I found no prior art applying splay tree frequency-biasing to MoE gating:

- **Sleator & Tarjan (1985)** is classical splay trees with structural rotations -- the soft bias adaptation is a novel reframing.
- **Jordan & Jacobs (1994)** HME has no adaptive bias mechanism.
- **Fast Feedforward Networks (Belcak & Wattenhofer, 2024)** use binary tree routing but with no runtime adaptation.
- **FINDINGS.md** mentions a previous "Phase 3: Splay cache" that had "zero effect on loss" -- but that was a different mechanism (caching, not routing bias). The current experiment is a clean rethinking.

The analogy is well-drawn in the MATH.md comparison table (Section 2.2). The key insight -- that soft bias achieves the working-set property without topology changes -- is a reasonable contribution, even though the mechanism failed empirically at this scale.

**Delta over closest work: Genuine. No prior art found for frequency-adaptive bias correction on hierarchical MoE gates.**

## Experimental Design

The experimental design is appropriate for the hypothesis being tested:

**Strengths:**
- 3-seed main experiment with clear aggregation
- Domain shift protocol (train A, switch to B) directly tests the adaptation hypothesis
- Static tree as the correct control (identical architecture minus splay)
- Alpha sweep explores the mechanism's sensitivity
- Multiple metrics: val_loss, routing entropy, wall-clock time, early convergence

**Weaknesses (fair critiques within micro constraints):**
1. **The "domain switch reset" is an oracle signal.** Line 184: `model.on_domain_switch("n_z")` explicitly resets splay state. In production, you don't know when domain shifts happen. The paper acknowledges this in Limitations (point 5) but the main experiment uses the oracle reset. This means the experiment tests the BEST CASE for splay -- and it still loses. This actually strengthens the kill verdict.

2. **Alpha sweep is single-seed (seed=42).** The paper correctly flags this as unreliable. The best alpha (0.5, val 0.4970) vs baseline alpha (0.0, val 0.5061) is a 1.8% difference on one seed. Not meaningful.

3. **Both models continue gradient training on domain B.** This means the comparison is "gradient + splay" vs "gradient alone", not "splay alone" vs "gradient alone". The splay mechanism never gets tested in isolation. A frozen-gates experiment (train on A, freeze gate weights, then evaluate on B with only splay adaptation vs no adaptation) would have cleanly isolated the splay contribution. However, this missing control doesn't matter for the kill verdict -- the hypothesis was that splay would HELP on top of gradient adaptation, and it didn't.

4. **KC1 tolerance of 0.5%** (line 275: `splay_b_mean <= static_b_mean * 1.005`) is reasonable but not pre-registered. The actual delta (+0.57%) would fail even a 1% tolerance, so this doesn't affect the verdict.

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node:
- **Kill criteria**: "splay restructuring does NOT reduce routing cost on non-stationary data" and "splay overhead exceeds routing savings" -- both tested, both triggered.
- **Dependency**: depends on `exp_hierarchical_capsule_tree` (proven) -- correct, the experiment builds on the proven parent.
- **Status**: `disproven` -- correct.
- **Blocks**: nothing -- correct, no downstream experiments depend on this.

The evidence is sufficient to change the node status to disproven.

## Macro-Scale Risks (advisory)

The paper's speculation about L=256+ trees is the most interesting open question, but several concerns apply:

1. **Gradient recalibration at scale**: The paper claims "at L=256 with 255 gates, the gradient adaptation would be much slower." This is not obviously true. Adam maintains per-parameter momentum, so even 255 gates each get individually adaptive learning rates. The gradient signal is local to each gate (binary left/right decision), not a global 255-way optimization. The recalibration could still be fast.

2. **EMA stability at scale**: With L=256 leaves, the EMA frequency vector has 256 entries, each initialized to 1/256 = 0.0039. Small batch sizes would produce noisy frequency estimates, and the EMA would need many more steps to converge. The half-life doesn't change (still 13.5 steps at gamma=0.95), but the signal-to-noise ratio per leaf worsens linearly with L.

3. **The splay-gradient conflict scales**: At larger models, optimization is more sensitive. A non-parametric bias that shifts the loss landscape could interfere with optimizer state (Adam momentum/variance buffers were trained expecting a different operating point). This was observed in FINDINGS.md for the ART-spawn mechanism: "merging experts disrupts Adam momentum/variance buffers."

## Verdict

**KILL -- confirmed.**

The researcher's self-kill is correct. Both kill criteria trigger with clear evidence:

- KC1: Splay domain-B val_loss +0.57% worse than static (3 seeds). Splay provides no adaptation advantage.
- KC2: Wall-clock overhead +51.5%. Even attributing this entirely to Python overhead, the mechanism provides negative quality benefit, so any overhead at all is unacceptable.

The experiment was well-designed and well-documented. The failure mode is clearly identified: gradient descent recalibrates 7 gates faster than EMA accumulates useful statistics. The "salvageable direction" (alpha=0.5 as soft regularizer) is interesting but is really just Huffman frequency shaping by another name, which is already explored in a separate experiment.

No revisions needed. The experiment accomplished its purpose: testing a novel mechanism, finding it dead at micro scale, and documenting why. The splay-tree MoE routing idea can be archived.
