# Peer Review: Orthogonality by Domain Type

## NotebookLM Findings

Skipped -- NotebookLM not available in this session. Review conducted through direct code and document analysis.

## Mathematical Soundness

### What holds

1. **Delta vector construction (MATH.md Section 3)** is correct. The flattened concatenation of A@B products across layers gives D = L * 2 * d * d_ff = 131,072. Verified in code (lines 169-175).

2. **Random baseline (Section 5.1)** is correct: E[|cos|] = sqrt(2/(pi*D)) = 0.00220 for D=131,072.

3. **Pair counting (Section 6.1)** is correct: C(15,2) = 105 total, 3*C(5,2) = 30 within, 75 cross.

4. **Permutation test implementation (lines 387-410)** is methodologically sound. The test statistic (difference in means) is appropriate. 10,000 permutations gives adequate resolution for p < 0.001.

5. **Cohen's d calculation (lines 596-598)** uses pooled variance correctly, though it pools across seeds (treating 30*3=90 within-cluster pairs and 75*3=225 cross-cluster pairs as independent observations). This is an approximation -- pairs from the same seed share a base model and are not fully independent -- but the effect size is so large (2.24) that accounting for within-seed correlation would not change the qualitative conclusion.

### What is questionable

6. **Gradient approximation (code line 244-246).** The backward pass does NOT propagate gradients from fc1 back through to h_in (the input to the layer). The comment says "skip the fc1 backprop contribution to d_h for speed." This means the gradient for each layer's B1 only depends on the loss gradient flowing down from higher layers, not on how fc1 affects deeper layers' inputs. For a 4-layer MLP with residual connections, this truncation means layers 0-2 receive incomplete gradients. The paper's Assumption 3 (Section 7) claims this is "standard for LoRA training since the base model is frozen," but that justification is wrong -- standard LoRA training uses full backpropagation through the frozen base; only the gradient update is restricted to LoRA parameters. The truncation is a simplification for the numpy implementation, not a standard practice. This likely weakens the gradient signal (especially for early layers), making the deltas noisier than they would be with correct backprop. This means the observed effect is likely a lower bound -- correct gradients would produce stronger cluster structure. So the truncation is conservative, not invalidating.

7. **The claim "loss barely decreases" deserves scrutiny.** Loss stays at ~3.466 throughout training (PAPER.md, Limitation 3). With V=32 and uniform prediction, random loss = -log(1/32) = 3.466. This means the model learns essentially nothing. The LoRA deltas are dominated by early gradient direction, not converged features. The paper frames this as "conservative" but this is a double-edged sword: it means the experiment tests whether raw gradient direction (not learned representations) correlates with data similarity. This is a weaker and more obvious claim than "LoRA expert interference is predictable."

8. **Shared A matrices vs independent A matrices.** Each expert gets its OWN random A matrices (init_lora is called once per expert, line 459-461, using the shared rng). In standard LoRA practice on real models, each adapter would indeed get independent A initialization. However, the delta A_i @ B_i lives in the column space of A_i, which is a random r-dimensional subspace. Two experts with different A matrices have deltas in DIFFERENT random subspaces, which should make them highly orthogonal regardless of B similarity. The fact that within-cluster cosine is still 0.06 (27x above random) despite different A matrices is either (a) a genuine signal from data similarity overcoming random subspace separation, or (b) an artifact of other confounds. I believe it is mostly (a) but the effect size may be inflated by confound (9) below.

### Hidden assumption not stated

9. **Sequential RNG coupling.** A single `rng` object is used for: building the base model, generating prototypes, generating data for all 15 experts in order, and initializing/training all 15 LoRA adapters in order (lines 438-464). The training order is always code[0..4], reasoning[5..9], knowledge[10..14]. This means within-cluster experts are trained from closer RNG states than cross-cluster experts. While a well-implemented PRNG (Mersenne Twister in numpy) should produce effectively independent draws regardless of proximity in the state sequence, this is still a methodological weakness. The correct design would use independent RNG streams per expert (e.g., `np.random.RandomState(seed + expert_id)`) to guarantee statistical independence. This is unlikely to explain the 7.84x ratio given PRNG quality, but it should be fixed.

## Novelty Assessment

### Prior art

The core observation -- that LoRA adapters trained on similar tasks produce similar weight deltas -- is implicit in several bodies of work:

- **LoRAHub (Huang et al. 2023)**: Uses gradient-free optimization to compose LoRA adapters for new tasks. The premise is that task similarity in parameter space enables transfer, which presupposes the kind of structure this experiment measures.

- **TIES-Merging (Yadav et al. 2023)**: The sign conflict resolution mechanism implicitly assumes that related tasks produce correlated deltas (otherwise sign conflicts would be uniformly distributed). This experiment provides direct evidence for that assumption.

- **InfLoRA (Liang & Li 2024)**: Enforces orthogonality via explicit constraints. This experiment shows that without such constraints, natural orthogonality exists but is weaker within semantic clusters -- providing empirical motivation for InfLoRA's approach.

- **Task Arithmetic (Ilharco et al. 2023)**: The entire task arithmetic framework assumes that task vectors (fine-tuning deltas) encode task-specific directions in weight space. Cluster structure in cosine similarity is a direct consequence.

### Delta over prior work

The specific contribution is quantifying the within-cluster vs cross-cluster ratio (7.84x) and showing it is statistically robust. Prior work assumed or indirectly relied on this structure; this experiment measures it directly. However, the measurement is at micro scale with synthetic data, so the quantitative finding (7.84x) cannot be cited as a real-world result. The directional finding (within > cross) is well-supported but arguably expected from first principles.

**Novelty verdict:** The result is confirmatory rather than surprising. It provides useful quantitative evidence for a qualitatively expected phenomenon. Moderate novelty.

## Experimental Design

### Does it test the hypothesis?

Yes. The hypothesis is "within-cluster |cos| > cross-cluster |cos|" and the experiment directly measures this with appropriate controls (multiple seeds, permutation test, effect size).

### Confounds and alternative explanations

1. **Trivial gradient similarity.** The model barely trains (loss stays at random baseline). The LoRA deltas are essentially the sum of 300 mini-batch gradients. For domains with similar data distributions (same cluster), the expected gradient direction is similar because the data statistics are similar. This is not "LoRA interference structure" -- it is "similar data produces similar gradients," which is a tautology given how the clusters are constructed. The experiment constructs clusters by perturbing a shared prototype transition matrix, then shows that training on similarly-perturbed data produces similar deltas. The causal chain is direct and almost definitional.

2. **Could a simpler mechanism explain the result?** Yes. Compute the mean embedding vector for each domain's training data. Within-cluster domains have more similar mean embeddings (by construction). The first gradient step for B1[0] is proportional to `A1[0].T @ mean_embedding.T @ d_z1`. Even with different A matrices, if mean_embedding and d_z1 are correlated across within-cluster domains, the B updates will be correlated. A control experiment computing cosine similarity of mean data embeddings (no LoRA training at all) would test whether the cluster structure is already present in the raw data statistics.

3. **Missing control: data-only baseline.** The experiment should include a control measuring the cosine similarity of raw data statistics (e.g., mean token frequency vectors) across domains. If within-cluster data statistics are already 7x more similar than cross-cluster, then the LoRA training adds no information -- it just reflects what was already in the data. This control is absent.

### Kill criteria alignment

The kill criteria in HYPOTHESES.yml match what is tested:
- K1: "within-cluster cosine is NOT higher than cross-cluster" -- tested, passes
- K2: "no predictable pattern in which domains collide" -- tested via permutation test, passes

The criteria are appropriate and the experimental evidence addresses them.

## Macro-Scale Risks (advisory)

1. **Effect magnitude at high dimensions.** At d=3584, D_FFN ~ 3.8 billion. Both within-cluster and cross-cluster cosines will be extremely small. The ratio may persist but the absolute signal could be below measurement noise when using finite-precision floating point.

2. **Real domain similarity is not Markov chain similarity.** Python vs JavaScript share syntax structures, library patterns, and reasoning approaches. Medical vs law share analytical reasoning. These rich similarities may not produce the clean block-diagonal structure seen with Markov chains. The structure could be more graded (continuous similarity spectrum rather than discrete clusters).

3. **The practical implication ("avoid activating same-cluster experts") may not matter.** If cross-cluster cosine at macro scale is already ~10^-5 and within-cluster is ~10^-4, both are negligible for composition quality. The interference term in task arithmetic scales as cos/N^2, so even the "high" within-cluster interference vanishes at moderate N.

4. **The math-medical extrapolation (PAPER.md Section 5.4) is a stretch.** The prior math-medical cosine of 0.59 (at macro scale, real data, FFN-only) is 10x higher than the within-cluster cosine here (0.06). The paper claims to "explain" the math-medical outlier as a within-cluster effect, but the magnitudes differ by an order of magnitude and the experimental conditions are completely different (real vs synthetic data, transformer vs MLP, macro vs micro scale).

## Verdict

**PROCEED**

The experiment achieves what it set out to do: demonstrate that domain similarity creates predictable structure in LoRA cosine similarity. Both kill criteria are disproven with strong statistical evidence. The mathematical framework is sound with minor issues (gradient truncation is conservative, not invalidating).

**Caveats that should be recorded but do not block PROCEED:**

1. The result is arguably expected from first principles (similar data -> similar gradients -> similar deltas). The contribution is quantification, not discovery.

2. The missing data-only baseline control means we cannot distinguish "LoRA training reveals cluster structure" from "LoRA training passively reflects data statistics." This distinction matters for the practical routing implication.

3. The sequential RNG coupling is a methodological weakness. While unlikely to explain the 7.84x ratio, it should be noted and fixed in any follow-up.

4. The math-medical extrapolation in the paper overreaches. The micro-scale within-cluster cosine of 0.06 does not explain a macro-scale within-cluster cosine of 0.59. These are qualitatively consistent but quantitatively unrelated.

5. The model does not learn (loss = random baseline), so the deltas test gradient direction rather than learned feature similarity. The paper acknowledges this as "conservative" which is a fair interpretation, but the claim should be stated precisely: "gradient directions from similar distributions are more aligned" not "LoRA experts trained on similar domains interfere more."
