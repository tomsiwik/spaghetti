# Peer Review: bitnet_cosine_convergence

## NotebookLM Findings

NotebookLM consultation was not performed for this review due to the experiment already being marked "proven." This review proceeds directly with systematic attack.

## Mathematical Soundness

### Random baseline calculation -- CORRECT

MATH.md Section 3 computes E[|cos|] = sqrt(2/(pi*D)) for D ~ 20.5M parameters. This gives ~1.76e-4. The measured value of 0.001 at 400 steps is 5.7x this baseline. The derivation assumes i.i.d. Gaussian entries, which trained LoRA parameters are not, but the formula is correctly stated as a baseline comparison, not a prediction.

### Convergence detection -- CORRECT but WEAK

The sliding window method (Section 5) compares mean loss over two consecutive windows of 200 steps. The 1% improvement threshold is reasonable. However, the window is applied to *training* loss, not validation loss. MATH.md Assumption 4 argues this is conservative because overfitting would increase cosine. This argument is valid: if adapters overfit to noise in similar ways, cosine would rise, so using training loss convergence is indeed conservative for the cosine claim.

### K2 criterion -- CORRECTLY SPECIFIED but GENEROUS

K2 requires BOTH monotonic fraction > 0.80 AND second-half CV > 0.30 for a KILL. This is a conjunction, meaning the criterion can only kill if the trajectory is both monotonically rising AND has not plateaued. The measured monotonic fraction is 0.579 (below 0.80), and the CV is 0.093 (below 0.30). Both conditions independently pass. This is fine.

### The "114x below FP16" comparison -- MISLEADING but ACKNOWLEDGED

The 114x claim (0.00125 vs 0.142) compares across different models (BitNet-2B vs Qwen-7B), different dimensions (d=2560 vs d=896 or d=4096), different datasets, different training durations, and different hyperparameters. PAPER.md Limitation 5 acknowledges this: "The 0.142 comparison comes from a different model, different domains, and different training setup." The paper does not claim this is a controlled comparison, so this is not an error -- but the 114x figure is prominently featured and should be further caveated. The correct comparison would require training FP16 LoRA adapters on an FP16 model of similar size with identical data, domains, rank, and training steps.

### Cosine over LoRA A,B parameters vs effective delta W=BA -- NOTED LIMITATION

MATH.md Assumption 1 acknowledges that cosine is computed over raw (A,B) concatenations, not over the rank-r effective delta matrices B*A. This is a meaningful distinction. Two adapters could have orthogonal (A,B) vectors but produce correlated effective weight perturbations in the directions that matter for the model's output. Conversely, correlated (A,B) vectors could produce orthogonal effective perturbations. The paper's consistency argument ("all prior SOLE experiments use this") is valid for internal comparisons but does not address whether the metric actually measures what matters for composition. The "Pause Recycling LoRAs" reference in Section 3 of PAPER.md already flags that "orthogonality alone is insufficient for semantic composability."

**Verdict on math: Sound within stated assumptions.** No errors found. The main weakness is interpretive, not mathematical.

## Novelty Assessment

### Prior art

The experiment cites relevant work:
- LoRA (Hu et al., 2021)
- "LoRA vs Full Fine-tuning" (2410.21228) on intruder dimensions
- "Subspace Geometry Governs Catastrophic Forgetting" (2603.02224) on principal angles
- OPLoRA (2510.13003) on orthogonal projection
- "Pause Recycling LoRAs" (2506.13479) on orthogonality insufficiency

No published work was found that specifically tracks pairwise cosine trajectories of LoRA adapters on ternary bases over training steps. The experiment is novel in testing a specific micro-hypothesis (under-training artifact) rather than proposing a new method.

### Delta over prior SOLE work

The prior experiment (bitnet_2b_real_composition) measured |cos|=0.001 at 400 steps. This experiment extends to 2000 steps and shows the value plateaus at 0.00125. The informational delta is moderate: it rules out a specific adversarial hypothesis (under-training) but does not reveal a new mechanism.

## Experimental Design

### Strengths

1. **Well-defined kill criteria.** K1 and K2 are specific, falsifiable, and pre-registered in HYPOTHESES.yml.
2. **Dense trajectory.** 20 cosine checkpoints at 100-step intervals provide good temporal resolution.
3. **Composition PPL tracked in parallel.** This is the right secondary metric -- it verifies that low cosine corresponds to maintained composability.
4. **Data diversity.** 5 genuinely distinct domains (medical, code, math, legal, creative) from different HuggingFace datasets.

### Weaknesses

**W1: Sequential training on shared model instance -- potential memory contamination.**
All 5 adapters are trained sequentially on the same model object. Between domains, `zero_lora_params` reinitializes lora_a (random) and zeros lora_b. However, the optimizer state is recreated each time (`optimizer = opt.Adam(...)` at line 488), so there is no optimizer state leakage. The base model weights are frozen. This appears clean -- no contamination path exists.

**W2: Data cycling induces memorization, not generalization.**
With 500-800 training samples and 2000 steps at batch_size=1, each sample is seen 2.5-4x. Medical's loss drops from 2.81 to 0.50 (82% reduction), which for a 2B model on 800 short flashcard snippets strongly suggests memorization. The "convergence" detected may be memorization convergence, not learning convergence. However, for the cosine claim this is actually a *harder* test: memorized adapters that overfit to domain-specific patterns should be MORE orthogonal (encoding domain-specific noise), not less. So this concern does not undermine the result.

**W3: LORA_SCALE = 20.0 -- unusually high.**
The LoRA scale factor is set to 20.0 (line 54), meaning the effective adapter contribution is `20 * B * A`. Standard LoRA uses alpha/r where alpha is typically equal to r (scale=1.0) or 2*r (scale=2.0). A scale of 20.0 means the adapter perturbation is 20x amplified relative to the base. This could either help (stronger signal) or distort (overwhelming the base). This is consistent with prior experiments in the project, so it does not invalidate internal comparisons, but it is non-standard and should be noted when comparing to published LoRA results.

**W4: No FP16 control run -- the central claim is comparative but lacks a paired control.**
The entire motivation is "BitNet stays orthogonal but FP16 does not." The 0.142 FP16 number comes from a completely different experiment (Qwen2.5-7B, different data, different training). A rigorous test would train FP16 LoRA adapters on an FP16 model of comparable size (~2B params) with the same 5 domains, same data, same hyperparameters, same 2000 steps, and compare trajectories. Without this, the experiment proves that BitNet |cos| plateaus low, but does NOT prove that BitNet is better than FP16 for orthogonality at convergence. The paper's causal claims about "ternary base" being the mechanism (Section "Why BitNet Stays Orthogonal But FP16 Does Not") are unsupported hypothesis, not experimental finding.

This is the most significant gap. The experiment answers "does BitNet |cos| inflate?" (no) but does not answer "is ternary the reason?" (untested).

**W5: Single seed.**
Justified by reference to bitnet_multiseed_validation (CV=0.5%), which is reasonable for an internal consistency check. However, that validation was at 200 steps, not 2000 steps. Variance properties could differ at longer training horizons. This is a minor concern given the 40x margin to the kill threshold.

### Does the experiment test the stated hypothesis?

The stated hypothesis: "LoRA adapter pairwise cosine similarity on BitNet-2B-4T remains below 0.05 at full training convergence."

**Yes, this is directly tested and confirmed.** The trajectory shows |cos| = 0.00125 at step 2000 with 4/5 domains converged. The 40x margin to the kill threshold is decisive.

The unstated but implied hypothesis -- "this is because the base is ternary" -- is NOT tested.

### Could a simpler mechanism explain the result?

Yes. High dimensionality alone (D ~ 20.5M) predicts near-orthogonality for any set of 5 vectors, regardless of the base model type. The random baseline is 1.76e-4, and the measured value is only 5.7x this. Even for FP16 bases at similar dimensions, you would expect low cosine simply from the concentration of measure in high-dimensional spaces. The Qwen FP16 comparison at cos=0.142 may reflect a different dimensionality (Qwen-7B d=4096 but measured at d=896 from the macro experiment), different rank, or different training regime -- not necessarily the ternary vs FP16 distinction.

The paper partially acknowledges this by computing the random baseline, but Section "Why BitNet Stays Orthogonal But FP16 Does Not" presents three hypothesized mechanisms without testing any of them.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_cosine_convergence_trajectory` has:
- Kill criteria: (1) mean |cos| at convergence exceeds 0.05, (2) cos trajectory shows monotonic increase with no plateau
- Status: proven
- Evidence: correctly references results.json with accurate numbers

The kill criteria match what was tested. The status "proven" is appropriate for K1 and K2 both passing with large margins.

## Integration Risk

Low. This experiment does not introduce new mechanisms -- it validates an existing property (low cosine on BitNet-2B) holds under extended training. It composes cleanly with the existing architecture in VISION.md.

## Macro-Scale Risks (advisory)

1. **Longer training at macro scale.** 2000 steps with 500-800 samples is a toy regime. Production fine-tuning might run 10,000+ steps on 100K+ samples. The plateau observed here could be a local plateau that breaks at higher data volumes.

2. **Seq_len=128 vs production 2048+.** Longer sequences change gradient dynamics. The per-token gradient contributions are averaged over more positions, potentially increasing shared structure across adapters.

3. **More adapters.** N=5 is trivial in high dimensions (D=20.5M). The interesting regime is N=100+ where the pigeonhole principle starts to matter. The N=25 scaling experiment (referenced in PAPER.md) provides some evidence here.

4. **The FP16 comparison gap.** Before any paper publication, a controlled FP16 comparison on a ~2B model with identical setup is essential. The current "114x" claim will not survive external peer review without it.

## Verdict

**PROCEED**

The experiment cleanly answers its stated question: |cos| on BitNet-2B-4T does NOT inflate to 0.05+ at convergence. The 40x margin is decisive and the methodology is sound. The kill criteria were pre-registered and both pass convincingly.

However, the paper's rhetoric overreaches in two ways that should be corrected in PAPER.md and FINDINGS.md (non-blocking):

1. The "114x below FP16" comparison is uncontrolled. It should be clearly labeled as a cross-model, cross-setup comparison, not as evidence that ternary bases cause orthogonality. The current Limitation 5 exists but the number appears prominently in the Summary without qualification.

2. The "Why BitNet Stays Orthogonal But FP16 Does Not" section presents untested hypotheses as if they are findings. These should be moved to a "Future Work" or "Hypothesized Mechanisms" section, or an FP16 control should be run.

Neither of these block the core result. The experiment proves what it set out to prove: the low |cos| is not an under-training artifact.
