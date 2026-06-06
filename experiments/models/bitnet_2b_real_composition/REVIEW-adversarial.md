# Peer Review: BitNet-2B Real Composition

## NotebookLM Findings
Skipped -- this experiment has clear enough structure that manual review is sufficient. The math is straightforward (LoRA addition, cosine similarity, PPL ratios) and does not require deep study-guide generation.

## Mathematical Soundness

**Correct:**
- The LoRA composition formula (naive addition with 1/N scaling) is standard and correctly stated.
- Cosine similarity computation between flattened adapter vectors is correct.
- The composition ratio definition (avg composed / best individual) is internally consistent with the kill criterion.
- Memory analysis is reasonable (2 bits/weight for packed ternary, bfloat16 for unpacked).
- The unpacking logic in `unpack_ternary()` correctly handles the 2-bit encoding: each uint8 stores 4 ternary values as {0,1,2} mapped to {-1,0,1}.

**Issues:**

1. **Composition ratio denominator is misleading.** The ratio is computed as `avg_composed / best_individual` (line 627 of code, confirmed in results.json). The "best individual" is python at PPL 2.22. Dividing the average across 5 domains by the single best domain's PPL inflates the ratio. A fairer metric would be `avg_composed / avg_individual` = 7.96 / 6.40 = 1.24x, or per-domain ratios. The 3.59x number is dominated by the legal domain dragging up the average while python pulls down the denominator. This is not mathematically wrong (it matches the kill criterion definition), but it is a poor summary statistic. **Severity: low** -- the kill criterion is stated as-is and was passed either way.

2. **d_eff calculation needs scrutiny.** MATH.md claims d_eff = 21,626,880 for the expected |cos| calculation. This equals 210 layers * (r * d_in + r * d_out) summed. But the actual number of LoRA layers is 7 projections * 30 transformer layers = 210, and each has both a lora_a (d_in, r) and lora_b (r, d_out). The flattened adapter vector concatenates ALL parameters (both A and B matrices). The actual d_eff = sum of all adapter parameter counts = 21,626,880 (confirmed by `trainable_params` in results.json). The expected |cos| ~ sqrt(2 / (pi * 21.6M)) ~ 0.00017. The observed 0.0010 is ~6x higher than this theoretical baseline, not "consistent with near-random" as claimed. However, at these magnitudes (0.001 vs 0.0002), both are functionally zero for composition purposes. **Severity: low** -- the qualitative conclusion (near-orthogonal) is correct even if the quantitative comparison is loose.

3. **The unit-weight vs 1/N comparison has a subtle error.** The `compose_adapters` function with `scale_per_adapter=1.0` computes `mx.sum(stacked, axis=0) * 1.0`, which gives the raw sum of all 5 adapters. But in the LoRA forward pass, the LoRA scale factor (alpha=20.0) is already baked into the LoRALinear layer. So unit-weight composition applies each adapter at full alpha=20.0 strength, while 1/N applies each at alpha*1/5 = 4.0 effective strength. The fact that unit-weight (sum of 5 adapters at full strength) gives LOWER PPL than 1/N is indeed surprising and worth noting, but the paper correctly identifies this. No mathematical error, just confirming the mechanism is understood.

## Novelty Assessment

**Genuine novelty:**
- First demonstration of LoRA fine-tuning on BitNet-b1.58-2B-4T via MLX. The vjp workaround (unpack ternary to bfloat16 nn.Linear for training) is a practical engineering contribution. No prior work in references/ implements this.

**Prior art:**
- LoTA-QAF (referenced in BITNET_SOLE_BRIEFING.md) is the closest published work -- ternary adaptation of ternary bases. However, LoTA-QAF uses ternary ADAPTERS (not FP16 LoRA on ternary base), so this experiment explores a different point in the design space.
- MoTE (2506.14435) uses ternary routed experts on frozen shared base, which is architecturally similar to what SOLE proposes. This experiment validates a component (FP16 LoRA composition on ternary base) but does not compare to MoTE's approach.
- The paper correctly cites prior SOLE BitNet micro experiments (d=64 track).

**Delta over existing work:** Moderate. The engineering contribution (vjp workaround) is real. The composition results are directionally useful but limited by the caveats below.

## Experimental Design

**What it tests well:**
- K1 (loads and runs) is definitively answered.
- K4 (individual adapter improvement) is clean -- all 5 domains improve.
- Orthogonality measurement is properly executed.

**Critical design flaws:**

1. **Train/val contamination is severe, not a minor caveat.** Training data is `texts[:500]`, validation is `texts[500:550]`. These come from the SAME dataset, sequentially. For datasets like GSM8K or TinyStories, adjacent examples may share structure, topics, or even be near-duplicates. The 26.5% average PPL improvement over base could be partially or wholly explained by the adapter memorizing distribution-specific patterns from training data that leak into the validation set. This is explicitly acknowledged in limitations but understated. For K2 (convergence) and K4 (domains improved), this contamination means the PPL numbers are upper bounds on true generalization.

2. **Two domains show training DIVERGENCE, not just non-convergence.** Python: loss 1.03 -> 1.12 (+8.7%). Creative: loss 1.17 -> 1.58 (+35.0%). These adapters got WORSE during training. Yet both still show PPL improvement over base (python +19.1%, creative +22.3%). This is paradoxical and suggests one of: (a) the validation set is so similar to training data that even a partially-trained adapter looks good, (b) the first few steps of LoRA training capture the domain signal and subsequent steps overfit/diverge, or (c) the random LoRA-A initialization already provides useful signal. The paper does not investigate this paradox. If (a) is the explanation, it undermines the entire PPL evaluation.

3. **Convergence criterion is too weak.** "last_50_avg < first_50_avg * 0.95" means a 5% reduction in loss. With 200 steps and batch_size=1, the loss curve is extremely noisy. The convergence criterion should use a proper statistical test or at minimum compare to a no-learning-rate control. With only 200 steps cycling through 500 examples (each seen at most once with seq_length=256), the model barely trains.

4. **25 validation samples is very few.** The standard error of PPL estimated from 25 samples is large. For legal (base PPL 21.89), the 95% CI could easily span +/-30%. The paper acknowledges this but still makes per-domain PPL comparisons as if they are precise.

5. **The composition comparison is confounded by scale.** The composed model uses alpha/N = 4.0 effective scale vs individual adapter at alpha = 20.0. The composition is testing "weakened signal from 5 domains" vs "full signal from 1 domain." The 3.59x ratio is expected to be >1.0 by construction. The more informative test would be: does the composed model beat BASE on each domain? It does (composed PPL < base PPL for all 5 domains), which is the actually interesting result. This is reported but not emphasized as the primary metric.

**Controls missing:**
- No random-adapter baseline (what PPL do you get from a randomly initialized but untrained LoRA adapter?). Since `zero_lora_params` initializes lora_a randomly and lora_b to zero, the "zero adapter" produces zero output. But there is no test of whether a random non-zero adapter produces similar "improvement."
- No seed variance (single run).
- No comparison to FP16 base model of similar size (is the ternary base a better or worse composition substrate than FP16?). This would require a separate experiment but is the key claim motivating the BitNet track.

## Hypothesis Graph Consistency

The kill criteria in HYPOTHESES.yml match those in the code and paper. The evidence entry is accurate. Status "supported" is appropriate given the caveats.

However, the node `depends_on: [exp_bitnet_composition_stability]` which has status KILLED. The paper lists this prior experiment's finding as "Ternary base 4.2% better composition" with status "KILLED (instability)." It is unusual for a supported experiment to depend on a killed one. The dependency should be interpreted as "motivated by" rather than "built upon."

The note about "5 consecutive BitNet kills" from the grassmannian_init review is important context. This is the 6th BitNet experiment and the first to produce a positive result, but at real scale (d=2560) rather than toy (d=64). The question is whether the positive result comes from the scale change or from testing a fundamentally different question (real model vs toy model).

## Macro-Scale Risks (advisory)

1. **The 3.9GB training memory requirement may limit adoption.** While inference uses packed ternary (~531MB), training requires unpacking to bfloat16. At BitNet-30B scale, this would be ~60GB for training. The vjp workaround is clever but does not scale to larger models on consumer hardware.

2. **The unit-weight-beats-1/N finding directly contradicts macro Qwen results.** At macro (Qwen-0.5B), unit-weight composition gave PPL in the trillions. Here it gives 7.90. The paper attributes this to the ternary base bounding adapter magnitudes. This is a testable hypothesis but ONLY at this specific scale and with these specific adapters. At N=50 on BitNet-2B, unit-weight may still catastrophe.

3. **The FP16 adapter on ternary base creates a type mismatch at inference.** The merged adapter (bfloat16) added to ternary base means inference cannot use pure integer arithmetic. The BITNET_SOLE_BRIEFING.md discusses LoTA-QAF (ternary adapters) as the solution. This experiment validates FP16-on-ternary but the production path may require ternary-on-ternary.

4. **200 steps is insufficient to reveal interference.** With more training, adapters may move into overlapping subspaces. The near-random orthogonality (|cos|=0.001) may be an artifact of under-training. The macro finding (cos=0.142 for trained Qwen adapters) suggests this is likely.

## Verdict

**PROCEED**

The experiment achieves what it set out to do: demonstrate that LoRA fine-tuning and composition on real BitNet-2B-4T is mechanistically feasible on Apple Silicon. The engineering contribution (vjp workaround via ternary unpacking) is genuine and novel. All 4 kill criteria are passed.

The result is directionally useful for the BitNet track of SOLE. However, the evidence is weaker than the paper suggests due to contamination, under-training, and small validation sets. The "supported" status in HYPOTHESES.yml is appropriate -- this is not "proven."

**Non-blocking concerns (should be noted in FINDINGS.md caveats):**
1. Train/val contamination inflates all PPL improvement numbers. The 26.5% average improvement is an upper bound.
2. Two domains (python, creative) show training divergence yet still "improve" on validation -- this paradox is unexplained and suggests contamination or evaluation artifact.
3. The orthogonality result (|cos|=0.001) likely reflects under-training rather than a ternary base advantage. Fully converged adapters at macro showed 142x worse orthogonality.
4. Single seed, 25 validation samples, 200 training steps -- the statistical power is minimal.
5. The "unit-weight beats 1/N" finding should not be generalized without testing at higher N and with fully converged adapters.
