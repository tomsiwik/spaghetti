# Finding Critique: Expert Promotion

## Verdict: INVALID (Metric Hacking & Capacity Confound)

The conclusions drawn in this experiment regarding "safe" expert promotion are completely undermined by severe methodological flaws, metric hacking, and a critical implementation bug that artificially boosts capacity.

### 1. Statistically Vacuous MMLU Evaluation
The claim of "0pp MMLU degradation" is statistically meaningless. The evaluation uses a mere 50 questions from MMLU.
- A 50-question sample has a large margin of error (95% CI) making any precision claims invalid.
- Zero degradation on 50 questions is indistinguishable from random noise. Claiming the Davis-Kahan bound is "conservative" based on a 50-question sample is unscientific.

### 2. The Unfreeze Capacity Confound
The finding admits to an "unfreeze confound" where `model.unfreeze(keys=["lora_b"])` unfroze the promoted adapter's B-matrices.
- This is a massive capacity exploit. The new adapters trained with 35M parameters instead of 17M.
- The promoted medical adapter was actively updating during code/math training. This isn't "adapter stacking" or "promotion"; it's just joint training of a larger capacity LoRA. The claim that "adapters train and converge on the promoted base" is invalid because the base wasn't frozen.

### 3. Fake Quantization Promotion (No Requantization Error)
The experiment claims to prove "promotion into a pre-trained base" but admits that due to `QuantizedLinear`, they used a "frozen LoRA overlay" instead of actually modifying the base weights.
- True weight promotion into a quantized base requires dequantizing, adding the delta, and *requantizing*.
- By using a floating-point LoRA overlay, the experiment entirely dodges requantization error, which is often the dominant source of degradation in quantized model editing. The conclusion that "0.14 GB overhead for permanent domain expertise" is viable ignores that a true weight update would suffer severe quantization noise.

## Conclusion
The findings are built on a statistically invalid benchmark, a massive trainable capacity confound, and an implementation that completely sidesteps the mathematical reality of requantization error. The results must be discarded.