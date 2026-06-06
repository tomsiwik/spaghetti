# Follow-up Experiment Design: Strict Expert Promotion

## Verdict: INVALID

The original expert promotion experiment is invalid due to a combination of severe methodological and implementation flaws:
1. **Capacity Exploit:** A global `unfreeze(keys=["lora_b"])` accidentally unfroze the promoted adapter during subsequent training phases, effectively doubling the trainable parameters and masking any convergence instability.
2. **Fake Quantization:** The promotion used a floating-point LoRA overlay rather than actual weight updating and requantization of the 4-bit base model, completely dodging requantization error.
3. **Statistically Vacuous Evaluation:** The claim of "0pp MMLU degradation" was based on a 50-question sample, rendering the empirical calibration of the Davis-Kahan bounds mathematically meaningless.

## Strict Follow-up Design

To determine if expert promotion is actually viable and mathematically sound, we must design a rigorous follow-up that enforces strict capacity isolation, true weight quantization, and statistically significant evaluation.

### Hypothesis
True weight promotion into a quantized base model (4-bit) incurs a non-trivial requantization error ($E_{quant}$) that causes measurable capability degradation and alters the loss landscape enough to negatively impact the convergence of subsequent adapters, provided the capacity confound is eliminated.

### Math Sketch
1. **True Promotion Operation:** Instead of $W' = W + E$, model the true quantized update: 
   $W' = \text{Quantize}(\text{Dequantize}(W) + \alpha B^T A^T)$
2. **Requantization Error Term:** Define the error introduced by quantization:
   $E_{quant} = W' - (\text{Dequantize}(W) + \alpha B^T A^T)$
3. **Modified Perturbation Bound:** Re-evaluate the Davis-Kahan bounds and Hessian Lipschitz continuity incorporating $\|E_{quant}\|_{op}$, testing if the theoretical safety threshold holds under discrete weight updates.

### Experimental Methodology
1. **Statistically Significant Benchmark:** Replace the 50-question subset with the full MMLU evaluation suite (or a statistically significant sample of at least 1,000+ questions) to measure degradation with a tight 95% confidence interval.
2. **True Weight Quantization:** Implement actual dequantization, addition, and requantization of the Qwen3-4B-4bit base model weights for the "promoted" adapter. Compare this against the "fake" floating-point LoRA overlay as a baseline.
3. **Strict Capacity Isolation:** Fix the unfreeze confound. When training phase 3 adapters, explicitly target only the new adapter keys (e.g., using exact string matching or `strict=True` freezing) to ensure the promoted weights remain absolutely frozen.

### Kill Criteria
The original claims of "safe" expert promotion are dead if:
1. **Requantization Degradation:** True quantized promotion degrades the full MMLU benchmark by >2% compared to the floating-point overlay baseline.
2. **Convergence Failure:** Without the unfreeze capacity exploit, new adapters trained on the truly promoted base exhibit significantly higher loss or slower convergence than those trained on the unmodified base.