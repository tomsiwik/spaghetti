# Code Audit: Expert Promotion

## 1. Capacity Exploit via Unfreeze Confound
**Location**: `run_experiment.py:_train_adapter`
**Line**: `model.unfreeze(keys=["lora_b"], strict=False)`
**Flaw**: This command indiscriminately unfreezes all parameters with the string `lora_b` in their name. During phase 3 (training new adapters on the promoted base), the "promoted" medical adapter is implemented as a frozen LoRA wrapper. However, this global unfreeze call reactivates the B-matrices of the promoted medical adapter. 
**Impact**: Instead of training a new adapter on a fixed base, the model trains *both* the new adapter and the promoted medical adapter simultaneously on the new domain data. This effectively doubles the trainable parameter count (from ~17M to ~35M) and overwrites the medical domain knowledge with code/math gradients. The observed "convergence" is not proof of the promoted base's stability, but rather the result of a massive capacity exploit.

## 2. Fake Quantization Promotion
**Location**: `run_experiment.py:phase_promote_and_measure`
**Flaw**: The script claims to promote the adapter into the base model weights ($W' = W + scale \times B^T A^T$). However, because the base model uses `QuantizedLinear` (Qwen3-4B-4bit), the weights cannot be directly modified. The script works around this by instantiating a permanent, frozen `GrassmannianLoRALinear` layer.
**Impact**: This is mathematically equivalent at inference time but physically distinct. It sidesteps the realities of actual weight merging, including potential quantization errors if the continuous delta were folded back into a quantized representation.

## 3. Statistically Vacuous MMLU Evaluation
**Location**: `run_experiment.py:MMLU_QUESTIONS`
**Flaw**: The script hardcodes exactly 50 MMLU questions.
**Impact**: A 50-question evaluation yields a 95% confidence interval of roughly $\pm 7.5$ percentage points. The script observes a $0$pp degradation and uses this to assert the promotion is "safe" and perfectly preserves general knowledge. This is a severe methodological flaw masquerading as empirical proof. A sample size of 50 is vastly insufficient to prove zero degradation in a model's broad reasoning capabilities.
