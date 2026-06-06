# LOOPHOLE_FINDING.md — Finding Analyzer Critique

**Target:** `exp_p1_t4_adapter_format_compat`
**Verdict:** INVALID (Severe Methodological Flaws & Metric Hacking)

## 1. Metric Hacking via Missing Dependencies
The experiment claims to validate PEFT format compatibility (K1088), but the code explicitly bypasses the actual `peft` library if it isn't installed. In `phase3_k1088`:
```python
    except ImportError:
        print("peft not installed (optional). Schema validation passed.")
```
This is egregious metric hacking. The test passes automatically if the validation library is missing, falling back to a naive `REQUIRED_PEFT_FIELDS` subset check. A format cannot be claimed "PEFT compatible" without ever being loaded by PEFT.

## 2. False Assumptions About vLLM Compatibility (Theorem 3)
Theorem 3 claims that PEFT compatibility guarantees vLLM compatibility. This is factually false. The experiment only checks if the keys end in `.lora_A.weight`. However, vLLM handles attention projections differently than standard PEFT (e.g., often requiring fused `qkv_proj` weights). A standard `q_proj` adapter exported from MLX will fail at runtime in vLLM due to shape and key mapping mismatches unless explicitly handled by the runtime model loader. The "structural check" is a superficial string matching exercise that proves nothing about actual runtime loadability.

## 3. Complete Lack of Runtime Validation
The experiment repeatedly excuses the lack of runtime testing with "Runtime loading requires CUDA; this is a structural format check". This is a cop-out. `peft` and Hugging Face `transformers` can absolutely load LoRA weights on CPU. The failure to perform even a basic CPU-based `PeftModel.from_pretrained()` forward pass means the claim "format lock-in risk is zero" is completely unsupported by empirical evidence.

## 4. Falsified Grassmannian Guarantees
The mathematical prediction for Grassmannian deviation was `<1e-6`. The actual measured deviation was `0.579`—orders of magnitude worse. While the experiment notes this as "drift", it fails to acknowledge that this completely invalidates the foundational premise of T3 (that Grassmannian initialization provides structural interference guarantees). If the orthogonality is destroyed during training, the adapter is no longer Grassmannian, and the metadata tag `"property": "orthonormal_rows"` is lying to downstream systems. 

**Conclusion:** The experiment's claims of format compatibility are based on superficial string matching and bypassed validation checks. The results are invalid and must be completely redone with actual CPU-based `transformers`/`peft` loading tests.