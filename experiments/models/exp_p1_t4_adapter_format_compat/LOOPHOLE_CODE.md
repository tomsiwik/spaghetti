# LOOPHOLE_CODE.md — Code Analyzer Audit

**Target:** `exp_p1_t4_adapter_format_compat`
**Verdict:** INVALID (Severe Implementation Flaws)

## 1. Hardcoded Bypass of `peft` Validation (K1088)
In `phase3_k1088` (lines 145-156), the code intentionally bypasses the `peft` validation if the library is not installed:
```python
    except ImportError:
        print("peft not installed (optional). Schema validation passed.")
```
The test then relies entirely on `len(missing) == 0`, which merely checks if keys exist in a set (`REQUIRED_PEFT_FIELDS`). This is a hardcoded metric cheat. If the `peft` library is not installed, the test silently skips the actual schema validation but reports a `PASS`. 

## 2. Superficial String Matching for vLLM Compatibility (K1089)
In `phase4_k1089` (lines 180-192), the validation of vLLM compatibility relies purely on matching string suffixes (`VLLM_KEY_PATTERN_A = ".lora_A.weight"`) and basic rank shape checks:
```python
    a_shape_ok = a_shape[0] == r   # first dim is rank
    b_shape_ok = b_shape[1] == r   # second dim is rank
```
This entirely ignores vLLM's internal mechanics, which often require fused `qkv_proj` weights for attention layers. An MLX adapter outputting `q_proj` will structurally match these string tests but will immediately crash when loaded into a vLLM runtime that expects fused mappings. The script excuses this with `NOTE: Runtime loading requires CUDA; this is a structural format check`, which is a false justification for an invalid test.

## 3. Ignored Grassmannian Drift in Metadata (K1091)
In `phase1_audit` (lines 62-65), the script calculates the Grassmannian deviation (`AtA = A.T @ A` compared to `np.eye`), but it never asserts a threshold bound:
```python
        deviation = np.max(np.abs(AtA - np.eye(AtA.shape[0])))
        max_deviation = max(max_deviation, deviation)
```
In `phase6_k1091` (lines 239-251), it validates that `pierre_metadata` exists in the configuration and round-trips correctly, but it *fails to assert* that `verified_max_deviation < 1e-6`. It merely stores the drifted value (measured at 0.579) alongside the hardcoded `"property": "orthonormal_rows"`. The code writes falsified guarantees into the `adapter_config.json`, claiming orthogonality regardless of the actual mathematical state of the tensor.

## 4. Fake Unsloth Compatibility Check (K1090)
In `phase5_k1090` (lines 208-220), there is no integration with Unsloth at all. The script merely checks if `lora_alpha` is a number and if four keys exist in a dictionary. It makes a massive logical leap assuming that if K1088 passes, Unsloth compatibility automatically follows, offering no actual validation.
