# MATH.md — T4.5: Pierre Adapter Format Compatibility

## Problem Statement

Pierre adapters are trained with mlx_lm.lora and stored in MLX-native format.
To integrate with the broader ecosystem (HF PEFT, vLLM runtime LoRA, Unsloth),
we must prove that our format is structurally equivalent to standard HF PEFT format
and define the lossless bijection between the two representations.

## Theorem 1 (LoRA Format Equivalence)

**Claim:** The MLX adapter representation and the HF PEFT representation encode
the same linear transformation up to transposition. A lossless bijection exists.

**Setup:**
- Input dimension: d_in = 2560 (Gemma 4 q_proj input)
- Output dimension: d_out = 2048 (Gemma 4 q_proj output)
- Rank: r = 6

**MLX representation:**
```
lora_a ∈ R^{d_in × r}  (shape [2560, 6])
lora_b ∈ R^{r × d_out}  (shape [6, 2048])
Δy = x · lora_a · lora_b   (x ∈ R^{1 × d_in})
```

**HF PEFT representation:**
```
lora_A.weight ∈ R^{r × d_in}   (shape [6, 2560])  ← transposed lora_a
lora_B.weight ∈ R^{d_out × r}  (shape [2048, 6])  ← transposed lora_b
Δy = x · lora_A.weight^T · lora_B.weight^T
   = x · (lora_A.weight^T) · (lora_B.weight^T)
```

**Proof:**
```
Δy (MLX)  = x · lora_a · lora_b
           = x · A · B           [A = lora_a, B = lora_b]

Δy (PEFT) = x · lora_A.weight^T · lora_B.weight^T
           = x · A^{T^T} · B^{T^T}  [using A_peft = A^T, B_peft = B^T]
           = x · A · B              [double transpose = identity]
           = Δy (MLX)  ∎
```

**Bijection:**
```
MLX → PEFT:   A_peft = lora_a.T,  B_peft = lora_b.T
PEFT → MLX:   lora_a = A_peft.T,  lora_b = B_peft.T
```

## Theorem 2 (Grassmannian Metadata Preservation)

**Claim:** The Grassmannian initialization constraint (A has orthonormal rows,
i.e., A·A^T ≈ I_r) is preserved under the PEFT representation as a VALUE property
of the stored matrix, independent of the format schema.

**Proof:**
The constraint is A_peft · A_peft^T = I_r (rows of A_peft are orthonormal).
Since A_peft = lora_a.T, we have:
```
A_peft · A_peft^T = lora_a.T · lora_a ≈ I_r
```
This is a property of the stored floating-point values, not of the JSON schema
or file format. Any reader of the safetensors file observes the same matrix.

The Grassmannian property can be annotated as metadata:
```json
{
  "pierre_metadata": {
    "construction": "qr",
    "property": "orthonormal_rows",
    "rank": 6,
    "scale": 6.0,
    "verified_max_deviation": "<1e-6"
  }
}
```
This metadata is stored in adapter_config.json as a custom field.
PEFT, vLLM, and Unsloth all ignore unknown fields in adapter_config.json. ∎

## Theorem 3 (RETRACTED — subset-direction fallacy)

**Original (wrong) claim:** "If f ∈ F_peft and F_vllm ⊆ F_peft, then f is vLLM-compatible."

**Why it was wrong:** F_vllm ⊆ F_peft means every vLLM-valid file is PEFT-valid, i.e. vLLM
imposes stricter structure. It does NOT imply the converse (f ∈ F_peft ⟹ f ∈ F_vllm).
In practice vLLM often requires fused `qkv_proj` weights while PEFT accepts separate
`q_proj` / `k_proj` / `v_proj`, so a PEFT-valid adapter that is not fused will crash at
vLLM load. The same direction reversal applied to the Unsloth claim.

**Consequence for this experiment:** We cannot conclude vLLM or Unsloth runtime compat
from PEFT compat alone. K1089 and K1090, as written in the DB, require a real runtime
load on CUDA — unreachable on the MLX/Apple-Silicon target (PLAN.md Part 2, "MLX only").
These two KCs are therefore marked `skip (platform-unavailable)` for this rerun and
their verification is delegated to the follow-up `exp_followup_format_compat_peft_required`
(to be run on a CUDA machine).

## Substrate Note (audit-rerun)

The dependency `exp_p1_t2_single_domain_training` was killed and its
`adapters.safetensors` was not retained. For a FORMAT test this is acceptable:
we build synthetic Grassmannian A-matrices via QR decomposition (known to satisfy
A^T·A = I_r up to float32 noise). This isolates K1091 from training-drift (see
prior run's 0.579 deviation) and is a valid substrate for bijection + schema
tests. Training-drift of Grassmannian weights is a distinct scientific question
handled by interference experiments, not here.

## Kill Criteria Predictions (audit-rerun)

**K1088 (PEFT LoraConfig):** PASS
- `peft` is HARD-required (no ImportError bypass). Experiment fails outright if
  the library is missing.
- `peft.LoraConfig(**cfg)` must construct cleanly.
- `peft.PeftConfig.from_pretrained(dir)` must round-trip `r`, `target_modules`,
  `peft_type`.
- 42 `.lora_A.weight` + 42 `.lora_B.weight` keys must be present.

**K1089 (vLLM runtime):** SKIP (cuda_unavailable_on_platform)
- Theorem 3 subset fallacy retracted; PEFT compat does not imply vLLM compat.
- Runtime test requires CUDA + vLLM — not available on MLX target. Marked SKIP,
  not PASS. Verdict cannot be `supported` while this KC is unreached.

**K1090 (Unsloth runtime):** SKIP (cuda_unavailable_on_platform)
- Unsloth requires CUDA + bitsandbytes. Same reasoning as K1089.

**K1091 (Grassmannian metadata):** PASS iff max_deviation < 1e-6
- Metadata round-trip through `adapter_config.json`.
- `"property": "orthonormal_rows"` is only written when the measured deviation
  satisfies the tolerance; otherwise `"drifted_from_orthonormal"` is written.
  The prior run wrote `"orthonormal_rows"` regardless of a 0.579 deviation —
  that antipattern is now structurally impossible via `property_claim_truthful`
  check.

## Quantitative Predictions

| Metric | Predicted | Basis |
|--------|-----------|-------|
| Converted keys count | 84 (42 layers × 2) | Known adapter structure |
| A-matrix transposition error | 0.0 (exact float copy) | No approximation |
| B-matrix transposition error | 0.0 (exact float copy) | No approximation |
| adapter_config.json fields | ≥ 8 required PEFT fields | PEFT spec |
| Pierre metadata fields | 5 (construction, property, rank, scale, verified_max_deviation) | Theorem 2 |
| PEFT schema validation | Pass | Theorem 1 |
