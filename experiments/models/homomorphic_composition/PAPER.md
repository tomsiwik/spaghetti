# Homomorphic Expert Composition: Research Digest

## Hypothesis

LoRA expert deltas can be composed (averaged) via Paillier homomorphic
encryption without quality degradation, enabling privacy-preserving
expert contribution.

**Falsifiable**: encryption noise or quantization degrades composition
quality by >5%, OR encrypted composition is >100x slower than plaintext.

## What This Model Is

This experiment tests whether the additive composition of LoRA expert
deltas -- which is mathematically just weighted averaging -- can be
performed in encrypted space using the Paillier partially-homomorphic
encryption scheme.

The protocol:
1. Each contributor quantizes their float32 LoRA deltas to fixed-point
   integers (16/24/32-bit precision)
2. Values are batch-packed into large integers and encrypted with a
   shared Paillier public key
3. The aggregator adds all encrypted packs homomorphically (no decryption)
4. The aggregator decrypts the sum and divides by N to get the average
5. The averaged deltas are applied to the base model

The key insight: Paillier encryption is **noiseless** -- decryption
exactly recovers the plaintext. The only quality risk is float-to-integer
quantization, which at 24-bit precision introduces errors below float32's
own precision limits.

## Lineage in the Arena

```
gpt (base)
  +-- lora_gpt (LoRA fine-tuning)
        +-- lora_merging_bakeoff (composition methods)
              +-- homomorphic_composition (this: encrypted averaging)
```

## Key References

- **Paillier (1999)**: "Public-Key Cryptosystems Based on Composite
  Degree Residuosity Classes". Defines the additively-homomorphic
  encryption scheme used here.
- **BatchCrypt (USENIX ATC 2020)**: "Efficient Homomorphic Encryption
  for Cross-Silo Federated Learning". Batch packing technique for
  amortizing Paillier encryption cost.
- **python-paillier (data61)**: Open-source Paillier implementation
  used in this experiment.
- **Prior project work**: lora_merging_bakeoff proved simple averaging
  is optimal for orthogonal LoRA deltas. This experiment adds privacy
  to that averaging step.

## Empirical Results

### Quality (KC1): Encryption Preserves Composition Exactly

Composition quality measured as average validation loss across 2 domains
(a-m, n-z name splits), 3 seeds (42, 123, 7).

| Method | avg val loss | vs plaintext | max weight error |
|--------|-------------|-------------|-----------------|
| Plaintext avg | 0.5236 | baseline | -- |
| Encrypted 16-bit | 0.5236 | -0.000012% | 8.2e-06 |
| Encrypted 24-bit | 0.5236 | +0.000002% | 3.7e-08 |
| Encrypted 32-bit | 0.5236 | +0.000000% | 1.5e-08 |

Multi-seed quality at 24-bit quantization:

| Seed | Plaintext | Encrypted | Gap |
|------|-----------|-----------|-----|
| 42 | 0.523171 | 0.523171 | +0.000002% |
| 123 | 0.524230 | 0.524230 | -0.000003% |
| 7 | 0.523216 | 0.523216 | -0.000003% |

**KC1 verdict: PASS.** Worst quality gap across all configurations and
seeds is 0.000012%. The kill criterion was 5%. Paillier is noiseless;
24-bit quantization error is below float32 precision.

### Performance (KC2): Paillier is ~7000x Slower

Timing measured at 2048-bit key (standard security), batch-packed,
with gmpy2 acceleration. Micro model: d=64, 4 layers, 32768 parameters.

| Operation | Time | Notes |
|-----------|------|-------|
| Plaintext averaging | 17ms | numpy sum + divide |
| Paillier encrypt (per expert) | 52s | 529 packs at ~98ms each |
| Homomorphic addition (N=2) | 0.06s | 529 pack multiplications |
| Paillier decrypt | 16s | 529 pack decryptions |
| **Total encrypted** | **~121s** | dominated by encryption |
| **Slowdown** | **~7000x** | vs plaintext |

Breakdown of encrypted composition time:
- Encryption: 86% (the bottleneck)
- Decryption: 13%
- Homomorphic addition: <0.1%

**KC2 verdict: KILL.** Slowdown is ~7000x, exceeding the 100x threshold
by 70x. Even the homomorphic addition itself is fast (60ms), but the
encrypt/decrypt overhead dominates.

### Overall Verdict: PARTIALLY PROVEN

KC1 (quality) passes trivially. KC2 (performance) fails by a wide
margin. The hypothesis is **half-validated**: encrypted composition
preserves quality perfectly, but Paillier's computational overhead
makes it impractical for element-level weight encryption.

## Micro-Scale Limitations

1. **Micro model is 32K parameters.** Macro LoRA (Qwen 0.5B) has ~19M
   parameters per expert, making Paillier ~600x more expensive. At macro
   scale, encrypted composition would take ~11 hours per expert.

2. **Only simple averaging tested.** More complex composition methods
   (TIES, DARE, router calibration) require operations beyond addition,
   which Paillier cannot support. These would need FHE or interactive
   protocols.

3. **2-domain test only.** Performance scales linearly with N (each
   expert encrypts independently), but we did not test N=5 or larger.

4. **No network latency.** In a real distributed system, ciphertext
   transfer adds overhead. Paillier ciphertexts are 2x the key size
   per value, so 32768 values at 24-bit packing need ~34MB.

5. **Single-machine timing.** In practice, N contributors encrypt in
   parallel, so wall-clock time is dominated by the slowest contributor
   plus aggregation. The aggregation step (decrypt + divide) is fast.

## What Would Kill This

### At Micro Scale (already measured)
- KC1 is so far from the threshold (0.000003% vs 5%) that no reasonable
  variation could kill it. Paillier is mathematically exact.
- KC2 is killed. Only a fundamentally different encryption scheme could
  bring the overhead below 100x.

### At Macro Scale
- **Quantization noise at scale**: With 19M parameters, the cumulative
  effect of quantization noise might be detectable. However, at 24-bit,
  the per-element error is ~1e-9, which should be negligible.
- **Memory**: 19M values at 62/pack = 306K Paillier ciphertexts at
  512 bytes each = 157MB per expert. For 100 experts: 15.7GB. This is
  manageable but not trivial.

## Implications for the Architecture

The quality result is the key takeaway: **additive expert composition
is perfectly compatible with homomorphic encryption**. This is not
surprising mathematically (Paillier is exact for addition), but it
validates the entire protocol end-to-end including quantization.

For practical privacy-preserving composition, alternatives to Paillier
that maintain the quality guarantee:

| Approach | Overhead | Quality | Complexity |
|----------|----------|---------|------------|
| Paillier (this work) | ~7000x | exact | simple |
| CKKS (approximate HE) | ~100x | ~exact | moderate |
| Secure aggregation (SMPC) | ~10x | exact | moderate |
| Differential privacy | ~1x | noisy | simple |
| Trusted execution (TEE) | ~2x | exact | hardware |

The recommended next step for the project: implement secure aggregation
via secret sharing (Shamir or additive), which provides exact composition
at ~10x overhead -- well within the 100x criterion.
