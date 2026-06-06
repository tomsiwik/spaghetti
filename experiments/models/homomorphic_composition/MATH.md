# Homomorphic Expert Composition: Mathematical Foundations

## 1. Problem Statement

Given N experts with LoRA deltas {Delta_1, ..., Delta_N} where each
Delta_i in R^{d_in x d_out}, compute the composition (average):

    Delta_avg = (1/N) * sum_{i=1}^{N} Delta_i

without any contributor revealing their raw Delta_i to the aggregator.

## 2. Paillier Cryptosystem

The Paillier scheme (Paillier, 1999) is an additively homomorphic
public-key encryption system.

### Key Generation

1. Choose two large primes p, q of equal length (n_bits/2 each)
2. Compute n = p * q, lambda = lcm(p-1, q-1)
3. Public key: pk = (n, g) where g = n + 1
4. Secret key: sk = (lambda, mu) where mu = (L(g^lambda mod n^2))^{-1} mod n
   and L(x) = (x - 1) / n

### Encryption

For plaintext m in Z_n:

    Enc(m) = g^m * r^n mod n^2

where r is a random value in Z_n*.

### Decryption

    Dec(c) = L(c^lambda mod n^2) * mu mod n

### Homomorphic Properties

**Additive homomorphism** (the key property):

    Dec(Enc(m_1) * Enc(m_2) mod n^2) = m_1 + m_2 mod n

This means we can add encrypted values without decrypting them.

**Scalar multiplication** (derived from additive):

    Dec(Enc(m)^k mod n^2) = k * m mod n

### Critical Property: Noiseless

Unlike lattice-based FHE schemes (BFV, BGV, CKKS), Paillier decryption
recovers the exact plaintext. There is NO noise accumulation from
homomorphic operations. Quality degradation, if any, comes entirely
from the float-to-integer quantization.

## 3. Fixed-Point Quantization

Paillier operates on integers in Z_n. We must convert float32 weights
to integers and back.

### Quantization Scheme

Given float array x with max absolute value M:

    scale = (2^{b-1} - 1) / M

    Q(x_i) = round(x_i * scale)      [float -> int]
    Q^{-1}(z_i) = z_i / scale         [int -> float]

where b is the precision in bits.

### Quantization Error Bound

For a single value x_i:

    |x_i - Q^{-1}(Q(x_i))| <= M / (2^{b-1} - 1) / 2

which is the rounding error of half an LSB.

For the averaged result:

    avg_quantized = Q^{-1}(sum(Q(x_i)) / N)

The error per value after averaging is bounded by:

    |error| <= M / (2^{b-1} - 1) / 2

since the rounding errors from different contributors are independent
and averaging does not amplify them (they partially cancel by CLT).

### Precision at Different Bit Widths

| Bits | Max int | Precision (rel to max) | For max_val=0.01 |
|------|---------|----------------------|------------------|
| 16   | 32767   | 3.1e-5               | 3.1e-7           |
| 24   | 8388607 | 1.2e-7               | 1.2e-9           |
| 32   | 2.1e9   | 4.7e-10              | 4.7e-12          |

At 24-bit, the quantization error is below float32 precision (~1.2e-7
relative), making it effectively lossless.

## 4. Batch Packing

### Motivation

Encrypting each weight individually is prohibitively expensive.
A 2048-bit Paillier encryption takes ~8ms (with gmpy2).
For 32768 weights: 32768 * 8ms = 262s per expert.

### Packing Scheme

We pack multiple quantized integers into a single Paillier plaintext.
With a 2048-bit key, the plaintext space is Z_n where n ~ 2^{2048}.

Each value is placed in a "slot" of s = b + h bits, where:
- b = quantization bits (e.g., 24)
- h = headroom bits for accumulation (8 bits supports up to 256 additions)

Values are shifted to be non-negative (add 2^{b-1}) and placed at
consecutive bit positions:

    packed = sum_{i=0}^{k-1} (v_i + 2^{b-1}) << (i * s)

### Packing Density

Available bits = key_bits - 64 (safety margin)

    k = floor((key_bits - 64) / (b + h))

| Key bits | Quant bits | Headroom | Slot bits | Values/pack |
|----------|-----------|----------|-----------|-------------|
| 2048     | 16        | 8        | 24        | 82          |
| 2048     | 24        | 8        | 32        | 62          |
| 2048     | 32        | 8        | 40        | 49          |

### Homomorphic Addition of Packed Values

When two packed ciphertexts are added (homomorphically), each slot
adds independently because the headroom bits prevent carry overflow
between adjacent slots (up to 2^h = 256 additions):

    Enc(packed_A) * Enc(packed_B) mod n^2

    = Enc(packed_A + packed_B) mod n^2

After decryption, each slot contains the sum of the corresponding
values from A and B.

### Cost Reduction

Total packs per matrix (d_in x d_out):

    n_packs = ceil(d_in * d_out / k)

For micro model (d=64, 4 layers, fc1+fc2):
- Total elements: 4 * 2 * 64 * 64 = 32768
- At 24-bit: 32768 / 62 = 529 packs
- At ~130ms per pack: 529 * 0.13 = 69s per expert

vs element-by-element: 32768 * 8ms = 262s per expert

Speedup from packing: ~4x.

## 5. Composition Protocol

### Contributor i (has Delta_i):

1. Receive public key pk from aggregator
2. Quantize: Z_i = Q(Delta_i)
3. Batch pack: packs_i = BatchPack(Z_i)
4. Encrypt: E_i = [Enc(p) for p in packs_i]
5. Send E_i to aggregator

### Aggregator:

1. Receive {E_1, ..., E_N} from N contributors
2. Homomorphic sum: E_sum = [E_1[j] * E_2[j] * ... * E_N[j] for j in packs]
3. Decrypt: Z_sum = BatchUnpack(Dec(E_sum))
4. Average + dequantize: Delta_avg = Q^{-1}(Z_sum) / N

### Security Guarantees

- Contributors only reveal encrypted deltas; aggregator cannot learn
  individual Delta_i values (semantic security of Paillier)
- Only the aggregator holding sk can decrypt the sum
- The aggregator learns only the sum (and thus the average), not
  individual contributions

### Computational Complexity

Per expert:
- Quantization: O(P) where P = total parameters
- Batch packing: O(P) (Python integer construction)
- Encryption: O(P/k * C_enc) where C_enc ~ O(n_bits^2) for modular exp
- Sending: O(P/k * 2*n_bits) bits of ciphertext

Aggregation:
- Homomorphic addition: O(N * P/k * C_mul) where C_mul ~ O(n_bits^2)
- Decryption: O(P/k * C_dec) where C_dec ~ O(n_bits^2)

Total: dominated by encryption, which is O(P/k * C_enc * N).

## 6. Worked Example (Micro Scale)

Parameters:
- d = 64, n_layer = 4, sublayers = 2 (fc1, fc2)
- Total elements P = 4 * 2 * 64 * 64 = 32768
- N = 2 experts, quantization = 24 bits
- Key size = 2048 bits, packing k = 62

Computation:
- Packs per expert: ceil(32768 / 62) = 529
- Total encryptions: 529 * 2 = 1058
- At 130ms/pack: 137s encrypt
- Homomorphic additions: 529 (one per pack)
- At 0.15ms/add: 0.08s
- Decryptions: 529
- At 4ms/pack: 2.1s
- Total encrypted: ~140s

Plaintext:
- 8 matrix additions + division: ~2ms

Slowdown: ~70000x

Quality loss: 0.000002% (24-bit quantization)

## 7. Assumptions

1. **Honest-but-curious threat model**: Contributors and aggregator
   follow the protocol but try to learn others' data. Malicious
   adversaries (who send corrupted ciphertexts) are not considered.

2. **Global scale agreement**: All contributors use the same
   quantization scale (max absolute value). This leaks the range
   of each contributor's weights. A fixed scale could be used
   instead, at the cost of precision for small-magnitude experts.

3. **No composition beyond averaging**: Paillier supports addition
   and scalar multiplication but NOT multiplication of two ciphertexts.
   More complex composition (e.g., TIES sign election, DARE masking)
   requires interaction or switching to FHE.

4. **gmpy2 acceleration**: Performance numbers assume the gmpy2 C
   library is available. Without it, Paillier is ~8x slower.

## 8. Scaling Analysis

For macro model (Qwen 0.5B, d=896, r=16):

Per-layer LoRA delta: 896 * 896 = 802816 parameters (both fc1 and fc2)
With 24 layers: ~19M parameters total

Packs at 24-bit: 19M / 62 = 306451 packs
At 130ms/pack: 39839s = 11.1 hours per expert

This is clearly impractical for element-level Paillier. Alternatives:
- CKKS (approximate HE): batches floats natively, ~100x faster
- Secure aggregation (SMPC): no encryption of weights needed
- Differential privacy: add calibrated noise pre-sharing, O(P) cost
- Trusted execution environments (TEE): hardware-level privacy
