# python-paillier (data61)

**Source**: https://github.com/data61/python-paillier

**Key insight**: Open-source Paillier partially-homomorphic encryption
library in Python. Supports additive homomorphism (encrypted addition)
and scalar multiplication on encrypted values. Includes built-in
float encoding and a federated learning example.

**Relevance**: Used directly in `exp_homomorphic_composition` to test
whether LoRA expert deltas can be composed in encrypted space. The
library's `phe.paillier` module provides keygen, encryption, decryption,
and homomorphic operations.

**Performance notes**:
- With gmpy2: ~8ms per encrypt, ~0.07ms per add, ~2ms per decrypt (2048-bit)
- Without gmpy2: ~8x slower (pure Python bignum)
- Batch packing amortizes encryption cost by ~60x (packing 62 values per ciphertext at 24-bit quantization)

**Install**: `pip install phe` (optional: `pip install gmpy2` for performance)
