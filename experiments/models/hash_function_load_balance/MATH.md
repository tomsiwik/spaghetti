# Hash Function Load Balance: Mathematical Foundations

## Variables

| Symbol | Definition | Domain |
|--------|-----------|--------|
| N | Number of experts on the ring | {4, 8, 16, 32, 64, 128} |
| V | Virtual nodes per expert | {100, 200, 500, 1000} |
| T | Total query tokens | 1,000,000 |
| h(x) | Hash function mapping bytes to [0, 2^32) | FNV1a, xxHash32, MurmurHash3, SHA-256, Python hash |
| L_i | Load on expert i (number of tokens routed to it) | [0, T] |
| R | Max/min load ratio: max(L_i) / min(L_i) | [1.0, inf) |
| J | Jain's fairness index | (0, 1.0] |
| sigma | Coefficient of variation of load | [0, inf) |

## Theoretical Load Distribution

A consistent hash ring with N experts and V virtual nodes each places NV points
uniformly on the ring [0, 2^32). Each expert's arc fraction determines its load share.

**Perfect balance**: Each expert owns exactly 1/N of the ring, receiving T/N tokens.

**Balls-into-bins model**: Placing NV virtual nodes on [0, 2^32) is equivalent to
throwing NV balls into a continuum of bins. The arc lengths follow a symmetric
Dirichlet distribution. For expert i with V virtual nodes among NV total:

    E[L_i] = T/N
    Var[L_i] = T * (N-1) / (N^2 * (NV + 1))

For large NV, the max/min ratio is bounded by:

    R_expected approx (1 + c * sqrt(ln(N) / (NV))) / (1 - c * sqrt(ln(N) / (NV)))

where c is a constant near sqrt(2). This predicts:

| V | N=8 R_expected |
|---|----------------|
| 100 | 1.155 |
| 200 | 1.107 |
| 500 | 1.067 |
| 1000 | 1.047 |

## FNV1a Failure Mode

FNV1a-32 is a simple multiply-and-XOR hash:

    h = 0x811C9DC5
    for each byte b:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF

The critical weakness: when hashing structured input (expert_id, virtual_node_index)
packed as bytes, the mixing is insufficient. Adjacent expert IDs produce correlated
ring positions, creating arc clustering rather than uniform distribution.

Empirically, FNV1a achieves R = 2.0-5.5x at V=100 (vs theoretical 1.15x), and
R = 1.6-2.3x even at V=1000. This is 10-40x worse than theoretical.

## Jain's Fairness Index

    J(L_1, ..., L_N) = (sum_i L_i)^2 / (N * sum_i L_i^2)

Properties:
- J = 1.0 when all loads are equal (perfect fairness)
- J = 1/N when one expert handles all load (worst case)
- J relates to CV: J = 1 / (1 + sigma^2)

## Why Hash Quality Matters for SOLE

In SOLE, load imbalance means some experts are over-utilized while others are
underused. At R = 1.8x (FNV1a, N=8), the busiest expert handles 22.5% vs
the theoretical 12.5%, nearly 2x overloaded. This:

1. Wastes capacity of underloaded experts
2. Concentrates quality degradation risk on overloaded experts
3. Makes removal impact uneven (removing the overloaded expert displaces 1.8x more tokens)

## Worked Example (N=8, V=150, seed=42)

FNV1a ring arc ownership:
- Expert 0: 12.46%, Expert 2: 16.27%, Expert 3: 10.36%
- Max/Min = 16.27/10.36 = 1.570

MurmurHash3 ring arc ownership:
- Expert 0: 12.63%, Expert 1: 11.45%, Expert 5: 13.39%
- Max/Min = 13.39/11.45 = 1.169

The 1.57x vs 1.17x difference is entirely due to hash function quality, with
identical ring construction logic.

## Computational Cost

All hash functions tested are O(len(input)) per call. For our 8-byte input
(expert_id + virtual_node_index):

| Hash | Theoretical cost | Observed (relative) |
|------|------------------|---------------------|
| FNV1a (Python) | O(n), simple | 1.0x (baseline, slow due to Python loop) |
| xxHash32 (C ext) | O(n), SIMD-optimized | ~0.1x (C library) |
| MurmurHash3 (C ext) | O(n), block-optimized | ~0.1x (C library) |
| SHA-256 (C ext) | O(n), crypto overhead | ~0.3x (OpenSSL) |
| Python hash | O(n), SipHash-2-4 | ~0.05x (built-in) |

Ring construction is a one-time O(NV * hash_cost) operation. At N=128, V=1000:
128K hash calls, negligible for any hash function.
