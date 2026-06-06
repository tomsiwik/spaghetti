# MATH.md: Third Structurally Diverse Domain — Caesar Cipher

## TYPE: guided-exploration
## PROVEN FRAMEWORK: M2P per-domain quality scaling (#359, #361, #362)
## UNKNOWN: Whether M2P quality holds on a domain structurally different from sequence reordering

---

## Problem

All M2P quality results (#359-#362) are validated on 2 domains: sort and reverse.
Both are sequence-reordering tasks. The adversarial review (critique #2) correctly
notes this is insufficient — we need a structurally different domain.

## Domain Design: Caesar Cipher

**Task:** Input = plaintext string, output = shifted string (shift by fixed offset per sample).
Example: "abc" with shift=3 → "def". Wraps around: "xyz" → "abc".

**Why this is structurally different from sort/reverse:**
- Sort/reverse: REARRANGE existing characters (permutation group)
- Cipher: TRANSFORM each character independently (substitution group)
- No positional dependency between input characters in cipher (each transforms independently)
- Cross-domain transfer from sort/reverse should be LOW (confirming diversity)

**Mathematical property:** Caesar cipher is a group homomorphism Z_26 → Z_26.
The M2P must learn a different computational primitive (modular addition) vs
sort/reverse (comparison/permutation).

## Kill Criteria

**K_3dom:** M2P quality ≥ 85% on ALL 3 valid domains (sort, reverse, cipher) at d=512.
**K_diversity:** Cross-domain M2P transfer from sort→cipher < 50% (confirming structural difference).
**K_replication:** sort and reverse quality within 5pp of Finding #361 (101.0%).

## Predictions

| Metric | Predicted | Reasoning |
|--------|-----------|-----------|
| Cipher M2P quality | 85-100% | Modular addition is learnable, d_int likely < 64 |
| Sort/reverse quality | 98-102% | Replication of #361 |
| Cross-domain sort→cipher | < 30% | Different computational primitive |
| Cross-domain cipher→sort | < 30% | Same argument, symmetric |
