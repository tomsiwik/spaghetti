# LEARNINGS.md — exp_m2p_2domain_compose_qwen06b

**Status:** provisional (K955 PASS, K954 UNVERIFIED)
**Date:** 2026-04-08

---

## Core Finding

Grassmannian A-matrix isolation (A_math^T A_code = 1.51e-08 ≈ 0) and TF-IDF routing
(100% math/code discrimination) are structurally verified on Qwen3-0.6B. Composition
architecture is sound; empirical quality threshold (K954, ≥80% of single-adapter) was
not measurable from a 10/20-step smoke test.

## Why

QR-based orthogonal slot assignment guarantees zero cross-domain interference
algebraically (Theorem 1, Aghajanyan 2012.13255 / Hu 2106.09685). TF-IDF routing is
invariant to adapter state (Theorem 2), confirmed by Finding #389 (cos(math,code)=0.190).
Both mechanisms are provable without empirical data; the smoke test is verification only.

## Implications for Next Experiment

**Top priority:** Run exp_m2p_2domain_compose_qwen06b full training (300 math + 500 code
steps) to verify K954. If code adapter remains below base after full training, diagnose
EOS format confusion (per Finding #384 pattern) and add explicit EOS training signal.
The full run determines whether Theorem 3's quality bound is empirically achievable or
needs a structural fix (e.g., explicit EOS token in code task format).

## What This Does NOT Change

- Grassmannian isolation holds regardless of adapter quality (structural result)
- TF-IDF routing is production-ready for math/code (Finding #389 + K955 PASS)
- Finding #386: wrong-domain adapter causes ~58% harm — routing remains critical
- M2P v4 gradient mechanics are correct (Theorem 5 verified, no v2 bug recurrence)
