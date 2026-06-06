# Adversarial Review — exp_prod_red_team_adapter_injection

## Verdict: KILL

Pre-registered KILLED outcome replicated: K1667 and K1668 FAIL, K1669 PASS (unexpected). Verdict is consistent across results.json / PAPER.md / DB. Findings #1667 and #1668 confirm the B-matrix non-orthogonality leakage predicted by dependency's K1644 FAIL. Routing this as `review.killed` so the analyst captures the N>1 privacy path (Gram-Schmidt on B during training).

## Adversarial checklist

| # | Check | Result |
|---|-------|--------|
| a | results.json verdict vs DB status | ✓ both KILLED |
| b | all_pass=false consistent with KILLED | ✓ |
| c | PAPER.md verdict line matches | ✓ "## Verdict: KILLED" |
| d | is_smoke=false on full run | ✓ |
| e | MATH.md post-run modification | ✓ only one commit (a3dcc0b pre-reg); no diff since |
| f | Tautology sniff | ✓ all three KCs use independent measurements (logistic probe, SVD vs ground-truth B, greedy decode vs training text) |
| g | K-IDs measure quantity named | ✓ K1667 operationalizes "recovery" as probe advantage (scope narrowing documented in MATH.md); K1668 subspace overlap; K1669 canary token overlap |
| h | Buggy adapter composition | ✓ `ComposedNullSpaceLoRALinear` sums z_a + z_b correctly; no `sum(lora_A`, no `add_weighted_adapter` |
| i | LORA_SCALE safe | ✓ 8.0 < 12 ceiling |
| j | Per-sample routing bug | n/a |
| k | shutil.copy of sibling adapter | ✓ no |
| l | Hardcoded pass:True | ✓ booleans computed |
| m | Target model matches MATH.md | ✓ gemma-4-e4b-it-4bit |
| m2 | MLX idiomaticity | ✓ uses `mx.eval` after forwards, `mx.clear_cache()` between phases, `tree_unflatten`, `mx.set_memory_limit`, `nn.QuantizedLinear` dequant path, `__call__` pattern. No torch mutation. Skill invocation not explicitly cited but code is clean. |
| n | Base accuracy 0% / thinking=0 | n/a (red-team, not benchmark) |
| o | Headline n < 15 | ✓ K1667 n=20/class, K1669 n=20, K1668 n=8 layers |
| p | Synthetic N padding | n/a |
| q | Baseline drift | n/a |
| r | Prediction-vs-measurement table | ✓ PAPER.md lines 28-32 |
| s | Math errors | ✓ ΔW = s·Q·A·B shape derivation correct; SVD right-singular (Vt rows) = output directions; subspace overlap via QR on B_u.T is standard |

## Non-blocking observations

1. **K1669 parameter deviation** (non-blocking; doesn't drive verdict). MATH.md §K1669 specifies prefix_len=30, n_gen=50; run_experiment.py uses prefix_len=15, n_gen=20 — a weakened test. This made PASS more likely. Because K1669 PASSING does not upgrade the verdict to SUPPORTED (K1667/K1668 KILL independently) and PAPER.md explicitly documents it as "lower-bound observation, not a proof of canary safety" with a caveat paragraph (lines 114-120), this is acceptable but should be fixed if K1669 is re-used as a positive claim in v2.

2. **K1667 statistical power**. n=20/class with +15pp advantage over chance is a real signal but wide CI (±0.10-0.15 at 95%). PAPER.md acknowledges this (lines 71-77). The verdict does not rely on statistical significance — the probe advantage simply exists, which suffices to show information leakage.

3. **Surprising K1669 decouple from K1667/K1668** is the behavioral finding worth capturing in LEARNINGS: composition-as-defense against naive verbatim extraction even under full membership-inference leakage. Weak but measurable de facto privacy operation.

## Assumptions (judgment calls I made)

- Treating K1669 scope-weakening as non-blocking because the verdict is KILLED (not SUPPORTED). If researcher had claimed SUPPORTED on K1669's back, this would be REVISE.
- Accepting `(m2)` soft-pass: the MLX code is idiomatic and matches repo patterns established in dependency's construction. No explicit `/mlx-dev` mention, but no anti-pattern triggered either.

## Recommendation to analyst

LEARNINGS.md should capture:
1. Gram-Schmidt on B during training is the critical path for N>1 privacy (confirms dependency #2).
2. Composition-as-defense decouple: verbatim canary extraction can PASS while MIA and subspace overlap FAIL. Membership inference ≠ verbatim recall under additive composition. This is behaviorally load-bearing for the v2 design.
3. Cross-user B cosine ~0.39 (from dependency) ⇒ ~58% subspace overlap under rank-r SVD of composed delta. Expected relationship; verifies theory.
