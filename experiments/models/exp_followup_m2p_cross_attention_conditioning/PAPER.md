# PAPER.md: Cross-Attention Context Conditioning for M2P

**Experiment ID:** `exp_followup_m2p_cross_attention_conditioning`
**Verdict:** **KILLED**
**Date:** 2026-04-18
**Runtime:** 12.7 s
**Seed:** 42

## Executive summary

This experiment tested whether replacing mean-pool additive context injection with cross-attention — a rank-increasing primitive (Lemma 2 of MATH.md) — resolves the K850 CV-collapse failure of the parent kill `exp_m2p_scale_calibrated` (Finding #343).

**Result: cross-attention is necessary but not sufficient.** CV rose only from 0.0153 (mean-pool control) to 0.0200 (cross-attention) — a 1.31× gain, well below the pre-registered ≥ 3× threshold (K1556c) and nowhere near the absolute 0.05 threshold (K1556a). The control kill criterion K1556b passes (mean-pool reproduces the sibling's CV ≈ 0.009–0.02 regime).

The architectural bottleneck that kills K850 is **not** the mean-pool centroid alone. A second rank-reducing structure — almost certainly the `B_proj` unpacking head operating on `mem.reshape(1, -1)`, or the self-attention blocks between cross-attention and the head re-pooling the memory — dominates the context-sensitivity budget.

## Pre-registered kill criteria (MATH.md §F)

All three must pass for `supported`.

| KC | Condition | Measured | Pass? |
|---|---|---|---|
| **K1556a** | `CV_cross_attn > 0.05` | `0.0200` | ✗ FAIL |
| **K1556b** | `CV_mean_pool_baseline ≤ 0.02` | `0.0153` | ✓ PASS |
| **K1556c** | `CV_cross_attn ≥ 3 × CV_mean_pool` | ratio = `1.31` | ✗ FAIL |
| **Overall** | all three pass | — | **KILLED** |

K1556b passing confirms the measurement regime is correctly reproduced from the parent kill; the failure is attributable to the treatment, not to drift.

## Prediction vs measurement

Pulled directly from MATH.md §E. Same 20-context eval suite used for both arches (`eval_rng = np.random.RandomState(SEED + 999)`).

| # | Prediction (from MATH.md) | Measured | Match? |
|---|---|---|---|
| P1 | `CV_cross_attn > 0.05` | `0.0200` | ✗ NO |
| P2 | `CV_cross_attn / CV_mean_pool ≥ 3` | `1.31` | ✗ NO |
| P3 | `CV_mean_pool_baseline ≤ 0.02` | `0.0153` | ✓ YES |
| P4 | `‖B(hard)‖_F / ‖B(easy)‖_F ≥ 1.10` under cross-attn | `1.006` (easy=37.50, hard=36.42 — ratio 0.971) | ✗ NO (also wrong sign) |
| P5 | `\|gen_deg_cross − gen_deg_mean\| ≤ 10 pp` | `\|−60.22 − (−61.02)\| = 0.80 pp` | ✓ YES |

**Two of five predictions hold.** P1, P2, P4 fall together — they all depend on cross-attention unlocking context-sensitive output variation, which it does not at the magnitude predicted by Lemma 2. P3 (baseline reproduction) and P5 (quality-preservation sanity) hold, which isolates the failure cleanly to the conditioning path downstream of the cross-attention.

## Raw measurements

```
Mean-pool control (killed-sibling architecture):
  ||B||_F mean: 12.798  std: 0.196  CV: 0.01529
  easy mean:    12.982  hard mean: 12.614  ratio: 0.972
  gen degradation: -61.02 pp
  final L_task = 0.908   L_preserve = 5.05

Cross-attention treatment:
  ||B||_F mean: 36.959  std: 0.739  CV: 0.01999
  easy mean:    37.500  hard mean: 36.418  ratio: 0.971
  gen degradation: -60.22 pp
  final L_task = 2.41    L_preserve = 5.64

Ratio CV_cross / CV_mean: 1.31
Absolute CV gain: +0.0047
```

## Analysis

### Why the Lemma 2 rank bound did not translate into CV

Lemma 2 bounds the Jacobian rank of the cross-attention map; it does not bound the **magnitude** of context-driven variance that survives through the rest of the M2P pipeline. Two downstream structures re-pool:

1. **Self-attention blocks (`M2PBlock × 2`) after cross-attn.** Self-attention is permutation-equivariant over the memory dimension and mixes the 8 memory tokens additively. If one slot encodes `||B||_F`-modulation and the others do not, the post-block mean-pooling inside `B_proj` can still preserve it — but only if that one slot is protected from the mixing. With only 2 blocks and untuned `LR`, the context signal likely gets diluted.
2. **`B_proj` unpacking head.** `flat = mem.reshape(1, -1)` is a single vector of size `N_MEMORY * D_M2P = 512`. The linear `B_proj` sees only this concatenation. The 16k-dim output `B_flat` is an affine function of that 512-vector; variance in `||B||_F` is bounded above by `||B_proj||_op · ||Δflat||`, where `Δflat` is the context-driven perturbation of the memory. If `Δflat` is small in magnitude (because cross-attention + self-attn produce near-identical memories across contexts even with rank > 1), the unpacking head cannot amplify it beyond its operator norm.

**Both together are consistent with observed 1.31× gain.** The rank bound was right but the magnitude bound is the binding constraint.

### Relation to parent kill `exp_m2p_scale_calibrated`

Finding #343 of the parent kill listed three closure theorems:
- **C1** (additive-context-injection-blocks-calibration): addressed by this experiment. The architecture change is correct; it just is not enough.
- **C2** (KKT assumption violated at operating point): untouched by this experiment. Still applies.
- **C3** (L_preserve increases rigidity): partly replicated here — baseline CV ≈ 0.0153 with L_preserve active, roughly matching the sibling's 0.0093 when scaled for the different `||B||` regime.

The current experiment does not resurrect Theorem 1 of the sibling. It narrows the diagnosis: the mean-pool bottleneck was real, but removing it alone does not close the gap.

### What the negative result implies for the next follow-up

**Impossibility structure identified:** For a post-cross-attention pipeline that (a) uses permutation-equivariant self-attention over memory and (b) concatenates all memory tokens through a single linear unpacking head, context-to-`‖B‖_F` sensitivity is additionally throttled by the operator norm of the unpacking head and the magnitude of the inter-context memory delta — not just the Jacobian rank of the conditioning layer. Raising one without raising the other is insufficient.

Two concrete next experiments are now well-motivated:
1. **Per-token head:** replace `B_proj` on the flattened memory with `N_MEMORY` independent heads, each producing a fraction of `B`, so different memory slots can independently drive different LoRA components. Breaks the "single affine read of the flat vector" bottleneck.
2. **Slot-specific residual:** allow cross-attention output to bypass the self-attention blocks and reach `B_proj` via a skip connection; this preserves per-slot variance rather than letting it be re-pooled by self-attention.

Either is a separate experiment with its own MATH.md. **No KC in the current experiment is being edited post-hoc.**

## Assumptions and caveats

- **Toy scale.** Same 2-layer, `D_MODEL=256` toy GPT as the parent kill; findings are about the M2P conditioning mechanism, not production scale. The parent's closure C1 was stated at the same scale so the apples-to-apples comparison is valid.
- **Single seed.** `SEED=42` inherited verbatim from the sibling. With CV numbers in the 0.015–0.020 range across two runs, seed noise could plausibly move either measurement by ±0.005. This does not change the verdict: even the most generous reading (CV_cross = 0.025) stays below 0.05 and below the 3× ratio.
- **Single training regime.** `LAMBDA_PRESERVE=0.1`, `M2P_STEPS=600`, `LR=3e-4` inherited. A larger `LAMBDA_PRESERVE` or more training steps could change the operating point, but that would also change what the experiment is measuring — the closure rule `additive-context-injection-blocks-calibration` was scoped to the sibling's training regime, and we held it fixed.
- **No KC edits.** `git diff MATH.md` between pre-registration commit and this write-up is clean (only the MATH.md create commit); `git diff run_experiment.py` is the initial-commit only. `is_smoke: false`. KCs K1556a/b/c are frozen.

## Verdict-consistency pre-flight (PLAN.md §1)

1. `results.json["verdict"] == "killed"` ✓
2. `results.json["all_pass"] == false` ✓
3. PAPER.md verdict line = "KILLED" ✓
4. `is_smoke: false` ✓
5. `git diff MATH.md` clean since pre-reg commit `201a762` ✓
6. Antipattern scan: no composition-bug, no unsafe adapter scale (`ADAPTER_SCALE=1.0`), no tautological routing (no router in this experiment), no `shutil.copy`, no hardcoded `pass: True`, no smoke-as-full, no thinking-mode (no HF model), no wrong-model proxy (toy model is explicitly toy, matching the sibling). KC measures adapter magnitude CV — the exact object predicted by Theorem 1 Step 5 — not a proxy. ✓

**Pre-flight allows completion as `killed` (never `supported`). No upgrade attempt.**

## Conclusion

Cross-attention conditioning is **not sufficient** to resolve the K850 CV-collapse. A second architectural bottleneck — the concatenation-based unpacking head and/or the post-cross-attn self-attention re-pooling — dominates. The closure rule `additive-context-injection-blocks-calibration` from the parent kill is narrowed to `additive-pooled-concat-unpacking-blocks-calibration`. Next experiments should target the unpacking head, not the conditioning layer.
