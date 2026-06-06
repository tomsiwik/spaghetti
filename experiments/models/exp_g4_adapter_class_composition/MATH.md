# exp_g4_adapter_class_composition — Measurement proxy

## Scope and honest reframe

The DB title states: *"Class A (LoRA) beats Class B (DoRA, MoLoRA) on Gemma 4 MMLU-Pro composition by >=3pp at N=5."*

The full form — training 3 adapter classes × 5 domains × Gemma 4 E4B + MMLU-Pro eval at N=5 — requires 15 training runs (each >30 min) + composition eval and exceeds the single-iteration researcher budget. More importantly, DoRA requires a *trained* magnitude vector `m` and MoLoRA requires *trained* router weights; neither can be synthesized from existing LoRAs without training.

This experiment runs a **measurement-only proxy** on existing Gemma 4 E4B 4-bit LoRA adapters (`exp_p1_t2_single_domain_training`: code, math, medical — r=6, self_attn.q_proj). The proxy measures **composition-geometry interference** — the quantity F#82 predicts separates class A (LoRA, linear/additive) from class B (DoRA/MoLoRA, caveated).

**This proxy does not test the 3pp MMLU gap.** It tests whether the *geometric cause* cited by F#82 — additive vs non-additive composition — shows up at Gemma 4 E4B scale.

## Motivation

- **Finding #82** (conclusive, 2026-03-11): FIT=0.875, 15 adapter types, 3 composition classes surveyed on micro-d models. LoRA = class A (additive), DoRA/MoLoRA = class B (caveated/non-additive). Proof at d=64, r=8; unverified at Gemma 4 E4B scale (d_hidden=2048, r=6).
- **Arxiv 2402.09353** (Liu et al, DoRA): `W' = m · (W + ΔW) / ||W + ΔW||_c`. Composition of N DoRA updates is **not** sum of ΔW because the column-wise normalization is nonlinear in ΣΔW.
- **Arxiv 2402.11260** (Feng et al, MoLoRA): `ΔW_total = Σ_i g_i · B_i A_i` where `g_i` is a learned router. Even with uniform gates `g_i = 1/N`, scaling breaks strict additivity of unit-scale LoRAs.

## Type: verification (measurement proxy on existing adapters)

## Theorem (class ordering by composition deviation)

**Statement.** Let `ΔW_i = α · B_i A_i` be N trained LoRA updates on a shared base `W_0`. Define:

- **LoRA composition**: `L(v) = Σ_i ΔW_i v`  (class A, additive).
- **Pseudo-DoRA composition**: `D(v) = m_d · (W_0 + Σ_i ΔW_i) v / ||W_0 + Σ_i ΔW_i||_c − W_0 v`, where `m_d = ||W_0 + Σ_i ΔW_i||_c` frozen at composition time is exactly LoRA, but with `m_d = ||W_0||_c` (the DoRA init, which is what forces the renormalization to be effective) gives a non-trivial deviation.
- **Pseudo-MoLoRA composition**: `M(v) = (1/N) · Σ_i ΔW_i v`  (uniform-gated mixture at init; class B via the router's presence even if untrained, since it rescales).

Composition-deviation from pure additive for class C:
`dev_C = || C(v) − L(v) || / (|| L(v) || + ε)`.

**Prediction.** On Gemma 4 E4B 4-bit q_proj with N=3 (code/math/medical):

1. `dev_LoRA = 0` exactly (trivially — definition).
2. `dev_DoRA > 0` (magnitude renormalization breaks additivity).
3. `dev_MoLoRA = |1 − 1/N| = 2/3 ≈ 0.667` exactly (uniform-gated rescale).

The class ordering `dev_LoRA < dev_DoRA` tests whether DoRA's nonlinearity is **non-trivial** at Gemma 4 scale (could be vanishing if `||ΔW|| << ||W_0||`).

**Proof sketch for (2).** `W_0 + ΣΔW` has column norm `||W_0||·(1 + ε_c)` where `ε_c = O(||ΔW||/||W_0||)` per column. The DoRA renormalization `m_d · (·) / ||·||_c` with `m_d = ||W_0||_c` gives `(W_0 + ΣΔW) / (1 + ε_c)` per column, differing from `W_0 + ΣΔW` by `−ε_c · (W_0 + ΣΔW) / (1 + ε_c)`. The deviation `dev_DoRA = ||·|| / ||ΣΔW·v||` is bounded below by `min_c |ε_c|` when `W_0 v` dominates `ΣΔW v`, and is typically `>> 0` because ε_c varies per column. QED-sketch.

## Pre-registered kill criteria (target-gated per F#666)

**K1 (structural).** All three class compositions produce finite, non-NaN outputs on ≥95% of (layer, probe) pairs across 10 probes × 10 sampled q_proj layers of Gemma 4 E4B 4-bit. PASS iff success rate ≥0.95.

**K2 (target).** Class-ordering prediction: `median(dev_DoRA) > median(dev_LoRA) + 10⁻⁴` AND `median(dev_MoLoRA) > median(dev_LoRA) + 10⁻⁴`. PASS iff both hold with finite medians.

Per Finding #666, both KCs must be target-gated pairs:
- K1 structural-PASS AND K2 target-PASS → class geometric ordering **holds** at Gemma 4 scale: measurement supports F#82's class-A-beats-class-B mechanism (composition-geometry level, does not directly imply the 3pp MMLU margin).
- K1 PASS + K2 FAIL → ordering breaks at Gemma 4 scale: the F#82 mechanism may not carry; MMLU-level experiment would be required to decide.
- K1 FAIL → inconclusive (numerical issues).

## Assumptions

1. Existing `exp_p1_t2_single_domain_training` adapters are valid Gemma 4 E4B 4-bit q_proj LoRA at r=6, scale=6 (verified: adapter_config.json shows exactly this).
2. Measurement proxy on q_proj weights generalizes to full attention composition (q_proj-only is a known limitation inherited from the source experiment).
3. Pseudo-DoRA with `m_d = ||W_0||_c` is a conservative lower bound on trained-DoRA deviation (trained `m_d` drifts from init, usually increasing deviation further).

## What this does NOT claim

- Does not measure MMLU-Pro quality (no training of fresh DoRA/MoLoRA).
- Does not verify the 3pp aggregate gap.
- Does not cover N=5 (uses N=3 because only 3 domain adapters exist).

The composition-geometry signal is reported as a proxy; full MMLU-Pro verification remains open for a future macro-scale iteration.
