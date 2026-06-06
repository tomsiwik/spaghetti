# MATH.md — T2.3: Local-Only vs All-Layer Adapter (Gemma 4 Dual Geometry)

**Experiment type:** Guided Exploration
**Platform:** Apple M5 Pro 48GB, MLX
**Date:** 2026-04-09

---

## Motivation

Gemma 4 E4B has a dual-geometry architecture: 35 **local** (sliding-window, 256-token context)
layers and 7 **global** (full-attention) layers, interleaved every 6 layers.
Confirmed by inspection: global layers at indices [5, 11, 17, 23, 29, 35, 41].

**Question:** Does domain knowledge (math notation, code syntax) reside primarily in
local layers, or do global layers also need to be adapted?

**Practical stakes:** If local-only (35 layers) is sufficient, we can:
- Leave global layers domain-agnostic (shared KV cache across domains)
- Reduce adapter params by 35/42 = 16.7% per domain

**Mathematical grounding:** Gemma 3 Technical Report (arxiv 2503.19786) establishes the
5:1 local-to-global ratio. Domain-specific features (terminology, syntax, notation) are
token-level patterns that fit within a 256-token sliding window. Global layers integrate
long-range discourse patterns, which are domain-agnostic structural features (paragraph
structure, argument flow). Therefore, LoRA adapters on local layers should capture the
domain signal; adapters on global layers should be redundant for single-domain specialization.

---

## Theorem 1: Local Layer Sufficiency

**Claim:** Let f = f_global ∘ f_local be the Gemma 4 E4B computation factored into
local-layer subnetwork f_local (35 layers) and global-layer subnetwork f_global (7 layers).
For domain adaptation tasks where the domain signal is token-local (fits within W=256 tokens),
LoRA adapters on f_local alone achieve quality >= 90% of all-layer adaptation.

**Proof sketch:**

Let D be a domain dataset where task-relevant features span at most W tokens
(math notation, code syntax, medical terminology are all token-local by construction).

A local attention layer l processes tokens within its window:
  h_l = LocalAttn_l(h_{l-1}) = Attn(Q_l, K_{window}, V_{window})

The domain-specific adaptation ΔW_q^{local} on Q_l modifies how queries attend to
in-window keys — sufficient to learn domain-specific attention patterns.

A global attention layer g processes all tokens:
  h_g = GlobalAttn_g(h_{g-1}) = Attn(Q_g, K_{all}, V_{all})

For token-local domains, the global attention's cross-context information is not
domain-specific. The global layers' role is context integration, not domain encoding.
Therefore, adapting global layers adds redundant capacity without new domain signal.

**QED** (prediction to be verified, not formal proof — this is guided exploration)

**Prediction (K1037):** local-only GSM8K >= 90% × all-layer GSM8K = 0.90 × 82% = 73.8%

---

## Theorem 2: Global-Only Insufficiency

**Claim:** LoRA adapters on only the 7 global layers achieve < 70% of all-layer quality.

**Proof sketch:**

Global layers (7/42 = 16.7%) have 7× fewer parameters available for adaptation:
  params(global-only) = 7 × 2 × r × d = 7/42 × params(all-layer)

More critically, global layers process full-attention representations that are built on
top of the local layers' output. If local layers' representations are not adapted to the
domain, the global layers receive domain-unaligned representations — they cannot compensate.

The feed-forward direction: local layers' Q projections determine what information
gets extracted from domain-specific tokens. Without adapting these, domain-specific
features are under-represented in the intermediate representations flowing into global layers.

**Prediction (K1038):** global-only GSM8K < 70% × all-layer GSM8K = 0.70 × 82% = 57.4%

---

## Theorem 3: Parameter Count (Analytical)

**Claim:** The parameter ratio is analytically exact.

Given LoRA r=6 on q_proj for all 42 layers:
  params(all-layer) = 42 × 2 × r × d_q = 42 × 2 × 6 × 2560 = 1,290,240
  (T2.1 measured: 1,247,232 — 3.3% discrepancy due to GQA dimension mismatch)

  params(local-only) = 35 × 2 × r × d_q = 35 × 2 × 6 × 2560 = 1,075,200
  → 35/42 = 83.3% of all-layer (K1039 threshold: 0.833 ± 0.01)

  params(global-only) = 7 × 2 × r × d_q = 7 × 2 × 6 × 2560 = 215,040
  → 7/42 = 16.7% of all-layer = 6× smaller

**Note:** The DB kill criteria stated "5x smaller" and "5x faster." This is **incorrect**.
- Local-only is 1.2× smaller than all-layer (not 5×)
- Global-only is 6× smaller than all-layer
The correct comparison is local-only vs global-only: 35/7 = 5× more layers for local.

---

## Architecture Reference

| Layer type | Count | Indices | Attention span |
|-----------|-------|---------|----------------|
| Local (sliding) | 35 | [0-4, 6-10, 12-16, 18-22, 24-28, 30-34, 36-40] | 256 tokens |
| Global (full) | 7 | [5, 11, 17, 23, 29, 35, 41] | Full context |
| **Total** | **42** | — | — |

---

## Kill Criteria

| K# | Criterion | Prediction | Type |
|----|-----------|-----------|------|
| K1037 | local-only GSM8K >= 90% × all-layer (73.8% threshold) | PASS | empirical |
| K1038 | global-only GSM8K < 70% × all-layer (57.4% threshold) | PASS | empirical |
| K1039 | local-only param ratio = 35/42 = 0.833 ± 0.01 | PASS | analytical |

**All-layer baseline (T2.1):** 82% GSM8K, 1000 steps, 1.25M params

---

## Implementation Plan

1. **Reuse:** T2.1 all-layer result (82% GSM8K) — no additional training needed
2. **Reuse:** T2.1 math training data at `exp_p1_t2_single_domain_training/data/math/`
3. **Local-only training:**
   - Load Gemma 4 E4B
   - Apply LoRA (r=6, q_proj) to all 42 layers
   - Freeze q_proj LoRA weights on global layers [5,11,17,23,29,35,41]
   - Train 1000 steps on GSM8K
   - Save adapter, eval on 50 examples
4. **Global-only training:**
   - Load Gemma 4 E4B
   - Apply LoRA (r=6, q_proj) to all 42 layers
   - Freeze q_proj LoRA weights on local layers [all 35 local indices]
   - Train 1000 steps on GSM8K
   - Save adapter, eval on 50 examples
5. **Compare all three:** K1037, K1038, K1039

---

## Prior Work

- Finding #421 (T2.1): All-layer (42) LoRA r=6 on q_proj → 82% GSM8K in 22 min
- Finding #412 (T0.4): q_proj is the primary adaptation bottleneck (K/V sharing confirmed)
- Gemma 3 Technical Report (2503.19786): 5:1 local/global architecture, dual RoPE
- HRA paper (2405.17484): domain knowledge primarily in lower (local) layers of transformers
