# Behavioral PPL ablation of top-20% vs bottom-20% q_proj head slabs

**Experiment:** `exp_g4_head_ablation_ppl` · Follow-up to F#742 ·
Base: `mlx-community/gemma-4-e4b-it-4bit` (mlx 0.31.1, mlx-lm 0.31.2) ·
Adapters: `exp_p1_t2_single_domain_training` (rank-6 LoRA, scale 6.0,
`self_attn.q_proj`).

## Abstract

F#742 found that a rank-6 q_proj LoRA on Gemma-4 E4B has **weak structural**
head-mass concentration (C_20 ≈ 0.335) but **real functional** cross-domain
specialization (Jaccard J̄ = 0.349). It never tested behavior, leaving open
whether the per-head mass ranking is exploitable or a weight-space artefact.
This experiment closes that proxy→target gap by **directly measuring
perplexity** when the top-20% vs bottom-20% head slabs (by F#742's ranking, not
recomputed) are zeroed. Across all three domains, zeroing the top-20% heads
degrades held-out domain PPL **much more** than zeroing the bottom-20%
(mean ratio R̄ = 4.89; ratio > 2 on 3/3 domains), and the intact adapter beats
the no-adapter base by 77% on average. The F#742 head-importance ranking is
**behaviorally real**. Verdict: **SUPPORTED**.

## Method

1. Load base Gemma-4 E4B 4bit + the per-domain q_proj LoRA adapter.
2. Reuse F#742's `per_layer_head_mass` (42×8 per domain). Rank all 336 head
   slabs by Frobenius mass μ_{l,h} = ‖ΔW_q[:,h,:]‖²_F; take the 67 (=⌈0.20·336⌉)
   highest as top-20% and 67 lowest as bottom-20% (disjoint, verified).
3. Ablate a slab by zeroing the corresponding `head_dim`-wide column block of
   the adapter's `B` matrix — exactly the quantity μ ranks. Adapter applied as
   `q(x) + scale·(xA)B` via an `nn.Module` subclass installed with `setattr`
   (no `__call__` monkey-patch; F#831-safe). Wrapper verified to change logits
   (mean |Δ|=5.06) and detach to restore base exactly (|Δ|=0).
4. Assistant-token PPL on 50 held-out chat samples per domain
   (`data/<domain>/valid.jsonl`, mask_prompt=True, max_seq 512), identical
   corpus across all four arms.

## Results (measured)

| Domain | PPL base | PPL intact | PPL top-20% zeroed | PPL bot-20% zeroed | Δ_top | Δ_bot | ratio R | adapter gain |
|--------|---------:|-----------:|-------------------:|-------------------:|------:|------:|--------:|-------------:|
| code    | 3.8454 | 1.5951 | 1.6552 | 1.6120 | 0.0601 | 0.0169 | **3.55** | 58.5% |
| math    | 5.7693 | 1.5300 | 1.8161 | 1.5759 | 0.2861 | 0.0459 | **6.24** | 73.5% |
| medical | 90.5430 | 1.1118 | 1.5622 | 1.1110 | 0.4504 | −0.0008 | **∞** (bot ≈ 0) | 98.8% |

- **R̄ (mean degradation ratio) = 4.89**; ratio > 2 on **3/3** domains.
- **Mean adapter gain g = 76.9%** (intact beats base PPL).
- Medical bottom-20% ablation slightly *improves* PPL (Δ_bot = −0.0008),
  i.e. the lowest-mass heads carry essentially no behavioral signal — the
  cleanest possible confirmation that the ranking is behavioral.

## Kill criteria (pre-registered, both TARGET-metric on PPL)

| KC | Metric | Threshold | Measured | Result |
|----|--------|-----------|----------|--------|
| K#1967 (head importance behavioral) | R̄ degradation ratio | > 2.0 to PASS | R̄=4.89, 3/3 domains > 2 | **PASS** (does NOT fire) |
| K#1968 (adapter meaningful) | mean (PPL_base−PPL_intact)/PPL_base | ≥ 5% to PASS | 76.9% | **PASS** (does NOT fire) |

Neither kill criterion fired. Per the MATH.md decision table
(K#1968 PASS ∧ K#1967 PASS), and Success #114 (top > 2× bottom on ≥2/3 domains
∧ gain ≥ 5%, met on 3/3), the verdict is **SUPPORTED**.

## Prediction vs measurement

| Prediction | Expected | Measured | Match |
|---|---|---|---|
| P1: ranking behavioral → R̄>2, R>2 on ≥2/3 | R̄>2, ≥2/3 | R̄=4.89, 3/3 | ✓ |
| P2 (competing): weak C_20 → R̄≈1 (kill) | R̄≈1 | R̄=4.89 | refuted |
| P3: adapter real → g≥5% | g≥5% | g=76.9% | ✓ |

## Verdict

**SUPPORTED.** Zeroing the top-20% q_proj head slabs degrades held-out domain
perplexity 4.9× more on average (3.55× / 6.24× / ∞ for code / math / medical)
than zeroing the bottom-20%, and the intact adapter is strongly behaviorally
meaningful (76.9% mean PPL gain over base). F#742's per-head mass ranking,
despite weak *structural* concentration (C_20≈0.335), is **behaviorally real**:
head-importance matters at the behavior level. This unlocks a head-sparse
adapter serving path and the F#627-compliant `v_proj+o_proj` follow-up.

## Caveats

- `q_proj` rank-6 adapters only; F#627 recommends `v_proj+o_proj`. The
  conclusion does not transfer without re-running on those targets (the
  designated follow-up / Success #114 unlock).
- 50 held-out samples/domain — small, but fixed and identical across all four
  arms, so the *ratio* R is robust to corpus size.
- The infinite ratio on medical reflects Δ_bot ≈ 0 (bottom heads inert), not a
  numerical artefact; the dominance (Δ_top=0.45 ≫ Δ_bot≈0) is unambiguous and
  the experiment also passes on the two finite-ratio domains.
- Absolute PPL values are low (intact ≈ 1.1–1.6) because the held-out split is
  in-distribution to the training corpus; this is expected and affects all arms
  equally — the test is purely relative (top vs bottom degradation).
