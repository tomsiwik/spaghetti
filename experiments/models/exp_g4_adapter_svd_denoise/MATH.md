# MATH.md — exp_g4_adapter_svd_denoise (PREEMPT-STRUCTURAL KILL)

**Status:** preempt-structural KILL before any MLX invocation. Five-lemma impossibility proof.

## Claim under test
SVD denoise: "adapters have a noise floor in small singular values; truncating below threshold improves composition by reducing interference."

## Registered KCs (from `experiment get`)
- **K1864** — "SVD-truncated adapter PPL > original adapter PPL + 0.05 (truncation removes signal)"
- **K1865** — "Truncated adapter composition PPL not improved vs untruncated composition"

Both KCs are **PPL-only**, variant-vs-variant comparisons, with **no target-metric pairing** (no MMLU/HumanEval/GSM8K behavioral anchor) and **no per-variant base-anchor** (neither references `PPL_base`). `depends_on = []` (standalone).

## Hygiene
- `success_criteria = []` (defect 1)
- `platform = ~` (defect 2)
- `references = []` (defect 3)
→ 3 defects ≥ F#703 canonical 3-defect threshold.

## Antipattern scan
| Antipattern | Fires? | Role | Evidence |
|---|---|---|---|
| F#666-pure-standalone (Finding #700-line) | ✅ **primary** | KC-class-level | Both K1864/K1865 are PPL (guardrail 1007 canonical proxy list). 0 target KCs. depends_on=[]. |
| §5 tautological-inter-variant-delta (F#709 promoted) | ✅ **secondary** | KC-form-level | K1864 inter-variant `truncated vs original`, K1865 inter-variant `truncated-composition vs untruncated-composition`, both without per-variant base-anchor. |
| Hygiene-multi-defect (F#703) | ✅ **tertiary** | metadata | 3 defects; but F#702 hygiene-patch path **unavailable** (zero target KCs cannot be "fixed" by adding references alone). |
| F#669-family (parent-target-FAIL inheritance) | ❌ | — | `depends_on=[]`; standalone. |
| Template-regression (F#705/F#708/F#709 promoted) | ❌ | — | No formal `depends_on` parent; kinship to F#712 `exp_g4_svd_truncate_adapter` is semantic not structural. |
| Proxy-only-lineage-inheritance (F#710/F#711 watchlist) | ❌ | — | No parent. |
| Cross-paper-combined-loss-tautology (F#714 watchlist) | ❌ | — | No composite loss. |

**Classification:** TRIPLE-FIRE. F#666-pure-standalone **primary**; §5 **secondary**; hygiene-multi-defect **tertiary with F#702 unavailable**. Same topology as F#714 (first triple-fire precedent) and structurally isomorphic to F#715's F#702-unavailability sub-case (but with §5 additionally firing).

## Lemmas

### L1 — PPL-only admits no F#666-compliant verdict (both KCs)
By guardrail 1007, PPL is an explicitly named proxy. For either K1864 or K1865, the 2-outcome truth table over verdict V ∈ {PASS, FAIL}:

- **PASS**: "SVD denoise preserved PPL" — tautological-support. PPL↔task-quality r ≈ 0.08 in this codebase (Part 1 project memory); no behavioral conclusion admissible.
- **FAIL**: "SVD denoise changed PPL beyond threshold" — by F#666/guardrail 1007, "proxy-FAIL + target-PASS = finding about the proxy, not a kill"; no kill admissible without paired target KC.

Both outcomes unidentifiable for both KCs. ∎

### L2 — §5 fires on both KCs (no per-variant base-anchor)
K1864 is a comparison `PPL_truncated − PPL_original > 0.05` without any `PPL_original vs PPL_base` or `PPL_truncated vs PPL_base` anchor. Under a regime where both variants collapse to `PPL_base` (degenerate-equivalence), `|ΔPPL| ≈ 0` passes trivially.

K1865 is `PPL_truncated_composition ≤ PPL_untruncated_composition`; under F#477 parent-target-FAIL regime for Gemma 4 r=6 adapters (K1226 adapted_acc 0.480 < 0.50 random 4-opt chance), composition of rank-truncated adapters sits in the same collapse basin — K1865 "not improved" holds in the degenerate regime with zero denoising-mechanism established.

This is §5 2nd instance of **intra-adapter-rank-delta** sub-variant (1st was F#712 K1611). §5 clause stable post-promotion (F#709). ∎

### L3 — Hygiene-multi-defect (F#703) fires but F#702 patch unavailable
3 defects (success_criteria, platform, references) ≥ F#703 canonical 3-defect threshold. However, per F#714/F#715, with zero target KCs the F#702 hygiene-patch path is **structurally unavailable**: you cannot "patch" missing references into existence without also designing new target-metric KCs — which changes the experiment identity, not a patch. 3rd F#702-unavailability instance (after F#714, F#715) → promotes impossibility-structure "F#666-pure-saturation (0 target KCs) ⇒ F#702-patch unavailable" to promotion-threshold per F#715 analyst note. ∎

### L4 — PPL bucket saturates at 3rd instance
F#666-pure-standalone PPL bucket instances:
1. F#705 — `exp_g4_o1_removal_naive` (1st)
2. F#708 — `exp_g4_hash_ring_remove_n25` (2nd, confirmed-recurrent)
3. **This — `exp_g4_adapter_svd_denoise` (3rd)**

3-instance reaches the F#711 taxonomy-refactor threshold for sub-bucket promotion: PPL is now the 2nd confirmed-recurrent bucket (after routing at 3-instance). Bucket taxonomy (5 total post-F#715) remains the correct level of abstraction; no super-category promotion needed per the F#711 convention. ∎

### L5 — Standalone topology distinct from upstream antipatterns
- NOT F#669-family: `depends_on=[]`.
- NOT template-regression: no formal parent edge; F#712 (the semantic cousin) uses MMLU-Pro target metric with §5 intra-rank-delta; this one lacks the target metric entirely, i.e. a **strictly weaker KC design**, not a stripped-from-parent regression.
- NOT proxy-only-lineage-inheritance: no parent.
- NOT cross-paper-combined-loss-tautology: single-method evaluation, no composite loss.
∎

## Consolidated verdict
All five lemmas independently preempt. Compute is unnecessary and — per F#711 policy — inadmissible: no MLX invocation, no adapter load, no PPL measurement. Artifact set is the graceful-failure stub.

## Predictions (testable at the structural level)
| # | Prediction | Basis |
|---|-----------|-------|
| P1 | F#666 2-outcome truth table yields tautological-PASS or proxy-only-FAIL for both KCs | L1 |
| P2 | §5 degenerate-equivalence regime trivially satisfies both K1864 and K1865 | L2 |
| P3 | F#702 hygiene-patch path unavailable due to zero target KCs | L3 |
| P4 | PPL bucket saturates at 3-instance; no bucket-taxonomy refactor needed | L4 |
| P5 | Distinctness from all upstream antipatterns is preserved | L5 |

## Unblock path
File `exp_g4_adapter_svd_denoise_v2_target_paired`:
1. **Scope** to 3 trained Gemma 4 r=6 domain adapters (code / math / medical, per F#627 SUPPORTED).
2. **Metric** per-domain task accuracy (HumanEval / GSM8K / MedQA) as the **target-metric KC**, with PPL retained only as a secondary sanity gate.
3. **KC design** per-variant base-anchor + paired target: `acc_truncated ≥ acc_base − ε` AND `acc_truncated ≥ acc_untruncated − δ`, each paired with the dual PPL sanity KC per F#133 template.
4. **Cite** F#712 (SVD intra-rank §5 preempt-KILL) + F#627 (adapter target config) + F#133 (paired-KC template).
5. Do NOT patch via `experiment update` (antipattern-u).

## Assumptions logged (autonomy guardrail 1008)
- PPL is a proxy under guardrail 1007 regardless of how it's computed (token-level vs sequence-level) — both interpretations fail L1.
- K1865 "not improved" reads as `PPL_truncated_composed ≤ PPL_untruncated_composed + ε` for small ε; both directional readings (<, ≤) fail §5 under degenerate-equivalence.
- The composition referred to in K1865 is `Σ B_i A_i` per audit-corrected composition math (antipattern-001); L2 holds under either `ΣB·ΣA` (audit-bug) or `ΣBA` (correct) composition because both collapse under F#477.
- Platform field `~` is treated as 1 hygiene defect (missing) per F#703 canon.

## mlx-lm version
Not invoked — preempt-structural path. No MLX code runs.
