# MATH.md — exp_hedgehog_loss_variant_mse (PREEMPT-STRUCTURAL KILL)

**Status:** preempt-structural KILL before any MLX invocation. Five-lemma impossibility proof. **3rd triple-fire precedent** (1st=F#714, 2nd=F#716).

## §0 Platform-skill invocation (per reviewer.md m2 carve-out)
Preempt-structural path; no MLX code runs. Platform-skill carve-out applies per F#716 precedent. No `/mlx-dev`, no `/fast-mlx` needed.

## Claim under test
K1872: "MSE attention-map loss produces adapter with cos-sim < 0.70 vs cos-loss baseline."
Third Hedgehog loss-variant ablation. MSE directly on attention weights tests whether magnitude matters or only direction (vs cos-loss baseline per F#683 Hedgehog mechanism).

## Registered KCs (from `experiment get`)
- **K1872** (sole KC) — "MSE attention-map loss produces adapter with cos-sim < 0.70 vs cos-loss baseline"

Single proxy KC (cos-sim, guardrail 1007 canonical proxy list). Inter-variant comparison `MSE-trained vs cos-loss-trained`. **Zero target-metric KCs.** `depends_on = []` (standalone).

## Hygiene
- `success_criteria = []` (defect 1)
- `platform = ~` (defect 2)
- `references = []` (defect 3)
→ 3 defects = F#703 canonical 3-defect threshold.

## Antipattern scan
| Antipattern | Fires? | Role | Evidence |
|---|---|---|---|
| F#666-pure-standalone (Finding #700-line) | ✅ **primary** | KC-class-level | K1872 is cos-sim (guardrail 1007 canonical proxy). 0 target KCs. `depends_on=[]`. |
| §5 tautological-inter-variant-delta (F#709 promoted) | ✅ **secondary** | KC-form-level | K1872 inter-variant `MSE vs cos-loss baseline` without per-variant base-anchor (no `cos-sim(MSE, base)` or `cos-sim(cos-loss, base)` reference). |
| Hygiene-multi-defect (F#703) | ✅ **tertiary** | metadata | 3 defects; F#702 hygiene-patch path **unavailable** per promoted impossibility memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (F#716). |
| F#669-family (parent-target-FAIL inheritance) | ❌ | — | `depends_on=[]`; standalone. |
| Template-regression (F#705/F#708/F#709 promoted) | ❌ | — | No formal `depends_on` parent; kinship to F#719 (Hedgehog KL-div loss-variant) is **sibling** not parent — both are Hedgehog loss-variant-ablation sub-type, not parent/child. |
| Proxy-only-lineage-inheritance (F#710/F#711 watchlist) | ❌ | — | No parent. |
| Cross-paper-combined-loss-tautology (F#714 watchlist) | ❌ | — | Single-loss evaluation, no composite loss. |
| Novel-mechanism + hygiene-secondary (F#719 promoted pairing memory) | ❌ | — | Primary fire is F#666-pure (not novel-mechanism); pairing requires novel-mechanism primary per promoted memory. |

**Classification:** TRIPLE-FIRE. F#666-pure-standalone **primary**; §5 **secondary**; hygiene-multi-defect **tertiary with F#702 unavailable**. **3rd triple-fire precedent** (after F#714 K1847/K1848, F#716 K1864/K1865). Hierarchy formalized at F#714 (F#666-pure > §5 > hygiene-multi-defect) holds across a **4th distinct §5 axis** (inter-training F#714, intra-adapter-rank F#712/F#716, and now **intra-loss-function-delta** here — new sub-variant).

## Sub-type context (Hedgehog loss-variant-ablation)
This is the **2nd Hedgehog loss-variant-ablation instance** (1st = F#719 `exp_hedgehog_loss_variant_kl_div`, PROVISIONAL novel-mechanism + hygiene-patch-secondary). Critical structural distinction vs F#719:

| Instance | Loss variant | Target-KC paired? | Verdict | Reason |
|---|---|---|---|---|
| F#719 K1870+K1871 | KL-div (teacher‖student) | ✅ K1871 behavioral-quality Δ > 3 pp | PROVISIONAL novel-mech + hygiene-patch | F#666-compliant paired KC design |
| **This K1872** | MSE on attention weights | ❌ **cos-sim only** | **PREEMPT-KILL** | F#666-pure, strictly weaker KC design |

Same sub-type, **strictly weaker KC design** (per F#716 precedent classification vs F#712). Hard-defer rule on "all further Hedgehog-framework design-locks" (analyst F#719 guidance) does NOT block preempt-KILL: kill rejects the design, does not accept it — pile remains at 7 designs / 0 measurements unchanged.

## Lemmas

### L1 — Cos-sim-only admits no F#666-compliant verdict (K1872)
By guardrail 1007, cos-sim is an explicitly named proxy. For K1872, the 2-outcome truth table over verdict V ∈ {PASS, FAIL}:

- **PASS** (cos-sim ≥ 0.70 between MSE and cos-loss adapters): "MSE and cos-loss produce similar attention maps" — tautological-support on a pure geometric similarity; no behavioral conclusion admissible. Cos-sim↔behavioral-quality relation never established in this codebase for attention-map comparisons (r≈0.08 PPL↔task analogue holds for adjacent proxies).
- **FAIL** (cos-sim < 0.70): "MSE and cos-loss produce dissimilar attention maps" — per F#666/guardrail 1007, proxy-FAIL + target-PASS could = finding about the proxy (both losses may still produce behaviorally-equivalent adapters despite structural divergence); no kill admissible without paired target KC.

Both outcomes unidentifiable. ∎

### L2 — §5 fires on K1872 (no per-variant base-anchor)
K1872 is a comparison `cos-sim(MSE-adapter, cos-loss-adapter) < 0.70` without any `cos-sim(MSE-adapter, base)` or `cos-sim(cos-loss-adapter, base)` anchor. Under a regime where both variants collapse toward `base` (degenerate-equivalence), both adapters → near-identity Δ, cos-sim between them → 1.0, **trivially exceeds 0.70 threshold** (kill does not fire) — "support" by collapse.

Under F#477 parent-target-FAIL regime candidate for Gemma 4 r=6 Hedgehog adapters (K1226 adapted_acc 0.480 < 0.50 4-opt random baseline for untrained framework), collapse is the operative regime until a Hedgehog-framework `_impl` validates behavioral departure from base. F#683/F#684 are PROVISIONAL design-locks without measurement; no evidence that MSE or cos-loss Hedgehog adapters depart from base behaviorally.

This is §5 **7th instance**, **1st intra-loss-function-delta sub-variant** (distinct from: F#712/F#716 intra-adapter-rank-delta; F#714 inter-training-method; inter-adapter-delta variants). Sub-variant split at 1-instance (below F#711 conservative 3-split threshold); defer promotion. ∎

### L3 — Hygiene-multi-defect (F#703) fires but F#702 patch unavailable (4th instance, post-promotion)
3 defects (success_criteria, platform, references) = F#703 canonical 3-defect threshold. Per promoted standalone memory `mem-impossibility-f666pure-saturation-implies-f702-unavailable` (F#716-promoted), zero target KCs ⇒ F#702 hygiene-patch path **structurally unavailable**: patching references/platform/success_criteria leaves the cos-sim-only KC set with no target KC to patch around; adding a target KC changes experiment identity, not a patch.

**4th F#702-unavailability instance** (F#714 triple, F#715 double, F#716 triple, **this triple**). Post-promotion confirmation: the impossibility memory is stable across 4 instances and 2 fire-modes. No further promotion needed; anchor-append only. ∎

### L4 — Cos-sim bucket: 1st F#666-pure-standalone instance
F#666-pure-standalone bucket ledger post-F#716:
- PPL bucket: F#705, F#708, F#716 (3, **confirmed-recurrent**)
- Routing bucket: confirmed-recurrent per F#711
- Classification/FNR bucket: F#706 (1)
- Routing-match-rate bucket: F#707 (1)
- Infrastructure-benchmark bucket: F#715 (1)
- **Cos-sim bucket: this — 1st instance**

1st-instance at new bucket — **not** confirmed-recurrent. Bucket taxonomy remains stable post-F#715 (5 buckets); this adds a **6th bucket** at 1-instance. Per F#711 convention: bucket-level promotion requires ≥ 3 instances (necessary, not sufficient). No taxonomy refactor. ∎

### L5 — Standalone topology distinct from upstream antipatterns
- **NOT F#669-family**: `depends_on=[]`.
- **NOT template-regression**: no formal parent edge. F#719 (Hedgehog KL-div) is a **sibling** not parent — both are Hedgehog loss-variant-ablation sub-type; neither `depends_on` the other. Structurally weaker than F#719 (F#719 had target pairing; this does not) = sibling-with-worse-KC-design, not stripped-from-parent regression.
- **NOT proxy-only-lineage-inheritance**: no parent.
- **NOT cross-paper-combined-loss-tautology**: single-method evaluation (MSE alone), no composite loss.
- **NOT novel-mechanism + hygiene-secondary pairing (F#719 promoted memory)**: primary fire is F#666-pure, not novel-mechanism. Pairing memory response is specifically for novel-mechanism-primary — does not apply when primary is F#666-pure.
∎

## Consolidated verdict
All five lemmas independently preempt. Compute is unnecessary and — per F#711 policy — inadmissible: no MLX invocation, no adapter load, no cos-sim measurement. Artifact set is the graceful-failure stub.

## Predictions (testable at the structural level)

| # | Prediction | Basis |
|---|-----------|-------|
| P1 | F#666 2-outcome truth table yields tautological-PASS or proxy-only-FAIL for K1872 | L1 |
| P2 | §5 degenerate-equivalence regime (Gemma 4 r=6 F#477 collapse basin) trivially satisfies K1872 via cos-sim → 1.0 | L2 |
| P3 | F#702 hygiene-patch path structurally unavailable per promoted impossibility memory (4th confirmation) | L3 |
| P4 | Cos-sim bucket opens at 1-instance; not confirmed-recurrent; no taxonomy refactor | L4 |
| P5 | Standalone topology preserves distinctness from all upstream antipatterns including F#719 sibling | L5 |

## Unblock path
File `exp_hedgehog_loss_variant_mse_v2_target_paired`:
1. **Scope** to Hedgehog politeness-axis (reuse F#683 corpus + rubric) or another _impl-available Hedgehog axis — same transitive-blocker pattern as F#719.
2. **Metric** behavioral-quality judge Δ > 3 pp (same target as F#683/F#719 K1871) as **target-metric KC**; cos-sim retained as proxy diagnostic ("did MSE arm converge?").
3. **KC design** per-variant base-anchor + paired target:
   - `behavioral_quality_MSE ≥ behavioral_quality_base + 3pp` (target, per-variant base-anchor)
   - `behavioral_quality_MSE ≈ behavioral_quality_cos_loss` (inter-variant target delta, within-ε)
   - `cos-sim(MSE, cos_loss) ≥ 0.70` (secondary proxy sanity)
4. **Cite** F#719 (1st Hedgehog loss-variant-ablation with target pairing — direct sibling template) + F#683 (Hedgehog politeness, PROVISIONAL) + F#133 (paired-KC template).
5. Do NOT patch via `experiment update` (antipattern-u). File new experiment id.
6. **Transitive blocker**: F#683 or F#684 `_impl` must land before this v2 — same blocker as F#719 _impl.

## Assumptions logged (autonomy guardrail 1008)
- Cos-sim between attention maps is a proxy under guardrail 1007 (explicitly named on the canonical list); directional variations (cos-sim of weight matrices vs attention outputs vs intermediate activations) all fail L1 identically.
- "MSE attention-map loss" reads as MSE over softmax'd attention probabilities (the Hedgehog architectural output) rather than pre-softmax logits; §5 collapse argument holds under either interpretation.
- F#477 collapse-regime applicability to Hedgehog MSE adapters is the default pending any Hedgehog `_impl` measurement; if an `_impl` lands and shows departure, this assumption may need revision — but L2 is preempt-time structural reasoning.
- Platform field `~` counted as 1 hygiene defect per F#703 canon.
- `references = []` is **INCOMPLETE** per the F#702 CLI-limitation precedent; marked as hygiene defect for counting purposes but non-blocking for preempt-KILL verdict (per F#716 precedent — the kill stands regardless of references patching being available via `experiment ref-add`).

## mlx-lm version
Not invoked — preempt-structural path. No MLX code runs.
