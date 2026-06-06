# PAPER.md — exp_g4_adapter_svd_denoise

**Verdict:** KILLED (preempt-structural, pre-measurement)

**Antipatterns fired (triple-fire, 2nd precedent after F#714):**
1. **F#666-pure-standalone (primary, 12th drain-window instance, 3rd PPL-bucket instance)** — both KCs PPL-only, zero target pairing, `depends_on=[]`.
2. **§5 tautological-inter-variant-delta (secondary, 6th §5 instance, 2nd intra-adapter-rank-delta sub-variant)** — K1864/K1865 comparisons without per-variant base-anchor.
3. **Hygiene-multi-defect (tertiary, 3 defects)** — F#702 patch path unavailable (0 target KCs → 3rd F#702-unavailability confirmation after F#714, F#715).

## Claim
K1864 + K1865: SVD denoising of adapter deltas via singular-value truncation should *preserve* adapter PPL and *improve* composed PPL.

## Prediction-vs-measurement

| # | Prediction | Basis | Measured | Verdict |
|---|------------|-------|----------|---------|
| P1 | F#666 2-outcome truth table yields tautological-PASS or proxy-only-FAIL for both KCs | L1 (guardrail 1007 names PPL as proxy; PPL↔task r≈0.08 in repo) | Not measured (preempt) | Structural — both classes unidentifiable |
| P2 | §5 degenerate-equivalence trivially satisfies K1864 and K1865 | L2 (F#477 Gemma 4 r=6 K1226 adapted_acc 0.480 < 0.50 4-opt random) | Not measured (preempt) | Structural — collapse regime |
| P3 | F#702 hygiene-patch path unavailable due to zero target KCs | L3 (3rd confirmation, F#714/F#715 precedent) | Confirmed at MATH.md | Structural — promotion candidate at 3rd |
| P4 | PPL bucket saturates at 3-instance; no bucket-taxonomy refactor needed | L4 (F#711 refactor threshold ≥ 3-instance per bucket is necessary-not-sufficient) | Confirmed at MATH.md | Taxonomy stable |
| P5 | Distinctness from F#669-family / template-regression / proxy-only-lineage / F#714 watchlist preserved | L5 | Confirmed at MATH.md | No cross-contamination |

## Summary
K1864 ("truncated − original > 0.05 PPL") and K1865 ("truncated composition ≤ untruncated composition PPL") are both single-proxy PPL comparisons over adapter variants. No MMLU-Pro, HumanEval, GSM8K, or any behavioral target is paired. `depends_on=[]` — standalone. 3 hygiene defects (success_criteria, platform, references).

Per MATH.md L1, PPL is explicitly a proxy under guardrail 1007; both 2-outcome verdicts (PASS=tautological-support, FAIL=finding-about-the-proxy-not-a-kill) are unidentifiable. Per L2, §5 fires on both KCs because neither anchors to `PPL_base`; F#477 parent-target-FAIL regime places Gemma 4 r=6 adapters in the collapse basin where rank-truncated variants approach `PPL_base`, trivially satisfying both delta KCs. Per L3, hygiene-multi-defect is tertiary but F#702 patch path is **structurally unavailable** when target KCs = 0 — this is the **3rd instance** of the F#666-pure-saturation ⇒ F#702-unavailability impossibility structure, reaching the 3-instance promotion threshold per F#715 analyst note. Per L4, the PPL bucket saturates at 3-instance (F#705 / F#708 / this) and is now **confirmed-recurrent** per F#711 bucket-level convention — analog to routing bucket at 3-instance. Per L5, topology is distinct from all upstream antipatterns.

Triple-fire (F#666-pure + §5 + hygiene-multi-defect) is the **2nd precedent** after F#714 (`exp_sigreg_hedgehog_combined`). Differs from F#715 (double-fire; no §5) and from F#714 (different §5 axis — F#714 was inter-training-method, this is intra-adapter-rank-truncation). Confirms the triple-fire mode is recurrent and follows the hierarchy formalized at F#714: F#666-pure (KC class) > §5 (KC form) > hygiene-multi-defect (metadata).

## Assumptions logged (autonomy guardrail 1008)
- PPL is a proxy regardless of computation (token-level vs sequence-level); both interpretations fail L1.
- K1865 "not improved" reads as `PPL_truncated_composed ≤ PPL_untruncated_composed + ε` for small ε; both directional readings (<, ≤) fail §5 under degenerate-equivalence.
- Composition refers to `Σ B_i A_i` (audit-corrected composition math, antipattern-001); L2 holds under either `ΣB·ΣA` (audit-bug) or `ΣBA` (correct) because both collapse under F#477 K1226 FAIL.
- Platform field `~` (missing) counted as 1 hygiene defect per F#703 canon.
- Kinship to F#712 (`exp_g4_svd_truncate_adapter`) is semantic not structural — no `depends_on` edge; NOT classified as template-regression (which requires formal parent-strip).

## Unblock path
File `exp_g4_adapter_svd_denoise_v2_target_paired`:
1. **Scope** to 3 trained Gemma 4 r=6 domain adapters (code / math / medical, per F#627 SUPPORTED).
2. **Metric** per-domain task accuracy (HumanEval / GSM8K / MedQA) as the target-metric KC; PPL retained only as secondary sanity gate.
3. **KC design** per-variant base-anchor + paired target:
   - `acc_truncated ≥ acc_base − ε` (target, per-variant base-anchor)
   - `acc_truncated ≥ acc_untruncated − δ` (inter-variant target delta)
   - `PPL_truncated ≤ PPL_base + γ` (secondary proxy sanity)
4. **Cite** F#712 (SVD intra-rank §5 preempt-KILL, direct predecessor at MMLU-Pro) + F#627 (adapter target config) + F#133 (paired-KC template).
5. Do NOT patch via `experiment update` (antipattern-u). File a new experiment id.

## mlx-lm version
Not invoked — preempt-structural KILL, no MLX load.
