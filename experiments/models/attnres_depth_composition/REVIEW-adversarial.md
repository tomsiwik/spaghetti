# Peer Review: AttnRes Depth-Wise Attention for LoRA Composition (RE-REVIEW)

## Re-Review Context

This is a re-review after 5 REVISE fixes from the initial adversarial review. Each fix is verified below with pass/fail status, followed by assessment of any remaining issues.

## Fix Verification

### Fix 1: K2 downgraded from PASS to INCONCLUSIVE
**STATUS: APPLIED**
- PAPER.md K2 section header: "INCONCLUSIVE" -- correct
- PAPER.md kill criteria table (line 119): "INCONCLUSIVE" -- correct
- PAPER.md final verdict (line 192): "K2 INCONCLUSIVE" -- correct
- PAPER.md body text (lines 57-59): explicitly states "No paired test can distinguish this from training noise at this sample size" -- correct
- **Minor residual:** results.json still contains `"k2_pass": true` (line 725). This is a data file inconsistency. The PAPER.md is authoritative, so this is non-blocking, but should be noted for anyone programmatically consuming the JSON.

### Fix 2: MATH.md Section 6 softmax corrected
**STATUS: APPLIED**
- MATH.md line 241: `alpha = [0.248, 0.248, 0.503]` -- correct
- Verification: e^0 = 1.0, e^0.707 = 2.028, sum = 4.028. [1/4.028, 1/4.028, 2.028/4.028] = [0.2483, 0.2483, 0.5034]. Rounds to [0.248, 0.248, 0.503]. Matches.

### Fix 3: Hybrid residual design documented in MATH.md
**STATUS: APPLIED**
- MATH.md lines 17-24: New paragraph explicitly states that AttnRes replaces only inter-block residual, intra-block attention sublayer still uses standard additive residual (`h = x + Attn(RMSNorm(x))`), and that `v_l` is the FFN output not the full block output. Notes the implication: "approximately half the residual connections remain additive" at deeper scales.
- This is consistent with the code (run_experiment.py lines 216-219): `h = x + attn_out` (additive internal), then returns `ffn_out` only.

### Fix 4: Grassmannian gap noted in PAPER.md limitations
**STATUS: APPLIED**
- PAPER.md limitation #5 (lines 168-174): Explicitly states random A initialization (`mx.random.normal`), not Grassmannian AP-packed orthogonal A matrices. Notes that interaction between AttnRes and Grassmannian orthogonality is "untested and would need explicit validation before integrating AttnRes into the SOLE pipeline."
- PAPER.md final verdict (line 195): "Uses random A init, NOT Grassmannian skeleton" -- correct.

### Fix 5: Overall verdict updated
**STATUS: APPLIED**
- PAPER.md line 192: "SUPPORTED (K1 PASS, K3 PASS; K2 INCONCLUSIVE)"
- All three caveats listed: K2 INCONCLUSIVE, S1 FAIL, Grassmannian gap.

## Mathematical Soundness (Re-check)

### Verified correct

1. **AttnRes formulation (Section 1).** Depth-wise softmax attention with pseudo-queries, zero-initialization yielding uniform start, shapes, and parameter overhead all check out against arXiv 2603.15031.

2. **Worked example (Section 6).** Softmax values now correct at [0.248, 0.248, 0.503]. The conclusion that "the adapter-heavy layer gets ~2x more weight" follows from 0.503/0.248 = 2.03x. Correct.

3. **Hidden state norm bound (Section 2).** "||h_L|| <= max_i ||v_i||" follows from convex combination (softmax weights sum to 1). Correct by Jensen's inequality on norms.

4. **Complexity analysis (Section 5).** L*d = 512 parameters, O(L^2*d) forward. Accurate.

5. **Code implementation matches math.** DepthAttention computes `w^T * RMSNorm(v_i)` per source, applies softmax over depth, returns weighted combination. Consistent with MATH.md formulation.

### Minor issues (not blocking)

6. **PreNorm dilution formula.** MATH.md claims "||h_L|| ~ O(sqrt(L)) * ||v_l||" for standard residuals. This assumes layer outputs are approximately independent with similar norms. At L=4 with correlated outputs (same base model, same input), the actual growth may differ. The paper correctly notes this is weak at L=4, so the imprecision does not affect conclusions.

7. **"Each layer contributes ~1/L" claim.** This holds for relative norm contribution under the independence assumption, but the actual attention pattern learned (K3 data) shows contributions are already non-uniform in the standard model (the standard model just cannot control them). This is a conceptual nuance, not an error.

## Novelty Assessment

No change from initial review. The experiment tests a legitimate gap (AttnRes + LoRA composition interaction) not addressed in Kimi or MoDA papers. The contribution is purely empirical at micro scale: mechanism validation (K3 PASS) is the main result.

## Experimental Design (Re-check)

### What works

- Three seeds with consistent direction (3/3 show AttnRes better) is appropriate for micro scale
- K1 (base quality preservation) passes with margin (0.984x, threshold 1.10x)
- K3 (depth specialization) passes clearly (entropy ratio 0.775, threshold 0.95)
- Honest assessment section correctly identifies why K2 is inconclusive

### Remaining concern (non-blocking)

**Composition ratio in results.json.** The JSON summary contains `"k2_pass": true` and `"k2_strong_pass": false`, but PAPER.md correctly states K2 is INCONCLUSIVE. The JSON was presumably generated by the script before the manual downgrade. This is a metadata hygiene issue. If any downstream code reads the JSON to determine experiment status, it will get the wrong answer. Recommend updating the JSON or adding a `"k2_verdict": "inconclusive"` field. Not blocking because PAPER.md is the authoritative document.

### Data cross-check

Spot-checked PAPER.md tables against results.json:
- K1: Mean standard base PPL = (1.7873 + 1.833 + 1.7731)/3 = 1.7978, mean AttnRes = (1.698 + 1.7935 + 1.8152)/3 = 1.7689, ratio = 0.9839. PAPER says 0.984. Matches.
- K2: Mean standard ratio = 0.9940, mean AttnRes ratio = 0.9902, improvement = 0.38%. PAPER says 0.39%. Matches (rounding).
- K3: Depth weights in PAPER table match seed 42 composed values from JSON. Verified.

## Macro-Scale Risks (advisory)

No change from initial review:

1. **L=4 to L=16+ extrapolation is speculative.** The mechanism's benefit may not scale linearly with depth.
2. **Memory:** O(L * B * T * d) for storing all layer outputs. At L=32, d=2560, T=2048, B=8: ~5.2GB additional.
3. **Grassmannian interaction untested.** AttnRes changes gradient flow through softmax; frozen Grassmannian A matrices may not maintain their interference bounds under non-uniform depth gradients.

## Verdict

**PROCEED**

All 5 required fixes from the initial REVISE have been properly applied:
- K2 correctly downgraded to INCONCLUSIVE throughout PAPER.md
- Section 6 softmax values corrected
- Hybrid residual design documented in MATH.md
- Grassmannian gap explicitly noted in limitations and verdict
- Overall verdict appropriately calibrated as SUPPORTED with K2 INCONCLUSIVE

The experiment demonstrates a genuine mechanism validation: depth attention learns non-uniform weights (K3 PASS, entropy ratio 0.775) without degrading base quality (K1 PASS, 1.6% better). The composition improvement is honestly reported as inconclusive (0.39% with 3 seeds). Claims are appropriately scoped. The Grassmannian gap and scale limitations are clearly documented.

**Recommendation for next steps:**
- Update results.json `k2_pass` field to reflect INCONCLUSIVE status (metadata hygiene)
- The real test of this mechanism is at L=16+ where PreNorm dilution is meaningful (~6% per layer vs ~25% at L=4)
- Any macro integration MUST test AttnRes with Grassmannian A matrices before claiming SOLE compatibility
