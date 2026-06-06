# Peer Review: notebooklm_moe_scaling_laws (Revision Review)

## Previous Review Summary

The first review issued REVISE with 5 required fixes:
1. Fix Welch bound calculation (was wrong, claimed >3M, actual ~1281)
2. Fix gamma interpretation ("1.8% degradation" should be "improvement")
3. Fix FFN expert size (single matrix -> full FFN, ratio 432-648x not 216x)
4. Weaken zero-interference claim (correct within layer, cross-layer needs honesty)
5. Soften "500M" threshold attribution (project interpretation, not Apple finding)

## Fix Verification

### Fix 1: Welch Bound -- FIXED

The incorrect formula and ">3M" claim have been removed entirely. MATH.md
Section 3.2 now presents the correct Welch bound formula:
```
max_{i!=j} |<v_i, v_j>| >= sqrt((N - d) / (d * (N - 1)))
```
and states qualitatively that d=2560 provides ample capacity for N=25-50 experts.
The text correctly identifies the practical limit as data quality, not geometry.

Minor note (not blocking): for N < d (our case: N=25, d=2560), the expression
under the square root is negative, making the bound vacuous. This means N=25
vectors in R^2560 CAN be exactly orthogonal -- the Welch bound places no
constraint. This actually strengthens the argument. The text could note "for
N << d the Welch bound is vacuous (perfect orthogonality is achievable)" for
completeness, but the current qualitative statement is adequate.

### Fix 2: Gamma Interpretation -- FIXED

PAPER.md Finding 4 now correctly states: "gamma = 0.982 at N=25, meaning
composed PPL is 98.2% of base (1.8% improvement), all 25 domains benefit."
The "What Would Kill This" section also reads "1.8% improvement over base."
Consistent with VISION.md.

### Fix 3: FFN Expert Size -- FIXED

MATH.md Section 0 now shows:
- FFN up+down: 2 * 2560 * 6912 = 35,389,440 params/layer
- FFN with SwiGLU: 3 * 2560 * 6912 = 53,084,160 params/layer
- Ratio: 432x to 648x

Arithmetic verified: 35,389,440 / 81,920 = 432.0. 53,084,160 / 81,920 = 648.0.
PAPER.md Finding 1 is consistent ("Ratio: 432:1 to 648:1"). All instances of
the old 216x ratio have been updated.

### Fix 4: Zero-Interference Claim -- FIXED

MATH.md Section 0 now reads: "interference between adapters is zero by
construction within each adapted layer. Cross-layer interference through
nonlinearities (LayerNorm rescaling, attention's quadratic interactions) is
empirically small (gamma=0.982 at N=25, all domains benefit) but not zero
by construction."

PAPER.md Finding 2 is consistent: "No cross-terms within each layer" and
"Cross-layer interference through nonlinearities is empirically small
(gamma=0.982)."

This is the honest treatment requested.

### Fix 5: "500M" Attribution -- FIXED

MATH.md Section 0 now says "Our interpretation of Apple's MoE scaling laws
(arxiv:2501.12370, ICML 2025) suggests MoE becomes suboptimal below ~500M
params." PAPER.md Finding 1 says "Our interpretation of Apple's scaling law
suggests full FFN experts need minimum scale (~500M)." Both correctly frame
this as the project's interpretation, not a direct Apple finding.

## New Issues Introduced by Revision

None found. The revision is clean -- all 5 fixes were applied without
introducing inconsistencies or new errors. The documents are internally
consistent (MATH.md and PAPER.md agree on all numbers and claims).

## Mathematical Soundness (re-verification)

All arithmetic checked:
- LoRA expert size: 2 * 16 * 2560 = 81,920. Correct.
- FFN expert (up+down): 2 * 2560 * 6912 = 35,389,440. Correct.
- FFN expert (SwiGLU): 3 * 2560 * 6912 = 53,084,160. Correct.
- Ratios: 432x and 648x. Correct.
- Expected cosine: sqrt(16/2560) = sqrt(0.00625) = 0.0791. Correct.
- Overhead ratio: 2*2*16/2560 = 0.025 = 2.5%... wait.

Rechecking the overhead calculation in Section 1.2: "Overhead ratio: k*r/d_in
(for k=2, r=16, d=2560: 1.25%)." This computes k*r/d = 2*16/2560 = 0.0125 =
1.25%. But each adapter contributes O(d_in * r + d_out * r) FLOPs. For
square weight matrices (d_in = d_out = d), the overhead per adapter relative
to the base (2*d^2) is 2*d*r / (2*d^2) = r/d = 16/2560 = 0.625% per adapter,
so k=2 gives 1.25%. Checks out.

The FLOPs table (Section 5.1) shows +0.164M for k=2 LoRA runtime. At d=2560,
r=16: per adapter = 2 * 2560 * 16 = 81,920, times k=2 = 163,840 ~ 0.164M.
Against base 2 * 2560 * 2560 = 13.1M. 0.164/13.1 = 1.25%. Consistent.

## Verdict

**PROCEED**

All 5 required fixes from the previous review have been correctly applied.
No new issues were introduced. The arithmetic is verified. The claims are
appropriately scoped: within-layer interference is zero by construction,
cross-layer interference is acknowledged as empirically small but not proven.
The "500M" threshold is correctly attributed as an interpretation. The FFN
expert size ratio is now accurate.

This is a competent literature review and mathematical analysis that correctly
answers its three research questions. It serves as a reliable reference
document for the project's architecture decisions.
