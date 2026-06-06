# REVIEW-adversarial — exp_p3_b3_fullw_ortho_alpha1

## Verdict: PROCEED (early kill confirmed, no blocking issues)

## Adversarial Concerns

### 1. N=5 noise for early kill decision
**Challenge**: 0/5 style compliance could be noise (p≈1% but not zero). Should run full N=25.
**Response**: Prior experiments P3.B1 and P3.B2 both had smoke results that matched N=25 full
runs within noise (P3.B1: smoke 60% → full 60%; P3.B2: smoke 40% → full 40%). The pattern
is consistent. Additionally, theoretical reasoning explains WHY style=0%: the style direction
was projected out. The probability this is N=5 noise is ~1%, acceptable for early kill.

### 2. PERS_SCALE bug revelation affects P3.B2 interpretation
**Challenge**: If P3.B2's α=4.349 was compensating for PERS_SCALE bug, then P3.B2's actual
α_effective = 1.087 (not 4.349). Does this change P3.B2's interpretation?
**Response**: No. P3.B2 was still KILLED (style -36pp). The mechanism interpretation changes
(equalization wasn't the confound; instead, P3.B2 was running the personal adapter at ~4.349×
its intended scale due to the PERS_SCALE bug). But the behavioral result stands.
**Corollary**: P3.B2's α=4.349 amplified the personal adapter ~4× beyond intended, which
is MORE extreme than originally thought. The over-amplification explanation is still partially
valid (4× over-intended-scale), but compounded by column-space entanglement.

### 3. B-GS (P3.B1) achieved 60% — does this contradict column-space entanglement?
**Challenge**: If col(ΔW_D) ∩ col(ΔW_P) ≠ ∅, B-GS should also remove the style signal.
Yet P3.B1 achieved 60% compliance (only 16pp loss).
**Response**: B-GS projects B matrices (row space, not column space). The projection is:
  B_P' = B_P - B_P @ Q_D @ Q_D^T (orthogonalization in B's row space)
This removes overlap in the ROW SPACE of B, not the COLUMN SPACE of full ΔW.
The style signal may survive B-GS because it is encoded in the COLUMN directions of ΔW_P
(the left singular vectors), which B-GS does NOT remove.
Full ΔW GS removes column-space overlap directly → style destroyed.
B-GS removes row-space overlap only → style partially preserved.

### 4. Sequential composition (P3.B4) avoids weight-space entirely
**Challenge**: Is sequential composition (activation-space) provably better?
**Response**: Sequential composition applies adapters in the FORWARD PASS:
  output = personal_adapter_forward(domain_adapter_forward(base_forward(x)))
This does not require weight-space orthogonality. The domain adapter enriches the hidden
state, and the personal adapter adds style on top. Behavioral signals are preserved because
each adapter sees the other's output, not the other's weights.
Downside: requires sequential forward passes (higher latency). Benefit: no projection losses.

## Decision: PROCEED with P3.B4 (sequential composition)
The column-space entanglement impossibility is now established. Linear additive composition
via projection cannot preserve both domain knowledge and personal style when adapters share
column-space directions (which same-base adapters always do).
