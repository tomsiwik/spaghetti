# PAPER — exp_pgolf_random_frame_compose

**Verdict: KILLED** (both pre-registered kill criteria triggered; `all_pass: false`, `is_smoke: false`)

## Question

Two claims, tested from scratch on a byte-level MLX GPT (d=256, L=4, H=4, ctx=256, mlx 0.31.1,
prose=tinyshakespeare, code=python-stdlib-4MB, seed 1234):

1. **P1 (replication of PGolf #707):** frozen random dense linears + trained rank-32 corrections
   are competitive with a fully-trained dense control at equal budget.
2. **P2 (DFA mechanism):** training two domain adapters over **disjoint** frozen orthonormal
   output frames (B₁ᵀB₂ = 0 at every layer, present at train time) cuts composition interference
   by ≥20% vs sharing one frame.

## Prediction vs measurement

| Criterion | Prediction | Measured | Result |
|---|---|---|---|
| K2320: `bpb_random_frame − bpb_dense` | ≤ 0.08 BPB | **0.1757 BPB** (2.6752 vs 2.4995) | **FAIL** |
| K2321: interference cut `(I_shared − I_disjoint)/I_shared` | ≥ 0.20 | **−0.0235** | **FAIL** |
| Validity gate: `I_shared` | ≥ 0.02 BPB | 0.2528 BPB | passed (K2321 decidable) |

### Phase 1 — random frame vs dense (K2320)

- Dense control: 2.4995 BPB (3,281,408 trainable params)
- Random-frame + rank-32: 2.6752 BPB (725,504 trainable, 77.9% param saving)
- Gap = 0.1757 BPB > 0.08 threshold → replication of #707 **fails** at this substrate/budget.
  The param saving is real but the equal-budget quality claim does not hold.

### Phase 2 — disjoint vs shared frames (K2321)

| Arm | I_prose | I_code | I_mean |
|---|---|---|---|
| shared (B₁ = B₂) | 0.2347 | 0.2709 | **0.2528** |
| disjoint (B₁ᵀB₂ = 0) | 0.2278 | 0.2897 | **0.2587** |

Interference was robustly present (I_shared = 0.2528 ≫ 0.02 gate), so the test had full power.
Disjoint frames were **slightly worse** (cut = −2.4%), nowhere near the +20% required. Exact
output-space orthogonality at every layer, enforced at train time, bought zero composition benefit.

## Interpretation

This closes the one degree of freedom left open by `exp_bet_dfa_r1_n2_composition` (frames imposed
post-hoc recovered only 17.6% of the gap): having the frame present **during training** does not
rescue the DFA mechanism either. The honest-risk note in MATH.md is confirmed — stacked layers mix
the disjoint output coordinates immediately at layer ℓ+1, so parameter/output-space disjointness
does not translate into functional non-interference. Interference here is functional, not a
coordinate-collision artifact; orthogonal frame allocation (shared vs disjoint) is not the lever.

**Verdict line: KILLED — K2320 fail (gap 0.1757 > 0.08 BPB) and K2321 fail (cut −2.4% < 20%).**
