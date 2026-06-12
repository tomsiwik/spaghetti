# LEARNINGS — exp_pgolf_random_frame_compose

**Core finding.** KILLED on both pre-registered criteria: (K2320) frozen random frames +
rank-32 corrections lag a fully-trained dense control by 0.1757 BPB (2.2x the 0.08 kill
threshold), failing to replicate PGolf #707; (K2321) training adapters over exactly disjoint
orthonormal output frames (B1'B2 = 0 at every layer, present at train time) cut composition
interference by −2.4% vs a shared frame — zero benefit despite large, real interference
(I_shared = 0.2528 BPB, far above the 0.02 validity gate).

**Why.** Stacked layers remix the disjoint output coordinates immediately at layer l+1, so
parameter/output-space orthogonality never becomes functional non-interference. Interference
is functional, not a coordinate-collision artifact. This closes the last degree of freedom
left open by exp_bet_dfa_r1_n2_composition (post-hoc frames recovered only 17.6%): having
the frame present during training does not rescue the mechanism either.

**Implication for the next experiment.** The dfa-init bet's R2 rung (train WITH the frame)
is dead before scaling; frame allocation (shared vs disjoint, post-hoc vs train-time) is not
the lever. Per the bet's pre-registered honest-risk plan, skip straight to R3: a
function-space objective (JEPA shared-frozen-predictor latent alignment) that targets
composition additivity directly, not parameter-space geometry.

**PIERRE-IMPACT:** shelved — killed finding; no code change to bet/dfa-init. Do not
implement DFA B-frame init in pierre's training path; the bet survives only via its R3
function-space objective.
