# LEARNINGS — exp_wildcat_static_velocity_surrogate (KILLED, K2334 upheld)

**Core finding.** The F#862 velocity core is NOT recoverable from final weights alone: the best
static surrogate (S2 top-SVD, EM 0.68) beat a matched-sparsity random mask (0.66) by only +2.0pp
(inside the pre-registered ≤+2pp kill band), with Jaccard 0.442 < 0.45 and the predicted surrogate
ordering failing badly (S1 magnitude 0.50 < random 0.66).

**Why.** The velocity criterion is a trajectory ratio test and the core is magnitude-balanced —
final-weight geometry (magnitude, top-SVD, factor energy) carries almost none of the "which entries
moved late" signal. Meanwhile a *random* mask at f*=0.4335 recovers +22pp of the +30pp gap
(0.44 → 0.66): most of the F#862 benefit is dilution of destructive interference from sparsifying
the thinking delta at all, not selecting the core entries.

**Implication for the next experiment.** Checkpoint-free deployment of the F#862 fix is dead;
the velocity core requires trajectory logging at training time. The cheap, live follow-up is the
random-mask dilution curve (EM vs sparsity, checkpoint-free, ~73% of GT recovery) — that, not
static core detection, is the surrogate worth specifying. GT core (0.74 vs random 0.66 at equal
sparsity) confirms the trajectory signal itself remains real (F#862 / velocity-mask spark intact).

**PIERRE-IMPACT:** shelved — wildcat (no bet ladder) and verdict killed; no code change to any
bet/<name> branch. Forecloses checkpoint-free velocity-mask extraction; any future use of the
F#862 fix must log trajectories during training or use the (unproven-but-cheap) random-sparsity
dilution route.
