# LEARNINGS — exp_followup_fisher_rao_nre_head_to_head

## Core Finding
Finding #677: Norm-Rescaled Euclidean (NRE) is the composition ceiling at production scale (N=25) on Gemma 4 E4B 4-bit, q_proj, rank-6 scale-6. Fisher-Rao Karcher-mean averaging provides **no measurable benefit** over NRE on either overall PPL (−0.352 favouring NRE) or conditional PPL (−0.031 favouring NRE), while costing **68.19× more wall-clock** to compose. K1 FAIL + K2 FAIL + K3 PASS → KILLED by design. Generalises F#275 (BitNet-2B N≤15) to a different architecture and larger N.

## Why
- **Norm preservation is the entire mechanism.** Euclidean→NRE gap is +2.43 PPL at N=25 (huge); NRE→FR gap is −0.35 PPL *against* FR. Riemannian manifold machinery contributes nothing beyond the scalar rescale.
- **Pennec (2006) small-dispersion bound predicts it.** `‖NRE − FR‖ = O(dispersion²)`; trained LoRA sources from the same task distribution have tiny angular spread, so the bound is below noise floor.
- **Numerical drift hurts FR slightly.** Karcher fixed-point iteration introduces direction perturbations that accumulate across 42 layers; NRE is a one-line rescale, bit-identical to extrinsic mean up to the scalar.
- Review ratified: KCs unchanged since MATH.md pre-reg; target-gated kill (F#666) holds because BOTH proxy and target failed; no antipattern triggered.

## Implications for Next Experiment
1. **Stop reintroducing FR / Karcher-mean composition.** Default composition for all Pierre / adapter-merge code is NRE. FR should not be revisited without an explicit *high-dispersion-source* hypothesis (e.g. cross-domain adapters where sources span a large solid angle on S^{d−1}).
2. **The ceiling question shifts.** NRE vs NRE is no longer where the gains are; the next lever is *what you compose*, not *how you average*. Candidates: independent-A adapters (relaxes shared-A convention), behaviour/procedural adapter pairs (`exp_hedgehog_*`), or memento-style distilled user adapters — not another metric variant.
3. **Don't re-run norm-preservation sanity checks on new bases.** The Euclidean→NRE jump (~2.4 PPL) is now replicated on two architectures; treat it as settled. Burn compute on composition *targets* and *routing*, not on composition arithmetic.
4. **Cost discipline.** A 68× wall-clock premium for a 0-to-negative PPL delta is a strong prior against any geometric-average proposal at production scale. Reject such proposals unless they come with a mechanism that breaks the small-dispersion assumption.

## References
- arXiv:2603.04972 (Wang, Ye, Yin 2025) — Fisher-Rao manifold merging; source of Karcher-mean method.
- Pennec (2006) — Intrinsic Statistics on Riemannian Manifolds; O(dispersion²) bound.
- Finding #274, #275, #666, #677.
