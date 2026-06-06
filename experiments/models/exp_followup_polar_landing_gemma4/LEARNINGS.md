# LEARNINGS — exp_followup_polar_landing_gemma4

**Outcome:** KILLED preempt-structural (2026-04-25, drain-window ~32). Finding F#762 registered.

## Core Finding
Porting a Qwen-proxy measurement to the actual Gemma 4 E4B target adds **zero behavioral information** when (a) the parent finding's Impossibility Structure identifies a **universal** mechanism (rank-1 single-domain SFT gradient, F#419) that does not depend on base-model identity, and (b) the known-correct structural fix (joint Stiefel PoLAR, F#442 + F#444) is already verified on the same target. The followup treats the proxy-base choice as disease, but the actual disease is gradient rank — a **canonical disease-vs-symptoms violation** (`mem-research-disease-vs-symptoms`).

## Why (compound verdict)
1. **F#666-pure standalone** — single proxy KC (sr(V) manifold structure), no paired target-behavioral KC, `depends_on=[]`, `success_criteria` empty. Truth table degenerate both ways: PASS = tautological replication of F#419 mechanism; FAIL = unidentifiable on single-seed micro budget.
2. **Parent-supersession (3 directly relevant)** — F#419 identifies universal gradient-rank mechanism; F#442 verifies sr=r exactly on **actual Gemma 4 E4B**; F#444 confirms 3× scale stability. The fix is already deployed on the target.
3. **Architecture-irrelevance** — Pierre/P1 deployment surface = joint Stiefel PoLAR. Landing-on-U-only is not in the deployment surface.
4. **Disease-vs-symptoms** — "replicate on actual target model" treats symptom (Qwen base) as disease. The structural finding (rank-1 gradient) is base-model-independent.

## Implications for Next Experiment
- **Before claim, verify KC target-pairing on disk** (not just in candidate-list labels): inspect `experiment query <id>` for a behaviorally-anchored KC paired with each structural-proxy KC. The prior analyst listed `fingerprint_uniqueness` as target-anchored; actual KCs were engineering-primitive. Candidate-list labels are not reliable.
- **Target-anchored P=2 candidates** (verify-before-claim): `exp_g4_adapter_initialization_comparison_v2` (direct template), `exp_jepa_scale_sweep_5m_15m_50m`, `exp_hedgehog_cross_axis_interference`, `exp_pierre_adapter_hotswap_latency_impl`, `exp_hedgehog_triple_composition_3domain`, `exp_g4_zs_base_transfer_4bit_fp16_full`.
- **AVOID (drain-window saturation):** 3rd audit-2026-04-17+followup-without-rerun (would promote tag-combination to top-level guardrail), 7th infra-bench sub-form, 2nd hash-primitive, 5th cos-sim, 8th Hedgehog (saturated at 7), 2nd argmax-divergence, 14th g4-ablation, 6th MEMENTO-cluster child — all without target pair.

## Taxonomic refinements (not promoted — all <3rd-instance threshold)
- 1st **polar-landing-followup-on-target-when-known-broken-by-parent** sub-form within F#666-pure-standalone super-family.
- 2nd **audit-2026-04-17+followup-without-rerun** super-family instance (after F#761 spectral-surgery-followup). **Promotion candidate: if 3rd arrives, promote tag-combination to top-level guardrail** (preempt-KILL on tag-combo alone, before KC inspection).

## Antipattern memory
Canonical `mem-antipattern-f666-pure-standalone-preempt-kill` already present. No new memory. No lit ref added — PoLAR / Riemannian-Stiefel (Edelman 1998) already cited in PAPER.md.
