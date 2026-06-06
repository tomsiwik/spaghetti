# exp_followup_polar_landing_gemma4 — Preempt-Structural KILL

**Status:** KILLED (preempt-structural)
**Date:** 2026-04-25
**Verdict basis:** F#666-pure standalone + parent-supersession (F#419 + F#442 + F#444) + architecture-irrelevance + disease-vs-symptoms compound.
**Measurements taken:** 0.

---

## Prediction vs Measurement Table

| Kill Criterion | Pre-registered Prediction | Measurement | PASS/FAIL |
|---|---|---|---|
| K#1561: PoLAR V-collapse reproduces on Gemma 4 E4B (Qwen-proxy finding survives port) | sr(V) ≤ ~4 at r=16 (V collapses with landing-on-U-only) | UNTESTED — preempt-KILL | **untested** |

No measurement performed; preempt-structural per MATH.md Theorems 1–4.

---

## Why this experiment was preempt-killed

### 1. F#666-pure standalone (guardrail 1007)
K#1561 is a structural-manifold proxy (sr(V) is a property of the V-factor, not a downstream behavioral target on Gemma 4 E4B). No paired target KC. `success_criteria` empty. `depends_on=[]`. Fails F#666-discipline pre-measurement — KILL gate cannot be satisfied (proxy-FAIL alone is unidentifiable; proxy-PASS alone is tautological per F#419's prior measurement on Qwen).

### 2. Parent-supersession compound (3 directly relevant prior findings)
- **F#419** (killed 2026-04-09, exp_p1_t1_polar_landing): K1023 PoLAR=3.3% vs LoRA=13.3% on Qwen3-4B; sr(PoLAR)=2.21 < sr(LoRA)=4.45. **Impossibility Structure**: "Task gradient (single-domain SFT) has rank-1 structure. Orthogonal U^T maps gradient isometrically but does not diversify it — all V columns co-adapt to same direction." This identifies the failure as **gradient-structural**, not Qwen-specific. Gemma 4 E4B trained on single-domain SFT also exhibits rank-1 gradient — base-model port does not change gradient rank.
- **F#442** (supported 2026-04-10, exp_p1_c1_polar_gemma4_correct): "Joint Stiefel PoLAR guarantees sr(ΔW)=r exactly on Gemma 4." sr(PoLAR r=16)=16.0000, sr(PoLAR r=6)=6.0000 to 7dp on **actual Gemma 4 E4B**. The correct fix has been verified on the target model.
- **F#444** (supported 2026-04-10, exp_p1_c1_polar_scale_invariance): Joint Stiefel PoLAR provides 3× scale stability vs LoRA on Gemma 4. Joint Stiefel is the deployment-grade variant.

The followup proposes to verify a known-broken variant (landing-on-U-only) on a target where the known-correct variant (joint Stiefel) is already verified. Replication of the broken variant adds zero behavioral information — the gradient-structural mechanism (F#419) explains why landing-on-U-only fails, and joint Stiefel (F#442/F#444) is the deployed fix.

### 3. Architecture-irrelevance
Pierre/P1 deployment surface uses **joint Stiefel PoLAR** (per `mem-pierre-p1`, `mem-pierre-v5-architecture`, F#442, F#444). Landing-field-on-U-only is not in the deployment surface. Even a clean PASS or FAIL of K#1561 transfers no actionable signal to Pierre's adapter family.

### 4. Disease-vs-symptoms violation
The followup's pre-reg note (`Fix: replicate on actual target model`) treats the Qwen-proxy choice as the disease. But F#419's `Impossibility Structure` field explicitly identifies the disease as the rank-1 gradient of single-domain SFT — model-architecture-independent. This is canonical disease-vs-symptoms violation (`mem-research-disease-vs-symptoms`).

---

## Truth Table (degenerate)

| K#1561 outcome on Gemma 4 (landing-on-U-only) | F#666 verdict | Pierre-applicability | Net info |
|---|---|---|---|
| PASS (V collapses, sr(V) ≤ ~4 at r=16) | Proxy-PASS-alone — replicates F#419 mechanism on a different base | None — landing-on-U not deployed | Tautological |
| FAIL (V doesn't collapse, sr(V) > ~4) | Proxy-FAIL-alone — single-seed micro budget cannot identify whether failure is gradient-structural-violation or measurement artifact; absent target KC, unidentifiable | None — landing-on-U not deployed | Unidentifiable |

No cell yields a behavioral conclusion; no cell transfers to Pierre.

---

## Path A — re-register v2 with structural rescue
- Add K#1562 target pair: "Gemma 4 E4B GSM8K-style task accuracy gap PoLAR-landing-on-U vs LoRA preserved within ±50% of F#419 Qwen ratio (3.3%/13.3% = 0.25 ± 0.125)" — direct port of F#419 behavioral signal.
- Add K#1563 multi-domain training KC: "Train on 3-domain mixture; sr(V) at r=16 measured ≥8" — neutralizes single-domain rank-1 disease.
- Add explicit deployment-relevance motivation tying landing-on-U-only to a non-joint-Stiefel use case (none currently exists in Pierre roadmap).

## Path B — subsume into joint-Stiefel scale-up
- F#442 + F#444 together establish joint Stiefel PoLAR on Gemma 4. The natural follow-up is N=2/N=3 joint-Stiefel composition under task-anchored evaluation, not landing-on-U replication.

---

## Related work

- **PoLAR** (Polar-Decomposed Low-Rank Adapter, 2025): polar decomposition for LoRA, Stiefel manifold parameterization. Joint Stiefel constraint U^T U = I_r AND V^T V = I_r is the structural fix; landing-field on U alone is the broken variant per F#419.
- **Riemannian/Stiefel optimization** (Edelman, Arias, Smith 1998 — geometry of algorithms with orthogonality constraints; arxiv:physics/9806030): theoretical foundation for landing-field retraction methods on Stiefel manifolds.
- **HRA, Cayley, Givens** (referenced in F#419 prediction-table): alternative orthogonality-preserving parameterizations for low-rank adapters.

## Assumptions / Notes

- Verdict assigned without measurement per guardrail 1006 (behavioral outcomes over metrics): even a clean K#1561 measurement on Gemma 4 carries no behavioral information for Pierre's joint-Stiefel deployment surface.
- F#419's "Impossibility Structure" is treated as authoritative for the gradient-structural mechanism; the parent finding's body explicitly states the mechanism is gradient-rank-driven, not architecture-driven.
- Pierre architecture inventory drawn from project memories `mem-pierre-p1`, `mem-pierre-v5-architecture`; assumed current as of 2026-04-25.

---

## Hand-off

Verdict: **KILLED preempt-structural**. F#762 to be registered: 1st polar-landing-followup-on-target-when-known-broken-by-parent sub-form within F#666-pure-standalone super-family; 2nd instance of audit-2026-04-17+followup-without-rerun super-family (after F#761 spectral-surgery-followup-on-irrelevant-test-pool, 2026-04-25). Drain-window index ~32.

Recommended next P=2 candidates (verify KC-target-pairing BEFORE claim):
- exp_g4_adapter_initialization_comparison_v2 (direct template, target-anchored)
- exp_jepa_scale_sweep_5m_15m_50m
- exp_hedgehog_cross_axis_interference
- exp_pierre_adapter_hotswap_latency_impl
- exp_hedgehog_triple_composition_3domain
- exp_g4_zs_base_transfer_4bit_fp16_full

AVOID: 3rd audit-2026-04-17+followup-without-rerun (after F#761 + this F#762), 7th infra-bench, 2nd hash-primitive, 5th cos-sim, 8th Hedgehog (saturated), 2nd argmax-divergence, 14th g4-ablation, 6th MEMENTO-cluster.
