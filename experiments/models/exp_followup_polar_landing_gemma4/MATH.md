# MATH: exp_followup_polar_landing_gemma4 — Preempt-Structural KILL

## Verdict (pre-measurement)

**KILLED — preempt-structural** under guardrails 1007 (F#666-pure standalone) + parent-supersession (F#419 / F#442 / F#444) + architecture-irrelevance (Pierre/P1 deployment surface uses joint Stiefel, not landing-field-on-U-only) + disease-vs-symptoms compound.

No measurement performed. Experiment is structurally underspecified, scientifically redundant, and architecturally irrelevant to the deployment surface.

---

## Theorem 1 (F#666-pure: KC structurally insufficient)

Pre-registration of exp_followup_polar_landing_gemma4 holds exactly one KC:

- **K#1561**: "PoLAR V-collapse reproduces on actual Gemma 4 E4B (Qwen-proxy finding survives port)."

K#1561 measures a **structural manifold property** (sr(V) collapse on the V-factor of the PoLAR adapter). Operationally this is sr(V) ≪ r, the same proxy F#419 measured. No paired target KC (downstream task-accuracy gap, behavioral oracle-gap, or any deployment-relevant outcome on Gemma 4 E4B) is present. `success_criteria` is empty. `depends_on=[]`. Therefore (KCs, success_criteria) violates F#666-discipline ∎.

## Theorem 2 (Parent-supersession compound — all 3 prior findings already rule)

Three parent findings jointly preempt the question:

- **F#419** (killed, 2026-04-09): "PoLAR Stiefel on U alone insufficient — V still collapses." Measured on Qwen3-4B-4bit proxy: K1022 FAIL sr(PoLAR)=2.21 < sr(LoRA)=4.45; K1023 FAIL PoLAR=3.3% vs LoRA=13.3% (4× quality gap). **Crucially, F#419's "Impossibility Structure" identifies the failure mode as gradient-structural, not model-specific**: "Task gradient (single-domain SFT) has rank-1 structure. Orthogonal U^T maps gradient isometrically but does not diversify it — all V columns co-adapt to same direction." This mechanism is universal across base models — Gemma 4 E4B's QKV-norm and architectural choices do not modify the gradient rank of single-domain SFT.

- **F#442** (supported, 2026-04-10): "Joint Stiefel PoLAR guarantees sr(ΔW)=r exactly on Gemma 4." sr(PoLAR r=16)=16.0000, sr(PoLAR r=6)=6.0000 to 7dp on **actual Gemma 4 E4B**. Joint Stiefel is the structural fix; it works on Gemma 4. The "Qwen→Gemma 4 port" question is moot — Gemma 4 PoLAR has been measured, and the fix (joint Stiefel) supersedes the broken variant (landing-on-U-only).

- **F#444** (supported, 2026-04-10): "Joint Stiefel PoLAR provides 3× scale stability vs standard LoRA on Gemma 4." Confirms joint Stiefel is the deployment-grade variant.

The followup asks to verify a known-broken variant (landing-on-U-only) on a target where the known-correct variant (joint Stiefel) has already been verified. Replication adds zero behavioral information ∎.

## Theorem 3 (Architecture-irrelevance — landing-on-U-only is not in the deployment surface)

By project-memory `mem-pierre-p1` and `mem-pierre-v5-architecture`, Pierre/P1 deploys **joint Stiefel PoLAR** adapters (per F#442 / F#444), not landing-field-on-U-only PoLAR. The followup's K#1561 measures the V-collapse property of landing-on-U-only on Gemma 4 — i.e., a configuration C' where C' ∩ deployment_surface(Pierre) = ∅. Even a clean PASS or FAIL transfers no actionable signal to Pierre's adapter family ∎.

## Theorem 4 (Disease-vs-symptoms compound — "Qwen proxy" is symptom, "rank-1 gradient" is disease)

F#419's *measurement environment* uses a Qwen3-4B proxy (`Caveats: Qwen3-4B proxy (Gemma4 not loadable)`). F#419's *finding* is gradient-structural ("Task gradient single-domain SFT rank-1 → V co-adapts"). The followup's note (`Fix: replicate on actual target model`) treats the Qwen-proxy choice as the disease — but the parent's `Impossibility Structure` field explicitly identifies the disease as gradient rank, not base model. Repeating the measurement on Gemma 4 cannot falsify a gradient-structural finding because Gemma 4's gradient rank under single-domain SFT is also 1 (Pierre uses Gemma 4 trained on multi-domain mixtures via PLE-M2P; single-domain SFT is the unrealistic configuration).

This is canonical disease-vs-symptoms violation (`mem-research-disease-vs-symptoms`) — addresses parent's measurement-environment symptom, not parent's structural finding ∎.

## Corollary (preempt-KILL is consistent)

Theorems 1–4 jointly imply:
- (T1) Even if the experiment ran and K#1561 PASSED (V collapses on Gemma 4 with landing-on-U-only), the result is a finding about a known-broken proxy variant on a non-deployed configuration.
- (T2) Even if K#1561 FAILED (V does not collapse on Gemma 4 with landing-on-U-only), the result would contradict F#419's gradient-structural mechanism — but absent target-KC pairing the verdict is unidentifiable (could be measurement noise on a single-seed micro budget).
- (T3) Architecture-relevance: nothing transfers to Pierre because Pierre uses joint Stiefel.
- (T4) Disease-vs-symptoms: even a "novel" Gemma-4 result on landing-on-U-only doesn't address the gradient-rank disease.

Measurement adds zero behavioral information; preempt-KILL preserves budget per guardrail 1006 ∎.

---

## Why the failure mode could not be made impossible *in this pre-reg*

To rescue the experiment one would need *all three*:
1. A re-registered behavioral target KC paired with K#1561 (e.g. "Gemma 4 E4B GSM8K accuracy gap PoLAR-landing-on-U vs LoRA ≥4× as in F#419 Qwen result, target ratio preserved within ±50%" — direct port of F#419's behavioral signal).
2. A multi-domain training mixture KC neutralizing single-domain rank-1 gradient (F#419's identified disease), OR explicit acknowledgment that the experiment tests the known-broken-by-F#419 single-domain regime to verify the gradient-structural mechanism.
3. An explicit motivation tying landing-on-U-only V-collapse on Gemma 4 to a deployment-relevant question (e.g., "joint Stiefel infeasible at scale; landing-on-U fallback needed at N>50 adapters") — none present, and Pierre's deployment surface is already joint Stiefel.

None present. The pre-reg notes only "replicate on actual target model" — symptom-level fix to a proxy choice, not a structural rescue.

## Predictions (verdict-table only — no measurement)

| Hypothetical Outcome on Gemma 4 E4B (landing-on-U-only) | F#666 verdict | Pierre-applicability |
|---|---|---|
| K1561 PASS (V collapses, sr(V) ≤ 4 at r=16) | F#666-pure proxy-only PASS — replicates F#419 mechanism on a different base; does not change deployment | NONE — landing-on-U-only ∉ Pierre surface |
| K1561 FAIL (V does not collapse on Gemma 4, sr(V) > 4 at r=16) | F#666-pure proxy-only FAIL — single-seed micro budget cannot identify whether failure is gradient-structural-violation or measurement artifact; absent target KC, unidentifiable | NONE — landing-on-U-only ∉ Pierre surface |

Truth table is degenerate: no cell yields a behavioral conclusion, no cell transfers to Pierre.

## Pre-flight (per researcher.md)

- **Reference**: parent F#419 (killed gradient-structural V-collapse on Qwen), F#442 (supported joint Stiefel PoLAR on Gemma 4 fixes V-collapse to sr=r exactly), F#444 (supported joint Stiefel PoLAR scale stability on Gemma 4); F#666 (target-gated kill); F#761 (immediate sibling: 1st spectral-surgery-followup-on-irrelevant-test-pool sub-form within F#666-pure-standalone super-family).
- **Platform skills invoked**: N/A (no MLX code emitted; preempt-structural).
- **Base model loaded**: N/A (no run).
- **Adapter targets**: N/A.
- **Dataset**: N/A.
- **Runtime budget**: 0 (preempt — no measurement).
- **KC count, target-gated per F#666**: 1 proxy / 0 target → fails F#666-discipline.
- **Antipattern scan**: F#666-pure standalone CANONICAL match (`mem-antipattern-f666-pure-standalone-preempt-kill`); also disease-vs-symptoms (`mem-research-disease-vs-symptoms`) — addresses parent's measurement-environment symptom (Qwen proxy choice), not parent's structural finding (rank-1 gradient → V co-adapts).

---

## Hand-off to PAPER.md

Verdict: **KILLED preempt-structural**. F#762 to be registered: **2nd audit-2026-04-17+followup-without-rerun-tag sub-form within F#666-pure-standalone super-family** (1st was F#761 spectral-surgery-followup; 2nd is polar-landing-followup). Compound with parent-supersession (F#419 + F#442 + F#444 jointly preempt — gradient-structural mechanism is universal, joint Stiefel fix is target-verified) + architecture-irrelevance (Pierre uses joint Stiefel, not landing-on-U) + disease-vs-symptoms (Qwen proxy = symptom, rank-1 gradient = disease).
