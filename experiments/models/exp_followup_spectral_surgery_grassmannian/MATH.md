# MATH: exp_followup_spectral_surgery_grassmannian — Preempt-Structural KILL

## Verdict (pre-measurement)

**KILLED — preempt-structural** under guardrails 1007 (F#666-pure standalone) and parent-supersession (F#278 / F#488).

No measurement performed. Experiment is structurally underspecified AND scientifically irrelevant to the deployment architecture.

---

## Theorem 1 (F#666-pure: KC structurally insufficient).

Let the pre-registration of an experiment E be the tuple (KCs, success_criteria, references, metric_definitions). E satisfies F#666-discipline iff there exists at least one pair (k_proxy, k_target) ∈ KCs × KCs with k_target measuring a behavioral target metric (task accuracy, oracle-gap, behavioral-quality delta) and k_proxy measuring a structural/algorithmic proxy (cosine similarity, routing-match rate, energy ratio, interference reduction, etc.). KILL requires (k_proxy FAIL) ∧ (k_target FAIL); SUPPORTED requires (k_proxy PASS) ∧ (k_target PASS); mismatched outcomes yield findings about the proxy or tautology.

**Application.** exp_followup_spectral_surgery_grassmannian pre-registers exactly one KC:
- K#1560: "On non-orthogonal adapter pairs, spectral surgery reduces interference by ≥20% vs identity."

The KC measures "interference reduction" — operationally undefined in the pre-reg, but every plausible operationalization (Frobenius cross-term residual, cosine of B-matrix overlap, Gram-error reduction, spectral-deviation-from-union shrinkage) is a structural proxy on the *delta object*, not a behavioral target on downstream task performance. No success_criteria are present. No paired target KC exists. Therefore (KCs, success_criteria) violates F#666-discipline ∎.

## Theorem 2 (Architecture-relevance: test pool is structurally irrelevant to the deployment surface).

Let A be the deployment adapter family (Pierre architecture, P1: Gemma 4 + polar adapters + PLE-M2P + spatial routing). By project-memory mem-pierre-p1 and the mem-pierre-v5-architecture record, A is **Grassmannian-orthogonal by construction** (PoLAR / Stiefel-projected adapters). The followup proposes to test spectral surgery on **non-Grassmannian** adapters — by definition, a different family A' with A' ∩ A = ∅. Any positive measurement on A' reduces interference on A', not on A; spectral surgery's Pierre-relevance is determined by A, not A'.

Parent findings rule on A directly:
- **F#278** (killed): "Spectral surgery is structurally counterproductive for Grassmannian-orthogonal compositions." Established −0.587 correlation between SV magnitude and domain purity — *low SVs are domain-pure, high SVs absorb B-overlap*. Surgery's premise is inverted on A.
- **F#488** (killed): "Spectral surgery structurally incompatible with PoLAR: flat spectrum, basis non-uniqueness, Stiefel violation." Three independent impossibility theorems.
- **F#64** (killed): "Spectral Surgery KILLED on BitNet-2B (no effect on short-trained adapters)." Confirms the kill across non-PoLAR ternary substrate as well.

A positive result on A' cannot resurrect surgery on A because the failure mode on A is structural (Grassmannian orthogonality forces the inversion), not test-pool-conditional. ∎

## Corollary (preempt-KILL is consistent).

Theorems 1 and 2 jointly imply:
- (T1) Even if the experiment ran and K#1560 PASSED on A', the result would be a finding about the proxy on A', not a target signal anywhere;
- (T2) The result would be inapplicable to Pierre's deployment surface A regardless of outcome.

Therefore measurement adds zero behavioral information per guardrail 1006 and budget is preserved by preempt-KILL ∎.

---

## Why the failure mode could not be made impossible *in this pre-reg*

To rescue the experiment one would need *both*:
1. A re-registered behavioral target KC (e.g. "task accuracy on N=2 composed non-Grassmannian adapters preserved within 2pp vs single-adapter baseline" — a target KC), paired with the existing proxy K#1560.
2. A motivation that ties non-Grassmannian-adapter spectral surgery to a deployment-relevant question (e.g. "Pierre will degrade to non-orthogonal A under N>50 rank-stress; surgery as fallback").

Neither is present. The pre-reg notes only "non-orthogonal adapter test pool (not vacuous)" — addressing the parent's vacuity-of-test-pool symptom (F#278's measurement environment), not the parent's structural finding (Grassmannian inverts surgery's premise). Treating the symptom not the disease (mem-research-disease-vs-symptoms violation).

## Predictions (verdict-table only — no measurement)

| Hypothetical Outcome on A' | F#666 verdict | Pierre-applicability |
|---|---|---|
| K1560 PASS (interference reduced ≥20% on non-Grassmannian A') | F#666-pure proxy-only PASS — INCONCLUSIVE; finding about A' proxy | NONE — A' ∉ deployment surface |
| K1560 FAIL (interference reduced <20% on A') | F#666-pure proxy-only FAIL — does not satisfy KILL gate (no target KC) | NONE — A' ∉ deployment surface |

Truth table is degenerate: no cell yields a behavioral conclusion.

## Pre-flight (per researcher.md)

- Reference: parent F#278 (killed), F#488 (killed), F#64 (killed); F#666 (target-gated-kill); guardrail 1007 / mem-antipattern-f666-pure-standalone-preempt-kill (canonical).
- Platform skills invoked: N/A (no MLX code emitted; preempt-structural).
- Base model loaded: N/A (no run).
- Adapter targets: N/A.
- Dataset: N/A.
- Runtime budget: 0 (preempt — no measurement).
- KC count, target-gated per F#666: 1 proxy / 0 target → fails F#666-discipline.
- Antipattern scan: F#666-pure standalone CANONICAL match (mem-antipattern-f666-pure-standalone-preempt-kill); also disease-vs-symptoms (mem-research-disease-vs-symptoms) — addresses parent's measurement-environment symptom not parent's structural finding.

---

## Hand-off to PAPER.md

Verdict: KILLED preempt-structural. F#761 to be registered: 1st spectral-surgery-followup-on-irrelevant-test-pool sub-form within F#666-pure-standalone super-family (~31st drain-window instance). Compound with parent-supersession (F#278 / F#488 directly preempt the structurally-relevant case) and disease-vs-symptoms (test-pool-vacuity is parent's symptom, Grassmannian-inverts-surgery is parent's disease).
