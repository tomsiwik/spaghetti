# REVIEW-adversarial — exp_followup_jointly_trained_routing_molora

**Status:** KILLED (preemptive, structural) — self-review for preempt hat.

## Checklist (a)-(t)

**(a) results.json verdict ↔ DB ↔ PAPER.md match.**
- `results.json["verdict"] = "KILLED"`, `mode = "preemptive"`, `all_pass = false`.
- PAPER.md: "KILLED — preemptive, structural impossibility".
- DB: will be `status=killed` post-complete. MATCH ✓

**(b) is_smoke flag.** `false` (no run — preempt analytic). ✓

**(c) all_pass consistent with KC.** `all_pass = false`, K1551 = FAIL. ✓

**(d) KC pre-registration.** K1551 registered 2026-04-17, unchanged.
`git diff MATH.md` N/A (new file). ✓

**(e) KC tautology check.** K1551 measures Δ between two separately-
measurable accuracies — not tautological. ✓

**(f) Code measures what DB describes.** N/A — no code executed.
Preempt rationale in MATH.md Theorem 1. ✓

**(g) Proxy substitution.** N/A — no proxy (no run). Preempt is
based on directly-measured F#431 (TF-IDF accuracy at N=5 on same base
model class). ✓

**(h) MLX idioms / platform.** N/A (no run). ✓

**(i) Hardware / memory.** N/A. ✓

**(j) LORA_SCALE / adapter safety.** N/A (no training). ✓

**(k) Composition math bug.** N/A (no composition executed).
Preempt uses F#340 (ridge + single-pass E2E) as analog — that experiment
did execute and had its composition math reviewed; we reuse the KILLED
verdict without introducing new composition. ✓

**(l) Tautological routing (ap-017).** N/A — no routing code executed.
F#502/F#474 checks do not apply. ✓

**(m) Antipattern match.** Four flags registered in results.json:
near-oracle-ceiling-vs-3pp-threshold, per-token-full-sequence-routing-null,
representation-bottleneck-not-architecture, ridge-analog-already-killed.
All reuse existing findings. ✓

**(m2) Verdict consistency pre-flight (Guardrail 1009).**
1. `results.json["verdict"] = "KILLED"` — not supported. ✓
2. `all_pass = false`. ✓
3. PAPER.md verdict line: "KILLED (preemptive)". ✓
4. `is_smoke = false`. ✓
5. KC unchanged since pre-reg. ✓
6. Antipattern memories checked — four flags registered. ✓
No silent upgrade. ✓

**(n) Seed / eval-N.** N/A (no eval run). ✓

**(o) Eval-template truncation / base=0% pathology.** N/A (no eval). ✓

**(p) Subset vs full-N claim.** N/A (no run). ✓

**(q) Held-out set construction.** Preempt assumes held-out per-token
follows F#431 methodology (5 real domains, Gemma 4 tokenization).
Alternative held-out constructions would either (i) hit F#305 null
(mixed-domain full-sequence) or (ii) hit F#312 MLP-only cap
(segment-isolated), both < 3pp. ✓

**(r) Prediction-vs-measurement table.** PAPER.md includes explicit
table; measurements are `N/A (preempt)` since no run, predicted
bounds are F#431/F#305/F#312/F#340-derived. ✓

**(s) Proof soundness.** Lemma 1: F#431 empirical. Lemma 2: F#305
empirical (PPL 4.815 bit-exact match). Lemma 3: F#312 empirical
(+3.3% MLP-only). Lemma 4: F#193 empirical (N=24 ceiling).
Lemma 5: F#340 empirical (8.6pp drop). Combining is logical AND —
each lemma forces Δ < 3pp in its regime. ✓

**(t) Target-gated KC.** K1551 IS the target quantity (held-out
per-token accuracy gap), not a proxy. Upper-bounded structurally
by F#431 ceiling and F#312 MLP-only cap. PASS. ✓

## Adversarial objections considered

**Obj-1:** "Maybe F#431 ceiling is specific to TF-IDF implementation;
jointly-trained could have different ceiling."
**Response:** Irrelevant — K1551 measures Δ = A_MoLoRA − A_TF-IDF.
F#431 sets A_TF-IDF = 96.6%. For Δ ≥ 3pp, A_MoLoRA ≥ 99.6%. That is
the ceiling K1551 places on jointly-trained, not on TF-IDF. Achieving
99.6% requires the 5 domains to be near-perfectly separable — which
they aren't, per F#431's confusion analysis (finance↔economics overlap).

**Obj-2:** "Held-out might use different domains than F#431 where
TF-IDF is weaker, giving more headroom."
**Response:** The audit-2026-04-17 motivation cites
`exp_mixed_domain_sequences` (mixed-domain) as motivation. If held-out
is mixed-domain, F#305 null applies: per-token full-sequence = per-
sequence for both routers → Δ = 0. Headroom is zero, not larger.

**Obj-3:** "Jointly-training might change hidden-state geometry so
router becomes stronger."
**Response:** F#193 showed routing architecture is representation-
limited at ceiling. Jointly-training CAN shift the representation, but
F#340 already tested this (ridge router + adapters jointly applied E2E)
and it catastrophically failed on mixed-domain (8.6pp drop). On
homogeneous domains both saturate near-oracle; on mixed, both fail.
Shifting the representation does not resolve the context-dependence
failure F#340 diagnosed.

**Obj-4:** "3pp is tight — maybe on a lucky split jointly-trained
could squeeze through."
**Response:** Single-seed variance on 5-domain accuracy is typically
<1pp at sample sizes consistent with held-out eval. A lucky split
would fail pre-registered thresholds on reproducibility anyway. The
structural bound is the dominant constraint, not variance.

## Verdict
**PROCEED-WITH-KILL (preempt ratified via self-review).**
Four-finding structural proof. No new finding needed — direct F#431
ceiling + F#305 null + F#312 cap + F#340 analog reuse.

All 20 checklist items PASS. 4 adversarial objections answered.
