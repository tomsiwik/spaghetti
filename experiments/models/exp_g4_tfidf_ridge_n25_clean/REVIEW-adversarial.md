# REVIEW-adversarial.md — exp_g4_tfidf_ridge_n25_clean

## Verdict: KILL (ratified)

26th cohort preemptive-kill. 5/5 theorems block SUPPORTED. 2nd
tautological-routing branch under ap-017.

## Adversarial checklist

| Item | Status | Notes |
|---|---|---|
| (a) verdict↔status | ✓ | results.json `KILLED_PREEMPTIVE` → DB `killed`, K1624=fail |
| (b) all_pass↔claim | ✓ | `all_pass: false` + status killed consistent |
| (c) PAPER verdict | ✓ | PAPER header `KILLED_PREEMPTIVE`; no PROVISIONAL/SUPPORTED drift |
| (d) is_smoke | N/A | preemptive (no empirical run) |
| (e) KC-in-git | ✓ | K1624 not modified post-run |
| (f) tautology sniff | ✓ | runner verifies — doesn't commit — the tautology (T5B subcategory label-embedding) |
| (g) K-ID↔measurement | ✓ | runner measures pipeline-absence + pin-count, not 90% accuracy; verdict reports structural block, not a synthetic PASS |
| (h-m) code↔math | ✓ | pure stdlib, no LoRA/LORA_SCALE/routing/shutil/hardcoded-pass |
| (m2) skills | N/A | no MLX / training code |
| (n-q) eval integrity | N/A | no eval was run |
| (r) prediction table | ✓ | PAPER §Prediction-vs-Measurement, all 5 theorems |
| (s) math | ✓ | F#474 caveats quoted literally; pair-count 30× from C(25,2)/C(5,2)=300/10 ✓; MMLU-Pro 14 top-level categories correct; N=25 forces subcategory split |

## Why KILL holds (defense-in-depth)

- T1 alone: 0 pipeline artefacts, shortfall=4 (splits/hard-neg/router/eval)
- T3 alone: `Success Criteria: NONE` + `⚠ INCOMPLETE` DB-literal
- T5 alone: 3/3 F#474 scope breaches (N-scale, subcategory-tautology,
  hard-negative circularity)

Any one of T1, T3, T5 by itself prevents SUPPORTED. Together they make
K1624 a function of the un-pre-registered hard-negative mining choice,
not of TF-IDF ridge generalization.

## ap-017 scope

Register F#474 as ap-017 preempt (n) under tautological-routing branch.
New branch tally: composition-bug 20 + scale-safety 2 + tautological-
routing 2 (this + exp_p2_dummy_routing) + projection-scope 2 +
tautological-duplicate 1 = 26 preempts across 5 axes.

## Non-blocking flags

- T4 ε regex `(?:p\s*<|CI|±|\+/-|significance|epsilon|ε)` excludes raw
  numeric threshold language ("<= 4pp"); cohort-wide patch still owed
  (methodology-ε keyword vs numeric threshold). Not blocking here
  (K1624 has no ε-language at all).

## Assumptions

- Independent-errors assumption for pair-confusion scaling is an
  upper-bound on F#474's single-pair impossibility proof — accepted as
  lower-bound argument, not a claim about true pair correlation.
- MMLU-Pro top-level category count = 14 (HF `TIGER-Lab/MMLU-Pro`)
  at time of review. If schema changes, re-check T5B.

## Route

`review.killed` → analyst. Register F#474→F#644 as ap-017 (n).
Analyst capped (HALT_ESCALATION.md §C) — event may drop silently;
ralph coordinator will drain-forward per prior pattern.
