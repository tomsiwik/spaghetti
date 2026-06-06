# REVIEW-adversarial.md — exp_g4_null_space_weighted

## Verdict: KILL (ratify researcher preemptive-kill)

5-theorem stack `all_block=true`. Defense-in-depth: T1 ∨ T3 ∨ T5 each alone blocks SUPPORTED.
K1623 is verbatim duplicate of K1303 (F#496 SUPPORTED @ 32.7pp on identical Gemma 4
e4b-it-4bit / v_proj / N=5 / 3pp / NTP-loss). DB already `status=killed`,
K1623=fail, dir set. F#643 registered (ap-017 (m) tautological-duplicate preempt).

## Adversarial checklist

| Item | Result |
|---|---|
| (a) verdict parity | PASS — results.json=KILLED_PREEMPTIVE, DB=killed |
| (b) all_pass vs claim | PASS — all_5_theorems_block=true, status=killed |
| (c) PAPER.md verdict line | PASS — "KILLED_PREEMPTIVE (5-theorem, defense-in-depth)" |
| (d) is_smoke | N/A — preemptive kill, no model run |
| (e) KC git-diff | PASS — files untracked fresh (no KC drift) |
| (f) tautology sniff | PASS — IS tautological; that IS the kill axis |
| (g) K-ID code vs math | PASS — K1623 in DB = KC_TEXT in runner |
| (h) composition bug | N/A — no training code |
| (i) LORA_SCALE hardcode | N/A — no training code |
| (j) per-sample routing | N/A — no routing code |
| (k) shutil.copy adapter | N/A — no adapter manipulation |
| (l) hardcoded pass dict | N/A — no KC dict |
| (m) proxy substitution | N/A — no model loaded |
| (m2) MLX skills | N/A — pure stdlib runner |
| (n)-(r) eval integrity | N/A — preemptive kill |
| (s) math errors | PASS — theorems derive blocks from literal DB/finding state |

## Direct verification

- **T1 infrastructure**: `adapters/` absent at `micro/models/exp_g4_null_space_weighted/adapters`;
  0 `null_space*` dirs under `micro/models/**`. Shortfall = 5. ✓
- **T3 framework**: `experiment get` literal `Success Criteria: NONE` + `⚠ INCOMPLETE`. ✓
- **T4 KC pins 3/5**: KC_TEXT="weighted > exclusive by 3pp mixed-domain"; ε regex
  `(?:epsilon|ε|±|\+/-|significance|p\s*<)` correctly ABSENT (no numeric over-count
  this iter — regex rejects "3pp"). Enum ABSENT. ✓
- **T5 scope caveat**: F#496 caveats literal via `finding-get 496`:
  "May be generic ensembling, not null-space-specific benefit" + "near-uniform
  TF-IDF weights (entropy 0.996-1.000) mean this tests ensemble averaging, not
  routing" + "Memorization-scale adapters (8 texts, 300 iters)" + "No behavioral
  eval". All 3 breach flags (mechanism-ambiguity, routing-vs-averaging,
  scale-nontransfer) triggered. ✓

## Assumptions

- K1303 verbatim-vs-K1623 is operationalized as shared baseline+delta+metric
  on identical model/projection/N/scale, not character-exact string match. The
  5-dim signature (model, projection, N, metric, baseline, delta) is identical.
- F#643 already registered by researcher iter 29 — no duplicate write.

## Routing

→ `review.killed` (DB already killed; analyst iter 24 owns LEARNINGS.md).

Non-blocking cohort-wide T4 gap: ε regex on "< N" raw threshold still over-counts
on future KCs (not this one). Methodology-epsilon keyword pattern (p<, CI, ±)
should replace numeric threshold matcher cohort-wide.

25th preemptive-kill. Branches under ap-017: composition-bug 20 + scale-safety 2
+ tautological-routing 1 + projection-scope 2 + tautological-duplicate 1.
