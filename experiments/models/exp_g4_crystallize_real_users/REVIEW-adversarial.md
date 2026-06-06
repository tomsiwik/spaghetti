# REVIEW-adversarial — exp_g4_crystallize_real_users

Self-review pre-handoff. Reviewer hat owns the formal pass.

## Could the verdict be wrong?

1. **Is T1 inventory shortfall the real blocker, or arithmetic?** Both:
   3 base T2.1 adapters {code, math, medical} are *single* canonical
   adapters per domain — not 5 within-domain heterogeneous users. Even
   if the 3 were repurposed as "users" of one domain, the cross-domain
   N=5 cohort is still missing 2 domains and 4 same-domain peers per
   domain. Sibling F#1564 made this concrete with mean_cos=0.9377.

2. **Could runner T1 be a false-negative?** Runner counts directories
   under `exp_p1_t2_single_domain_training/adapters/` as a proxy for
   "real same-domain users." This is the *most generous* read; it still
   counts 3 (one per domain) and the cohort needs 5 *per domain*. False-
   positive risk (over-counting) is bounded by the directory structure;
   false-negative risk (under-counting) would require yet-unknown
   adapters elsewhere on disk — checked via `find` on `*single_domain*`
   and `*g4*`, no additional Gemma 4 user-style adapters surfaced.

3. **Could T3 be reinterpreted?** No — DB row literally tags the
   experiment `⚠ INCOMPLETE: missing success_criteria`. The KC alone
   (cos≥0.95) cannot be promoted to SUCCESS without operator
   `experiment success-add`. Per repo guardrail 1009, no SUPPORTED
   path exists.

4. **Could T5(D) sibling-KILLED be a configuration artifact?** F#1564
   `results.json` shows per-user cos spreads from 0.27 to 0.95 across
   varied LR/steps/seeds — exactly the heterogeneity F#451 deferred.
   Configuration-artifact framing would require *re-arguing* F#451's
   own failure-mode prediction was wrong, which the source itself
   stated as load-bearing.

5. **Could T5(B,C) be downgraded if Gemma 4 hop adds behavioral test?**
   No — proxy-only at source means no SUPPORTED behavioral guarantee
   exists to inherit. Adding a fresh MMLU pass at the hop would be a
   *new* claim, not a transfer of F#451's SUPPORTED status.

## Could a SUPPORTED outcome still emerge?

Not under current conditions:
- (B-unblock) Operator unblocks T2.1 retraining → still requires F#451
  caveats to be revisited; sibling F#1564 measurement stands.
- (C-unblock) Analyst cap raised → only changes documentation flow,
  not the verdict.
- (KC-rewrite) Operator pre-registers behavioral SCs replacing K1630 →
  would create a *different* experiment; not a revival of this one.

The empirical refutation (F#1564) is the one piece of evidence that
no operator unblock can erase.

## Cosmetic / non-blocking items

- T4 ε regex pin coverage is partial (1/5); cohort-wide patch still
  owed (noted in scratchpad iter 30 onward).
- Runner T5 sub-breach checks are LITERAL substring matches against
  `experiment finding-get 451` output. If finding-get summaries change
  upstream, breach (A)/(B)/(C)/(D)/(E) anchors should be re-verified
  against the live finding text. (Same caveat noted for prior preempts.)

## Verdict

KILLED_PREEMPTIVE confirmed under self-review. T1 ∨ T3 ∨ T5 each
independently blocks; T5(D) carries unique empirical strength via
sibling F#1564.

Hand off to reviewer for formal ratification.

---

## Formal reviewer pass (iter 26)

**Adversarial checklist (a)–(s):**
- (a) results.json verdict=`KILLED_PREEMPTIVE` ↔ DB status=`killed`: consistent.
- (b) all_pass=false with status=killed: consistent.
- (c) PAPER.md verdict line "KILLED (preemptive, 5-theorem stack)": consistent.
- (d) is_smoke=false; preempt (pure stdlib, no model load): correctly labeled as full-run preempt.
- (e) MATH.md KC text matches DB K1630 ("cos(crystal, B*) >= 0.95"); not relaxed.
- (f) No tautology: T1 reads filesystem, sibling verdict; T3 reads DB marker; T5 reads F#451 finding text. Block flags are structural decisions justified by independent evidence.
- (g) K1630 measurement uses sibling F#1564 (mean_cos=0.9377), cited as prior empirical refutation — appropriate for preemptive kill.
- (h)–(k) No composition code; runner is pure stdlib; no LORA_SCALE; no shutil.copy.
- (l) `block: True` baked per theorem — justified by independent evidence (shortfall arithmetic, DB marker, finding-get text), not a hardcoded success.
- (m) No model loaded — proxy substitution N/A.
- (m2) Skill invocation N/A (no model code).
- (n)–(q) No eval run; headline is a preempt.
- (q) Sibling measurement (F#1564 mean_cos=0.9377) cited; direction correctly strengthens the KILL (not propping SUPPORTED).
- (r) PAPER.md prediction-vs-measurement table present.
- (s) Math coherent: 5-theorem stack well-structured; T1∨T3∨T5 each sufficient.

**Verdict: KILL** (ratifies preemptive kill). Defense-in-depth holds.
Sibling F#1564 mean_cos=0.9377<0.95 is load-bearing: no operator unblock
can erase empirical refutation already observed.
