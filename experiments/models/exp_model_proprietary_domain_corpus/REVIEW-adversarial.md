# REVIEW-adversarial.md — exp_model_proprietary_domain_corpus

Self-review by researcher. The reviewer hat will independently re-ratify.
All checks below are pre-emit by the researcher.

## (a)-(d) Verdict-chain consistency
- (a) `results.json["verdict"]` = `"KILLED_PREEMPTIVE"`. ✓
- (b) `results.json["all_pass"]` = `false`. ✓
- (c) `PAPER.md` verdict line: "KILLED_PREEMPTIVE — target is unrunnable …". No `PROVISIONAL`, `PARTIALLY SUPPORTED`, `NOT SUPPORTED`, `INCONCLUSIVE`, `DEGENERATE`. ✓
- (d) `is_smoke` = `false`. This is a preempt, not a smoke. ✓

## (e)-(g) KC integrity
- (e) KC pre-registered in DB at claim time (K1704, K1705) before code written. Verified via `experiment get`. ✓
- (f) No KC added, modified, or relaxed. `git diff MATH.md` would be clean (new file, not edited after data). ✓
- (g) KC expressed in `results.json["kill_criteria"]` both `false` with explanation that target was not run. ✓

## (h)-(m2) Code ↔ math alignment
- (h) T1 cooccur-grep scope matches MATH §2 T1 (5 required artifacts). Probe fires for every artifact via false-positive co-occurrence — A9 in MATH.md + PAPER.md documents this transparently. Automated shortfall 0/5 reported; manual re-read 5/5 absent. ✓ (with honest caveat)
- (i) T2 formula in code matches MATH §2 T2 arithmetic: `900 + 3600 + 2400 + 2400 + 300 = 9,600 s = 160.0 min`. ✓
- (j) T3 regex matches MATH §2 T3 (both `Success Criteria: NONE` and `⚠ INCOMPLETE ... success_criteria` patterns, plus `references_empty` check). ✓
- (k) T4 pin_ratio floor 0.20 matches MATH §2 T4. ✓
- (l) T5 declared as N/A in MATH §2 T5 and implemented as such in code (`block=False, reason="no_declared_parent", applicable=False`). Probe checks the DB pretty-print for a declared parent; none present. ✓
- (m) T5 does not participate in verdict — consistent with MATH. ✓
- (m2) A9 self-caveat about T1 cooccur-grep false-positives is **honest**: shortfall reported 0/5 automated, manual re-read 5/5. Runner does NOT silently inflate T1 to make the kill look stronger; the verdict is over-determined by T2 ∧ T3 without T1. ✓

## (n)-(q) Evaluation hygiene
- (n) Zero real eval ran (pure preempt). No benchmark subset bias, no temperature cheating. ✓
- (o) No hardcoded `"pass": True`. KC results both `false`. ✓
- (p) No `shutil.copy` or other fake-adapter artifacts. ✓
- (q) No proxy-for-target substitution. Runner does not claim that any other experiment's domain-corpus data counts for this target. The false-positive T1 hits on `data/distillation/*` are explicitly called out as proxies that **do not satisfy** the KC (public benchmarks cannot stand in for a "non-public specialized corpus" by definition). ✓

## (r)-(s) Deliverables
- (r) Files present: `MATH.md`, `run_experiment.py`, `results.json`, `PAPER.md`, `REVIEW-adversarial.md`. `LEARNINGS.md` is analyst-owned (still capped per HALT §C). ✓
- (s) Runtime wall 1.95 s ≤ 3 s budget. ✓

## Antipattern audit (auto-injected `type: fix` memories)
- `F#502` schema-completeness — covered by T3 (SC NONE + INCOMPLETE + empty references).
- `F#652` software-infrastructure-unbuilt — this is the **first drain
  preempt NOT primarily blocked by F#652**. Code for LoRA SFT on Gemma 4
  E4B exists (matches the T1(4) probe, albeit on closed benchmarks);
  what is missing is the provenance-gated corpus itself. PAPER.md §Novelty
  proposes a **new F-axis** `private-data-unobtainable-by-design`.
  Analyst hat owns final placement (sibling vs child of F#652).
- No new antipattern triggered. No composition math bug (no
  composition run); no unsafe adapter scale (no adapter load);
  no tautological routing (no routing); no `shutil.copy` for
  adapters; no hardcoded `"pass": True`; no eval-template
  truncation; no proxy substitution.

## Risks / known limitations
- **T1 cooccur-grep is structurally weak for this target**: no lexical
  signal distinguishes "non-public proprietary corpus" from any other
  data file. Fix would require a provenance marker (README tag,
  LICENSE-style file) or operator-mounted corpus outside `data/`.
  Non-blocking because T2 ∧ T3 over-determine. Backlog item:
  T1 should require either (a) a `PROVENANCE` sentinel file under
  `corpora/` or (b) a `proprietary: true` YAML/JSON key in a
  per-corpus manifest.
- **T3 treats the `references` field as empty** if no `- ` entry
  appears under the `references:` block of the pretty output. Prose
  mentions of Finding #478 in `notes` do not count. This is the
  pre-registered interpretation (MATH §4 A7) and matches how iter 42
  handled reference-less records.
- **N=5 composition is not required** for this KC (unlike iter 42).
  K1704/K1705 are single-adapter claims. The preempt does not need
  a T1 N=5 probe.
- **T5 N/A is the first in this drain**. Prior 38 preempts all had a
  declared parent. The pattern is documented in MATH §4 A6 and
  §2 T5. Does not weaken verdict — T2 ∧ T3 alone over-determine.

## Route decision
Verdict: KILL (preempt). Over-determined by T2 ∧ T3. No revise cycle.
Downstream: emit `experiment.done` → reviewer iter 35. Analyst iter
33 still capped per HALT §C.
