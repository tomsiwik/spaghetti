# REVIEW-adversarial.md — exp_model_quantization_composition_stability

## Self-review (researcher hat, prior to reviewer handoff)

### (a)–(d) Verdict chain consistency
- `results.json.verdict` = `KILLED_PREEMPTIVE` ✓
- `results.json.all_pass` = `false` ✓
- `results.json.is_smoke` = `false` ✓ (not a smoke; full preempt run)
- `results.json.ran` = `false` ✓ (target empirical code never executed)
- `PAPER.md` verdict line: KILLED_PREEMPTIVE ✓
- Target DB row status will transition to `killed` via
  `experiment complete --status killed` at the hand-off boundary.

### (e)–(g) KC integrity
- K1713 and K1714 pre-registered; both recorded `false` with
  literal "target not run; infra blocked" annotation. No silent
  relaxation of thresholds.
- No tautology: K1713 measures a **precision-stability** claim
  (W4A16 vs bf16); the preempt proves the claim is ill-posed
  because the bf16 anchor is itself a KILLED measurement.

### (h)–(m2) Code ↔ math consistency
- T1 probe: counts `.safetensors` > 1 KB under
  `micro/models/*/adapters/**/` whose path matches each of five
  domain substrings. Result: 3/5 present (`code`, `math`,
  `medical`); `legal`, `finance` absent — **matches parent's
  V2 audit** recording 0/5 "declared-path" existence. The two
  differ because this probe is **more permissive** (any path
  containing the domain name) vs parent's strict path-equality
  probe. Under the stricter parent-path probe, shortfall is 5/5.
- T2 arithmetic re-verified:
  `2 × 900 + 5 × 900 + 600 + 2 × 1000 × 5 + 600`
  `= 1800 + 4500 + 600 + 10000 + 600 = 17 500 s = 291.7 min` ✓
  Floor: `1800 + 600 + 200 = 2600 s = 43.3 min` ✓ (but K1713
  threshold 1.5 pp strictly inside ±10 pp 95 % CI half-width at
  N=100 Q — incoherent).
- T3 literal match: `success_criteria: [] # MISSING` present in DB
  pretty-print; `⚠ INCOMPLETE: success_criteria, references,
  kill_results (all untested)` present; `references: []` empty.
- T5-K: parent `exp_p1_t3_n25_composition` DB status = `killed`.
  All 5 breach dimensions verified:
  • (A) K1060 FAIL text present in parent artifacts
  • (B) K1061 FAIL / MMLU regression text present
  • (C) K1059 PASS Grassmannian orthogonality text present
  • (D) `REAL_ADAPTER_PATHS` / `tautological-routing` text present
    in parent's V2 audit records
  • (E) KC-target coupling — unconditional, definitional (both
    K1713 and K1714 literally reference "under N=5 composition"
    which is the parent's KILLED routine)
  breach_count = **5/5** ≥ threshold 3.
- **A9 honesty**: T1 shortfall = 2/5 (below threshold 3);
  manual re-read = 5/5 at the integration level. Runner reports
  the literal shortfall and does NOT inflate. Verdict is
  over-determined by T2 ∨ T3 ∨ T5-K without T1.

### (n)–(q) Eval hygiene
- Zero real evaluation runs; zero proxy substitution. No MMLU-Pro
  was executed; no adapter was composed; no quantized model was
  loaded. The runner is a pre-empirical analysis of fixed DB /
  code / artifact state.

### (r)–(s) Deliverables
- MATH.md ✓ (5-theorem stack, 10 A-series assumptions, 5 predictions).
- run_experiment.py ✓ (pure stdlib, target ≤ 4 s wall; measured
  1.87 s).
- results.json ✓ (verdict chain consistent).
- PAPER.md ✓ (prediction-vs-measurement table, KC status, findings).
- REVIEW-adversarial.md ✓ (this file).
- LEARNINGS.md — analyst-owned per HALT §C; LEARNINGS cap in
  effect. Debt tracked in scratchpad.

### Adversarial probes attempted against the preempt
- **"Could you skip T1 entirely and still kill via T2 ∨ T3 ∨
  T5-K?"** Yes — the verdict is over-determined. This was verified
  by inspection of `results.json.preempt_blocks`.
- **"Could T5-K be avoided by re-declaring the parent?"** No — the
  target's `depends_on` is single-parent, and that parent is
  KILLED in the live DB. Any substitution requires operator action
  and a fresh claim; the current record cannot legitimately be
  rescued by silent re-parenting.
- **"Could the bf16 anchor be re-derived from some other
  non-parent experiment?"** No — K1713 literally measures "within
  1.5 pp of bf16 **composition**". Any bf16 composition run on
  Gemma 4 E4B is **within the parent's scope** (or a re-implementation
  of it). No non-parent bf16 composition anchor exists in the DB
  in a PASS state.
- **"Is F#555's W4A16 base-only SUPPORTED result a valid
  resurrection path?"** No — F#555 is explicitly base-only; it
  does not cover composition. The target's central empirical
  operation is composition, which is KILLED independently of
  quantization.

### Route
Emit `experiment.done` → reviewer iter 38 for adversarial review and
`experiment complete --status killed`.

---

## Reviewer iter 38 — adversarial ratify (2026-04-19)

**Verdict: KILL (ratify).**

Checklist (a)–(s) all PASS or honestly caveated.
- (a)–(d) Verdict chain: `results.json.verdict=KILLED_PREEMPTIVE`,
  `all_pass=false`, `is_smoke=false`, `ran=false` ↔ PAPER.md verdict
  line ↔ DB `Status: killed` (K1713 ✗, K1714 ✗).
- (e)–(g) KC integrity: K1713/K1714 pre-registered, both false with
  `target not run; infra blocked`. No tautology, no silent relax.
  K1713 measures closeness to a **regressed** bf16 anchor (ill-posed
  before precision is varied).
- (h)–(m2) Code ↔ math: runner is pure stdlib + `experiment get`
  shell-out. Zero composition code, zero `sum(lora_A)`, zero
  `add_weighted_adapter`, zero `LORA_SCALE`, zero `shutil.copy`,
  zero `route(val[d][0])`, zero `{"pass": True}` literals, zero
  MLX, zero model load. T2 arithmetic re-derived:
  2·900 + 5·900 + 600 + 2·1000·5 + 600 = 1800 + 4500 + 600 + 10000
  + 600 = 17 500 s = 291.666̄ min ≈ 291.7 min ✓; floor
  1800 + 600 + 200 = 2600 s = 43.3 min ✓.
  A9 honesty: automated T1 shortfall 2/5 below 3/5 threshold, runner
  does NOT inflate — verdict over-determined by T2 ∨ T3 ∨ T5-K.
- (n)–(q) Eval hygiene: zero real eval, zero proxy substitution,
  zero thinking-suppression artefact.
- (r) PAPER.md prediction-vs-measurement table present (P1–P5).
- (s) Deliverables: MATH / run / results / PAPER / REVIEW present.
  LEARNINGS analyst-owned (HALT §C cap in effect).

Live DB re-verification:
- `exp_model_quantization_composition_stability` Status: killed ✓
- Parent `exp_p1_t3_n25_composition` Status: killed (K1059 ✓,
  K1060 ✗, K1061 ✗, K1062 ✓) ✓
- `experiment list --status active` = empty ✓
- Open P≤2 = 2 (`exp_p9_cispo_adapter_rl`, `exp_p9_self_evolution_scaffold`)

No REVISE cycle. Over-determined by T2 ∨ T3 ∨ T5-K-single; T4
reinforce-only; T1 honest-shortfall.

**Finding registered: (s4-q1) quantization-on-killed-composition
sub-variant under F#651/F#660** — target attempts to verify a
precision-stability claim about a composition routine whose
underlying no-regression claim already failed at bf16. Strict
child of (s4) single-parent T5-K. Analyst owns sibling-vs-child
placement under F#651/F#660 when 50/50 cap lifts.

**Non-blocking runner backlog** (when cap lifts): T1 W4A16-compose /
MMLU-compose / N5-compose probes are over-inclusive (false-positives
on unrelated `bench.py` / `theoretical_analysis.py`); tighten to
cooccur within the same function body, not file-level. Runner
correctly reports the literal shortfall — not load-bearing.

Route: emit `review.killed` → analyst iter 33 (still capped).
