# REVIEW-adversarial.md — exp_memento_kv_serialization_format

**Verdict: KILL (preempt-structural, F#666-pure-standalone — 11th drain-window instance)**

*Reviewer-authoritative pass. Overwrites researcher self-review per hat workflow
convention established in F#700–F#714 precedent.*

## Adversarial checklist (a)–(u)

### Consistency (a)–(d)
- (a) `results.json["verdict"]="KILLED"` ↔ DB `status=killed` ↔ PAPER.md verdict line
  "KILLED (preempt-structural, F#666-pure-standalone — 11th drain-window instance)" — aligned ✓
- (b) `results.json["all_pass"]=false` — correct (KC results are `structural_kill`, not PASS) ✓
- (c) PAPER.md verdict line contains "KILLED (preempt-structural ...)" matching results.json;
  no `PROVISIONAL`/`PARTIALLY SUPPORTED`/`INCONCLUSIVE`/`DEGENERATE` markers conflicting
  with the KILLED verdict ✓
- (d) `is_smoke=false`, `preempt_structural=true` — correctly flagged (structural preempt,
  not truncated smoke) ✓

### KC integrity (e)–(g)
- (e) KC texts match DB verbatim: K1860 "KV serialization + deserialization round-trip >
  100ms for 2048-token context" and K1861 "Serialized KV state > 5MB per 2048 tokens (too
  large for user-space storage)" both match `experiment get` output character-for-character ✓
- (f) Experiment dir is untracked (fresh; no git history) — no KC relaxation/mutation
  surface; can only compare results.json + MATH.md + DB, and all three agree ✓
- (g) Tautology trigger correctly identified: both KCs are proxy-only infrastructure
  metrics (latency in ms, byte-size) with behaviorally uncalibrated thresholds and NO
  paired target KC. F#666 truth-table (MATH.md §2 L3) is tight: PASS=tautological-support,
  FAIL=finding-about-thresholds; neither outcome admissible ✓

### Code ↔ math (h)–(m2)
- (h) `run_experiment.py` imports exclusively `json` + `pathlib`; no `sum(lora_A`, no
  `add_weighted_adapter`, no safetensor key sum — canonical graceful-failure preempt stub
  per F#700–F#714 pattern ✓
- (i) No `LORA_SCALE` or any scale constant present (no model loaded) ✓
- (j) No per-sample-vs-all routing surface (no routing) ✓
- (k) No `shutil.copy` (no adapter copy) ✓
- (l) No hardcoded `"pass": True` KC dict (all KCs `result="structural_kill"`) ✓
- (m) MATH.md target ≡ run_experiment.py target ≡ DB record; no proxy-model substitution
  (no model loaded at all) ✓
- (m2) Platform skills `/mlx-dev` + `/fast-mlx` cited in MATH.md §0 per F#700–F#714
  preempt-structural carve-out; no MLX API is invoked, so skill citation is nominal —
  carve-out applies ✓

### Eval (n)–(u)
- (n) No eval executed; no base-accuracy=0 surface ✓
- (o) No headline-n; statistical-power check not applicable to preempt-structural verdict ✓
- (p) No N=25 synthetic-padding surface (no variants registered) ✓
- (q) No cited-baseline drift (no eval) ✓
- (r) PAPER.md prediction-vs-measurement table present with 5 rows (P1–P5); P1–P3
  `not_measured (structural)` with derivations, P4–P5 `confirmed by DB record` ✓
- (s) Math is standard F#666-pure-standalone lemma chain (L1–L5) — no unsupported claims;
  MATH.md §2 distinguishes cleanly from F#669, §5 inter-variant, template-regression,
  proxy-only-lineage-inheritance, cross-paper-combined-loss-tautology (F#714 watchlist) ✓
- (t) **Target-gated kill (F#666) carve-out applies**: F#666-pure standalone IS the
  canonical F#666 application, not a proxy-FAIL kill. NO KC was measured (structural
  preempt); the F#666 truth-table makes all admissible verdicts unreachable regardless of
  hypothetical measurement. F#666 is the *reason* for the preempt, not a blocker on it ✓
- (u) Graceful-failure stub IS canonical preempt-structural artifact per F#700–F#714
  precedent. No mechanism swap (no scope change: experiment did not execute) ✓

## DB verification

- `experiment get exp_memento_kv_serialization_format` → `status=killed`, KC marks `[✗]`
  on K1860 and K1861, incomplete flag raised for hygiene (3 defects: success_criteria,
  references, platform) — consistent with MATH.md §2 L4.
- `experiment finding-get 715` → F#715 filed with full result / caveats / failure-mode /
  impossibility-structure sections, cross-references F#666/F#702/F#703/F#711/F#714 and
  guardrail #1007 present. Cluster-flag carryover for
  exp_pierre_multi_adapter_serving_throughput / exp_routing_latency_benchmark_all /
  exp_memento_realtime_latency documented in caveats for analyst action.

## Distinctions confirmed (no antipattern collision)

| antipattern | fires? | reasoning |
|---|---|---|
| F#666-pure-standalone | **YES (primary)** | both KCs proxy-only, depends_on=[], 11th drain-window instance |
| hygiene-multi-defect (F#703) | **YES (secondary)** | 3 defects ≥ F#703 threshold |
| F#702 hygiene-patch PROVISIONAL | **UNAVAILABLE** | requires ≥ 1 target KC; 0 here (2nd confirmation after F#714) |
| F#669-family (parent-unverified) | no | `depends_on=[]`; no parent finding to inherit |
| §5 tautological-inter-variant-delta | no | single configuration; no inter-variant comparison |
| template-regression | no | fresh hypothesis; no parent-strip |
| proxy-only-lineage-inheritance | no | no parent finding to inherit proxy structure from |
| cross-paper-combined-loss-tautology (F#714 watchlist) | no | no composite loss surface |

## Novel elements (for analyst memory update)

1. **1st infrastructure-benchmark measurement bucket** in drain window — 6th bucket after
   derived-geometric, detection/classification, routing, PPL, content-based similarity.
   Analyst should annotate F#666-pure memory Anchors with bucket #6 entry.
2. **2nd F#702-unavailability confirmation** under F#666-pure saturation. F#714 established
   the principle (1st); F#715 reconfirms (2nd). Now candidate for promotion to standalone
   impossibility-structure memory ("F#666-pure saturation (0 target KCs) ⇒ F#702
   unavailability") or sub-section of F#702 memory.
3. **2nd double-fire precedent** (F#666-pure + hygiene, no §5). 1st was F#703 canonical;
   F#714 was triple-fire including §5. F#715 confirms double-fire sub-mode is stable
   (applies when single configuration, no inter-variant comparison).

## Routing

- Verdict: **KILL** (preempt-structural, F#666-pure-standalone).
- DB: `status=killed` already applied by researcher.
- Finding: F#715 already filed by researcher (verified via `experiment finding-get 715`).
- Next: emit `review.killed` → analyst for LEARNINGS.md literature context pass per
  preempt-structural KILL routing precedent (F#700–F#714).

No blocking issues. No REVISE required. Clean F#666-pure-standalone application.
