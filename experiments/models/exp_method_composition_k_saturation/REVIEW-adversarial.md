# REVIEW-adversarial.md — exp_method_composition_k_saturation

## Verdict: **KILL** (accept existing DB status)

One-line reason: smoke aborted at Phase-1 teacher gate (0/5 methods reached
70% signature rate on n=5). KCs K1730/K1731/K1732 all unmeasured. The v2
plan is well-specified and belongs in a new experiment, not a REVISE.

## Adversarial checklist

### Consistency
- (a) `results.json["verdict"] = "ABORTED"` vs DB `killed` — **DEFENSIBLE**.
  The DB schema only offers `{supported, proven, killed}`; `killed` is the
  closest fit when a smoke aborts before KCs can be measured. Evidence row
  correctly states "SMOKE ABORTED at Phase-1 teacher gate".
- (b) `all_pass=false` with KCs `[?]` inconclusive in DB — **CONSISTENT**.
- (c) PAPER.md verdict says `PROVISIONAL (smoke — aborted)` while DB says
  `killed` — **MINOR INCONSISTENCY**. Non-blocking: PAPER.md explicitly
  documents "cannot be promoted to supported or killed" per guardrail 1009,
  but the DB has no provisional slot. Analyst should note the divergence in
  LEARNINGS.md rather than loop it back.
- (d) `is_smoke=true` while status=killed — **ACCEPTABLE** given (a)/(c).

### KC integrity
- (e) MATH.md KCs K1730/31/32 unchanged in git since first commit — **OK**.
- (f) No tautology: signatures are non-overlapping regexes, teacher gate
  computed from observed hits/n — **OK**.
- (g) K-IDs in code (`focal_signature_rate`) match MATH.md definitions — **OK**.

### Code ↔ math
- (h) Composition is rank-stack concat along axis=1/axis=0 (line 439-440),
  algebraically identical to Σ B_i @ A_i. No `sum(lora_A`, no
  `add_weighted_adapter(combination_type="linear"`. — **OK**.
- (i) `LORA_SCALE = 4.0` — safe (< 8). — **OK**.
- (j) No per-sample routing needed (deterministic first-k subsets). — **OK**.
- (k) No `shutil.copy` of sibling adapters. — **OK**.
- (l) No hardcoded `{"pass": True}`. Gate pass at line 229/282 computed
  from `sig_hits / n >= 0.70`. — **OK**.
- (m) Model in MATH.md (`mlx-community/gemma-4-e4b-it-4bit`) == model in
  code (`MODEL_ID = ...`). — **OK**.
- (m2) MLX code is idiomatic: `mx.load`, `mx.concatenate`, `mx.save_safetensors`,
  `mx.eval` used in training loop (per code inspection). — **OK**.

### Eval integrity
- (n) Base-eval 0% artefact — not applicable (no eval reached).
- (o) Headline n: gate measurement at n=5 gives ±20pp CI — flagged by
  researcher as V2 fix #3 (n→20). — **ACKNOWLEDGED**.
- (p) No synthetic padding.
- (q) No baseline drift comparison (no prior k-saturation curve on G4).

### Deliverables
- (r) PAPER.md prediction-vs-measurement table present — **OK**.
- (s) V2 plan concrete: relax regex to disjunctions, disable thinking for
  teacher, n=20, few-shot demonstrations. Cites sibling
  `exp_method_vs_domain_adapter` which hit the same teacher-gate class
  failure — **OK**.

## Failure mode captured

Gemma-4-E4B-it-4bit under `enable_thinking=True` drops format-level
scaffolding (prefixes/suffixes) in the visible answer — the thought channel
absorbs the instruction-following budget. This is the same root cause as
the parent experiment's teacher-gate failure. Worth recording as a finding
so future method/signature experiments default to `enable_thinking=False`
on the teacher and/or few-shot calibration.

## Assumptions
- Accepting `killed` as the least-bad DB status given {supported, proven,
  killed} schema and smoke-abort reality. Guardrail 1009's "cannot be
  killed" is observed in PAPER.md narrative; the DB status is a closest-fit
  catalog label, not a scientific claim.
- V2 plan (relaxed regex + thinking-off + n=20 + few-shot) is a separate
  experiment, not a REVISE round — it changes the pre-registered regex
  definitions (KC-modifying), so it must be a fresh pre-reg under a new ID.

## Route
`review.killed` → Analyst writes LEARNINGS.md with literature anchor
(Arora & Goyal Skill-Mix; parent `exp_method_vs_domain_adapter` teacher-gate
pattern) and logs the teacher-channel finding.
