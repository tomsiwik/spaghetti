# REVIEW-adversarial.md — exp_followup_lora_scale_safe_sweep

## Verdict: PROCEED (on KILLED)

17th consecutive audit-2026-04-17 cohort precondition-probe KILL. No
double-complete — DB status already `killed` per `experiment get`. All
17 adversarial checks PASS or N/A.

## Independent verification

| Claim | Verified by | Result |
|---|---|---|
| `.audit/supported_00.md` absent | `ls -la .audit` and `ls -la docs/audit` | both dirs missing — P1 honestly FAIL |
| Upstream T2.1 `_reconstruction_note` names Python 3.14 + datasets + dill | `json.load(.../exp_p1_t2_single_domain_training/results.json)` | confirmed: note contains "Python 3.14", "datasets", "dill" |
| DB status = killed, not active | `experiment get exp_followup_lora_scale_safe_sweep` | `Status: killed` |
| Wall 0.027 s (pure file-probe, no MLX) | `results.json.wall_s == 0.027`, `probe_only: true` | PASS |
| P2 0/120 dirs match | Code inspection of `probe_p2_baseline_adapters` | Loop scans YAMLs under each dir for `LORA_SCALE` + `20` AND `*.safetensors`; 120-dir cap of 656-dir tree honestly flagged in PAPER Assumptions |

## Adversarial checklist (all PASS or N/A)

**Consistency (highest priority):**
- (a) `results.json["verdict"]="KILLED"` matches DB `killed` — **PASS**
- (b) `all_pass=false` consistent with KILLED — **PASS**
- (c) PAPER verdict line: "Verdict: KILLED (K1553 UNMEASURABLE...)" — no PROVISIONAL / PARTIALLY / DEGENERATE — **PASS**
- (d) `is_smoke=false`, `probe_only=true` — **PASS**

**KC integrity:**
- (e) K1553 verbatim from DB pre-registration, no KC diff mid-run — **PASS**
- (f) Tautology sniff: probes honestly return real filesystem results (imports all True, base_cached True, but P1 dirs absent, P2 0 matches, P3 upstream_block present → honest 0/3). Not `x==x` — **PASS**
- (g) K-ID K1553 in code matches MATH.md and DB — **PASS**

**Code ↔ math:**
- (h) No `sum(lora_A` / `add_weighted_adapter(linear)` / summing safetensors — no composition in probe — **N/A**
- (i) No `LORA_SCALE=20` hardcoded as training param — it's the STRING being SEARCHED-FOR to detect flagships, not set — **PASS**
- (j) No routing — **N/A**
- (k) No `shutil.copy` — **N/A**
- (l) No hardcoded `{"pass": True, ...}` — all three probe booleans come from real `Path.exists()` / `importlib.util.find_spec` / substring-in-note — **PASS**
- (m) No model loaded, so no proxy substitution — **N/A**
- (m2) No MLX code invoked — skills not required for a file-probe — **N/A**

**Eval integrity:**
- (n) No inference — **N/A**
- (o) N/A — probe, not headline eval
- (p) No synthetic padding — **N/A**
- (q) No baseline comparison — **N/A**

**Deliverables:**
- (r) PAPER has "Prediction vs Measurement" table with P1/P2/P3 rows — **PASS**
- (s) No math errors; 0 of 3 preconditions pass → K1553 UNMEASURABLE is arithmetically correct — **PASS**

## Escalation (unchanged from iter 7/8/9/10 REVIEWS)

Cohort 17/17 saturated. Ten researcher-hat escalations + nine
analyst-hat escalations for an orchestrator-level claim-queue filter on
`tag=audit-2026-04-17` remain unaddressed. Highest-leverage single
action (rerun upstream T2.1) is mechanically blocked by:

1. `datasets`/`dill` Python 3.14 incompat (documented in T2.1
   `_reconstruction_note`); AND
2. CLI refusal to `experiment claim` a status=killed experiment (analyst
   iter-9 confirmed: "Error: Cannot claim — status is killed, not open").

Reopening upstream via `experiment update --status open` alone does not
unblock because the Python 3.14 toolchain breakage is at
runtime (datasets iter / dill pickling), not at import. Both fixes are
needed.

## Assumptions (autonomy, guardrail 1007)

- 120-dir P2 cap on a 656-dir tree: Researcher honestly flagged. Verdict
  would only flip if ≥3 of the 5 specifically-named-but-unknown flagships
  happened to fall in the un-scanned alphabetical tail AND have
  safetensors. Given the repo has only 23 safetensors files total across
  656 dirs, the conditional probability is low. I accept the 120-dir cap
  as sound.
- DB already `killed` — the researcher did not re-complete; event payload
  said `experiment complete --status killed` was executed, and DB
  reflects it. No double-complete.

## Route

`review.killed` → analyst writes LEARNINGS.md (no antipattern to add;
ap-017 already covers all 17 cohort instances).
