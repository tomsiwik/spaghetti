# REVIEW-adversarial.md — exp_hedgehog_early_stopping_cos_plateau

## Verdict: KILL (preempt-structural, F#666-pure standalone)

Confirms researcher's preempt-KILL. Already registered in DB as F#756; status=killed.

## Adversarial checklist

**Consistency:**
- (a) results.json `verdict="KILLED"` ↔ DB `status=killed` — match.
- (b) `all_pass=false` ↔ status=killed — consistent.
- (c) PAPER.md verdict line "KILLED (KC-structural preempt, F#666-pure standalone, 7th Hedgehog-ablation super-family sub-type)" — consistent.
- (d) `is_smoke=false` — N/A, no run executed.

**KC integrity:**
- (e) Pre-registered KC text preserved verbatim from `experiment get`; no post-claim mutation (antipattern-u: PASS).
- (f) Tautology sniff: KC structure IS tautological by F#666 — that is the *reason* for preempt-KILL, not a defect of the kill itself.
- (g) K1935 / K1936 descriptions in code/MATH/PAPER match DB.

**Code ↔ math:**
- (h)–(l) No execution path: imports only `json` + `pathlib`. No composition, no LORA_SCALE, no routing, no `shutil.copy`, no hardcoded pass dict.
- (m) No model loaded — preempt-stub never reaches MLX. Skills `/mlx-dev` + `/fast-mlx` cited in MATH.md §0 for (m2) compliance even though not invoked (graceful-failure pattern).

**Eval integrity:**
- (n)–(s) No measurement taken; not applicable.
- (t) Target-gated kill — **does NOT apply** per reviewer.md §5 F#666-pure standalone clause: F#666 is the *reason* for preempt, not a blocker on it. No KC was measured.
- (u) Scope-changing fixes — graceful-failure stub IS the canonical preempt-structural artifact pattern (F#700/F#701/F#703/F#722/F#755 precedent), not a scope reduction.

## Structural shape verified

- `depends_on: []` (parent-orthogonal F#666-pure, NOT F#669-family).
- K = {K1935 cos-sim tightness, K1936 training-time savings} — both proxy per guardrail 1007.
- No paired target-metric KC.
- Hygiene defects (5: missing success_criteria, null platform, empty references, null experiment_dir until this iteration, unanchored 50-step plateau threshold) — non-load-bearing for verdict; F#666-pure structural defect alone is sufficient (precedent: F#700/F#701/F#703/F#722/F#755).
- 4-cell {PASS, FAIL}² verdict truth table in MATH.md §1 confirms no admissible F#666-compliant cell.

## Taxonomic placement (recorded for analyst)

- **7th Hedgehog-ablation super-family sub-type** (training-stopping-criterion / early-stopping-ablation; super-family SATURATES at 7).
- **3rd cos-sim-bucket form** (tightness/distance — after F#720 final-value 1st, F#755 convergence-speed 2nd).
- **1st training-axis efficiency-bucket form** (adjacent to F#753 inference-axis routing-latency).
- **2nd cos-sim-on-cos-sim circularity** (intra-training-trajectory plateau-driving-eval; tighter than F#755 notes-level ordering-driving-eval).
- ~26th F#666-pure-standalone preempt-KILL drain-window instance.

## DB state (already actioned by researcher)

- `experiment complete exp_hedgehog_early_stopping_cos_plateau --status killed` — done.
- `experiment finding-add` → F#756 registered.
- Evidence row added (1 fail).

## Assumptions

- No `_impl` companion required — preempt-structural KILL excludes `_impl` per reviewer.md §5 + F#700/F#701/F#703/F#722/F#755 precedent. Unblock is pre-reg-external (re-register `_behavioral` variant after F#683 → SUPPORTED).
- Skill non-invocation is acceptable for preempt-structural stub since no MLX code paths execute (consistent with F#700/F#701/F#703/F#722/F#755 graceful-failure pattern).

## Verdict routing

KILL → emit `review.killed` → analyst writes LEARNINGS.md.
