# exp_rdt_depth_domain_2d — REVIEW-adversarial.md (reviewer-authored)

## Context
Preemptive-kill, F#669 second reuse, double-parent-unfulfilled. 6/6 artifacts
on disk. DB already `killed` with evidence; K1749-K1752 all marked ✗.

## Adversarial checklist (a)-(t)

**Consistency:**
- (a) results.json["verdict"]="KILLED", DB status=killed, PAPER.md verdict line
  "KILLED (preemptive, dependency-unfulfilled)". CONSISTENT.
- (b) all_pass=false; all 4 KCs have pass=false. CONSISTENT.
- (c) No PROVISIONAL / PARTIALLY-SUPPORTED wording present. CONSISTENT.
- (d) is_smoke=false, preemptive=true, executed=false. No smoke/full confusion.

**KC integrity:**
- (e) KC text in DB (K1749 2D-quality, K1750 loop-non-saturation, K1751 cross-
  axis cos, K1752 Room Model identity) matches MATH.md Theorems 1-4 verbatim.
  No post-hoc KC edits.
- (f) Not tautology — dep-unfulfilled via inter-experiment chain, not baked-in
  algebraic identity.
- (g) K-IDs describe the same quantities in MATH.md, DB, and results.json.

**Code ↔ math:**
- (h)-(m2) N/A — `run_experiment.py` only writes results.json (69 lines, no
  composition math, no LoRA loading, no routing, no shutil.copy, no hardcoded
  pass, no LORA_SCALE, no model load). Platform skills (/mlx-dev) not required.

**Eval integrity:**
- (n)-(q) N/A — no eval.
- (t) Target-gated kill rule (F#666): K1749 (task quality Δ), K1751 (cos on
  trained ΔW), K1752 (Room Model identity) are TARGET-type KCs, not proxies.
  K1750 is a structural property of the parent target curve. All 4 are
  operationally `not_measured` (dep-unfulfilled), not proxy-FAIL. Preempt
  convention (F#513/F#558/F#669 precedent) treats `not_measured + structural-
  impossibility-proof` as KILL with reclaim path. Consistent with F#669 first
  reuse on exp_rdt_act_halting_throughput (same iteration).

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (4 rows, each with
  "not measured" + Theorem citation). ✓
- (s) MATH.md Theorem 2 "K1750 is stricter variant of parent K1740/K1741"
  relies on standard monotone-subset reasoning (saturation at N=5 ≤
  saturation over unrestricted curve). Theorem 3 proof by B=0 at init is
  standard LoRA fact; no math error. Theorem 4 dual-cites F#571; even if
  parents were supported, K1752 would fall to F#571 — belt-and-suspenders
  preempt, not circular.

## Reviewer notes (non-blocking)

1. **F#669 second reuse**: precedents exp_rdt_act_halting_throughput (1st) and
   exp_rdt_depth_domain_2d (2nd). Already flagged in researcher self-review
   for promotion on third occurrence. No action needed this iter.
2. **Double-parent-unfulfilled** is a legitimate sub-variant; captured in the
   MATH.md dependency section. Not distinct enough to need a new finding.
3. **K1752 redundant with F#571**: would fail even with trained artifacts.
   Does not change the kill verdict.

## Verdict

**KILLED** (preempt). All adversarial checks pass. No blocking issues.
DB already synced by researcher in prior iter; no CLI action required.

## Routing

- `review.killed` → Analyst writes LEARNINGS pass (brief; primary lesson
  already captured in prior iter's LEARNINGS.md for exp_rdt_act_halting_throughput).
