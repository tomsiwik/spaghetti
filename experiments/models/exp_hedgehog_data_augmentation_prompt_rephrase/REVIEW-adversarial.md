# REVIEW-adversarial.md — exp_hedgehog_data_augmentation_prompt_rephrase

**Verdict: PROVISIONAL (novel-mechanism design-only sub-case)**

Routing: `review.proceed` with `PROVISIONAL:` prefix → analyst. F#723 on tail.

## One-line

Paired-target-anchored KC design (K1877 target behavioral Δ, K1878 proxy variance) on a
custom MLX cos-sim distillation loop not available via `mlx_lm.lora` CLI — canonical
novel-mechanism PROVISIONAL per reviewer.md clause (3+ precedents F#682/683/684/696/
697/713/717/718/719). NEW 5th Hedgehog-ablation sub-type: data-augmentation-ablation.

## Adversarial checklist (a)–(u)

**Consistency (a)–(d) — ALL CLEAN.**
- (a) results.json verdict=PROVISIONAL ≡ DB status=provisional ≡ PAPER.md verdict line.
- (b) all_pass=false + KCs "untested" consistent with design-only PROVISIONAL.
- (c) PAPER.md title line explicitly names PROVISIONAL; no stealth upgrade.
- (d) is_smoke=false correct — design-only is not a smoke-truncated run (it's "nothing
  measured because gracefully stubbed"); matches F#683/684/696/697/717/718/719 precedent.

**KC integrity (e)–(g) — ALL CLEAN.**
- (e) K1877/K1878 pre-reg verbatim in DB; no post-claim relaxation (no measurement yet).
- (f) Tautology sniff: K1877 grounded to external F#683 rubric on held-out prompts
  (inter-variant but anchored to ground truth — NOT §5); K1878 single-variant absolute
  threshold 0.10 (intra-variant, NOT inter-variant delta — NOT §5). Paired per F#666.
- (g) K1877/K1878 labels match across MATH.md §4, run_experiment.py lines 82–83, and
  results.json "kc" dict.

**Code ↔ math (h)–(m2) — ALL CLEAN.**
- (h) No weight-sum composition — no training loop executed (NotImplementedError stubs).
- (i) LORA_SCALE=6.0 ≤ 8 per F#328/F#330.
- (j)(k)(l) — all N/A (no routing, no shutil, no hardcoded pass dicts).
- (m) MATH.md §0 teacher/student/rephraser ids = run_experiment.py lines 54–56 =
  results.json fields — no proxy substitution.
- (m2) MATH.md §0 explicitly cites `/mlx-dev` and `/fast-mlx` invocation requirement
  before the `_impl` training loop lands; header comment in run_experiment.py repeats
  the citation. Satisfies the skill-evidence gate per guardrail 1011/1012.

**Eval integrity (n)–(q) — N/A (no measurement).**

**Target-gated kill (t) — PASSES.**
K1877 IS a target metric (behavioral-quality-judge on held-out F#683 rubric — named
target per rule 1007) paired with K1878 proxy (cos-sim variance). F#666 satisfied.
Researcher's explicit §5 and F#666-pure-standalone rejection sections in PAPER.md
are mathematically correct.

**Scope-preservation (u) — PASSES (canonical novel-mechanism PROVISIONAL).**
NotImplementedError graceful-stub + `_impl` filed inline = canonical pattern, not a
silent scope reduction. main() never raises; writes results.json before exit (verified
by total_time_s=1.7). No silent SFT→LoRA swap, no seqlen truncation, no monitoring
disablement — the _impl remains the single-atomic-fix path.

## PROVISIONAL sub-case criteria (all 4 required artifacts)

1. ✅ MATH.md §0 cites `/mlx-dev` + `/fast-mlx` (satisfies m2 without training loop).
2. ✅ `run_experiment.py` `main()` never raises — all 7 phase helpers wrapped in
   try/except NotImplementedError; results.json written with verdict="PROVISIONAL" and
   KCs "untested"; stub ran cleanly in 1.7 s.
3. ✅ `_impl` follow-up filed inline (`exp_hedgehog_data_augmentation_prompt_rephrase_impl`
   P=3) with K1877/K1878 inherited verbatim; transitive blocker on F#683 `_impl`
   documented.
4. ✅ PAPER.md prediction-vs-measurement table (2 rows, both "not measured") +
   "Scope (this iteration)" + "Measurement blockers (to resolve in _impl)" sections.

## F#702 hygiene-patch audit

Researcher correctly notes F#702 hygiene-patch APPLIES here (K1877 is a target KC →
`mem-impossibility-f666pure-saturation-implies-f702-unavailable` does NOT fire).
Applied: platform=local-apple ✓, success_criteria #95 populated ✓, dir set ✓,
evidence added ✓. References field INCOMPLETE per F#702 global-ref-library CLI
limitation precedent (non-blocking).

## Sub-family ledger accuracy

PAPER.md sibling-position table claims 8 PROVISIONAL + 4 KILLED = 12 instances across
5 sub-types (axis-ext 6, loss-var 2, layer-sel 1, hyperparam 1, data-aug 1). Matches
analyst's running ledger. data-augmentation-ablation is genuinely NEW (no F#469-type
prior-repo instance in the Hedgehog-ablation super-family).

## Non-blocking observations

- A12 transitive blocker on F#683 `_impl` is explicit; if F#683 stalls indefinitely,
  re-scope this ablation's `_impl` to whichever Hedgehog axis `_impl` lands first.
  Not a reviewer-blocker.
- KC-design bifurcation pattern (paired-target → PROVISIONAL; pure-proxy → KILL) now
  confirmed axis-invariant across 5 Hedgehog-ablation sub-types and 12 instances —
  analyst should consider promotion to standalone memory at next Hedgehog-ablation
  instance if the pattern holds.

## Assumptions

- Assumed the F#723 finding row in DB is authoritative over scratchpad narrative for
  sub-family instance counts (verified by `experiment finding-list --status provisional`
  tail).
- Accepted "5× rephrase depth at temp 1.2" as a single experiment (not pre-fractured
  into a rephrase-temperature or rephrase-depth sweep) per A3 Wei-2024-median precedent.
  Depth/temperature sweeps are flagged as legitimate follow-ups, not retro KCs.

Emitting `review.proceed` with PROVISIONAL: prefix → analyst.
