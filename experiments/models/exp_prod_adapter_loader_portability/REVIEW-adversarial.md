# REVIEW-adversarial.md — exp_prod_adapter_loader_portability

Pre-registered adversarial review by researcher hat before reviewer
claim, per PLAN.md §adversarial-first.

## Strongest rebuttal to the kill

**Claim:** the reviewer could try "this experiment can still run
Apple-only as a *partial* measurement and downgrade to PROVISIONAL
rather than KILLED_PREEMPTIVE."

**Rebuttal:**
- The KCs are *identity* claims ("Apple vs CUDA" / "Apple vs CPU").
  A single-backend measurement has zero signal for an across-
  backend identity predicate. cos(A, A) = 1.0 trivially.
- Downgrading to PROVISIONAL would require a well-formed cross-
  backend harness stub (even if small-N). No such harness exists,
  and T5(E) confirms no CUDA-capable loader is on disk.
- T3 independently blocks: DB flags `⚠ INCOMPLETE: missing
  success_criteria`. Per guardrail on schema-completeness
  (F#502/F#646), an INCOMPLETE row cannot be upgraded.

Conclusion: PROVISIONAL would silently upgrade a structurally
unmeasurable claim. Guardrail 1009 forbids this.

## Alternate rebuttal: "run the Apple↔Apple round-trip and declare K1637 transitivity"

- K1637 was already the parent's claim (SUPPORTED on Apple-only
  scope). Re-measuring it here is tautological and doesn't touch
  K1656/K1657.
- T5(A) breach stands: source Assumption 1 explicitly names MLX as
  the primitive, which physically excludes CUDA.

## Weakness in the kill

- T4's pin-ratio metric (2/15 = 0.133) uses a 5-pin template that
  is researcher-defined, not DB-enforced. If reviewer insists on
  counting only "present pin vs required pin", T4 becomes weaker.
  Mitigated: T4 is the weakest-load theorem; defense-in-depth
  holds on T1 ∨ T3 ∨ T5 alone.
- T2's "physical topology ceiling" is a re-labeling of T1 in
  resource terms. Reviewer could collapse T2 into T1. Mitigated:
  explicitly noted as "reinforcing" not "independent" in MATH.md.

## No smoke / no composition / no training

- Runner is pure stdlib, no MLX import, no model load. There is
  no composition math to audit, no LORA scale, no router, no
  `shutil.copy`, no training loop. Antipattern checklist is
  structurally vacuous here (explicitly enumerated in PAPER §6).

## Evidence of diligence

- `nvidia-smi` absence, `uname -m = arm64`, Apple M5 Pro confirmed
  by `system_profiler`.
- Repo grep for CUDA loaders returns zero structural matches.
- Parent PAPER read in full (lines 1–114); Assumptions 1 & 2
  quoted verbatim in MATH.md §Parent/source.
- Runner wrote `results.json` with `verdict=KILLED`,
  `all_pass=false`, `all_block=true`, `is_smoke=false`.

## Reviewer expected verdict

**Ratify KILLED_PREEMPTIVE** with ap-017 (s) registration.
Downgrade or upgrade would violate guardrail 1009.

## Reviewer ratification (2026-04-19, iter 27)

**Verdict: KILL ratified.** Adversarial checklist (a)–(s):

- (a) results.json verdict=KILLED ↔ DB status=killed — consistent ✓
- (b) all_pass=false, all KCs fail — consistent ✓
- (c) PAPER verdict "KILLED_PREEMPTIVE" not PROVISIONAL/PARTIAL ✓
- (d) is_smoke=false; structural preempt, not smoke ✓
- (e) No pre-reg KC modification — 3 KCs match DB verbatim (K1656/57/58)
- (f) No tautology — KCs structurally unmeasurable, not algebraic identity
- (g) K-IDs in runner (K1656/57/58) match DB quantities (cos>0.999, endianness doc)
- (h) No composition math — pure-stdlib probe, no adapter sum/concat
- (i) No LORA scale — no training
- (j) No routing — no router
- (k) No `shutil.copy` of adapters — only `shutil.which("nvidia-smi")` probe
- (l) No hardcoded `"pass": True` — all KCs `"fail"`
- (m) No proxy model substitution — no model loaded
- (m2) No MLX code — stdlib-only preempt; `/mlx-dev` not required for this runner
- (n) N/A — no eval
- (o) N/A — structural preempt, no n<15 statistic
- (p) N/A — no synthetic padding
- (q) N/A — no baseline comparison
- (r) PAPER has prediction-vs-measurement table (5 rows, all ✅ blocks) ✓
- (s) Math sound: 5-theorem stack; T1∨T3∨T5 each independently blocks;
      defense-in-depth = 3/3 independent blockers

**ap-017 (s) hardware-topology-unavailable registration** is defensible:
T1 evidence (`uname -m=arm64`, `nvidia-smi` absent, 0 CUDA grep hits) is
reproducible from disk. T5(A) is a *physical* scope breach — distinct
from the software-semantic breaches of (a)–(r).

**Blocking fixes: none.** **PROCEED to finding-add and review.killed.**
