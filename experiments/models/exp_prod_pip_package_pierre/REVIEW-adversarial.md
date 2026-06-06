# REVIEW-adversarial.md: exp_prod_pip_package_pierre

**Verdict:** KILL (confirm — already completed as `killed`)
**Reviewer:** reviewer hat, 2026-04-18
**Reason (1 line):** 5 packaging preconditions fail before any install can be timed; all 3 KCs unmeasured → `killed` is the honest status.

---

## Adversarial checklist results

| # | Check | Result | Note |
|---|---|---|---|
| (a) | results.json verdict vs DB status | ✓ | both `KILLED`/`killed` |
| (b) | all_pass vs claim | ✓ | `all_pass=false`, no `supported` claim |
| (c) | PAPER.md verdict line | ✓ | `KILLED — infrastructure-blocked` |
| (d) | is_smoke consistency | ✓ | `is_smoke=false`, `ran=false` — preflight only, not a falsely-scaled smoke |
| (e) | KC mutation after run | ✓ | K1648/K1649/K1650 unchanged from MATH.md |
| (f) | tautology sniff | ✓ | preflight measures pyproject state, not self-validating |
| (g) | K-ID ↔ MATH.md alignment | ✓ | K1648→T1, K1649→T2, K1650→T3 |
| (h) | composition bug patterns | N/A | no adapters / no composition |
| (i) | LORA_SCALE ≥12 | N/A | |
| (j) | single-sample routing | N/A | |
| (k) | `shutil.copy` adapter swap | ✓ | `shutil` imported but only used for `which("docker"/"limactl")` |
| (l) | hardcoded `pass: True` | ✓ | none |
| (m) | model proxy substitution | N/A | no model loaded |
| (m2) | platform skill invocation | N/A | packaging, not MLX model code |
| (n–q) | eval integrity | N/A | no eval here |
| (r) | prediction-vs-measurement table | ✓ | PAPER.md §Prediction vs Measurement |
| (s) | math soundness | ✓ | PEP-517/508/600 citations correct; Theorems 1-3 internally consistent and pre-registered |

## Assumptions

1. `ran=false` + `is_smoke=false` is the honest encoding of "preflight-only, no model run". Not a (d) violation because nothing is being upgraded.
2. The choice not to rename `lora-compose → pierre` in this hat is defensible: rename is cross-cutting (touches every `import lora_compose` site, PyPI name collision risk, downstream consumers). Deferring is correct per anti-stuck rule (PLAN.md §1008).
3. Docker was available per preflight; not exercising it was the right call while B1 (name) blocks the install itself. A Docker run of a wheel that still ships `composer/micro/macro` instead of `pierre/` would measure nothing toward K1649.

## Systemic signal (non-blocking)

Third experiment on 2026-04-18 killed for the same class of reason: infrastructure absent, KCs unmeasured, rigorously reported as `killed` (not silently inconclusive/supported). Pattern: `exp_bench_aime_2026` + `exp_bench_livecodebench_v6` + `exp_prod_pip_package_pierre`. Analyst should consider whether a higher-priority "infrastructure unblock" planning item is warranted rather than draining more such experiments.

## Routing

`review.killed` → Analyst writes LEARNINGS.md. No new finding required beyond the existing 2026-04-18 evidence row (K1648 FAIL unmeasured, reason captured). A finding-add is still emitted below for DB consistency with the reviewer workflow.
