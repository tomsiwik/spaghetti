# REVIEW-adversarial — exp_g4_grassmannian_ap_pretrain

**Verdict: KILL** (concur with researcher; DB already `status=killed`).

**One-line reason:** 11th audit-2026-04-17 precondition-probe KILL; same upstream blocker as Findings #605–#617; pre-registered tripwire fired honestly in 1.9 ms.

## Independent verification

- **P1:** `find micro/models/exp_p1_t2_single_domain_training/adapters -name '*.safetensors'` → 0; same for `exp_p1_t3_n25_composition` and `exp_p0_n25_vproj_composition`. Confirmed FAIL.
- **P2:** `find micro/models/exp_p1_t0_grassmannian_gemma4 -name '*.safetensors'` → 0; only `run_experiment.py` + docs. Confirmed FAIL.
- **P3:** Upstream T2.1 `results.json` → `verdict: "KILLED"`, `all_pass: false`, no `lora_scale` field. Confirmed FAIL.

## Adversarial checklist (a)–(s)

| # | Check | Result |
|---|---|---|
| a | results.json verdict (`killed`) ↔ DB status (`killed`) | PASS |
| b | `all_pass=false` consistent with `killed` | PASS |
| c | PAPER.md verdict line = "KILLED (K1589 UNMEASURABLE)" | PASS |
| d | `is_smoke=true` consistent with probe-only run | PASS |
| e | KCs unchanged post-claim (single snapshot, K1589 only) | PASS |
| f | Tautology sniff: K1589 requires AP/random ratio measurement; UNMEASURABLE was registered as `fail`, not finessed into `pass` | PASS |
| g | K1589 in code measures `interference ratio <= 0.67`, identical to MATH.md and DB | PASS |
| h | No composition math; no `sum(lora_A`, no `add_weighted_adapter`, no LoRA arithmetic | N/A |
| i | No `LORA_SCALE` in code | N/A |
| j | No routing | N/A |
| k | No `shutil.copy` | N/A |
| l | No hardcoded `{"pass": True}` KC dicts | PASS |
| m | No model loaded — pure file probe | N/A |
| m2 | No MLX code → `/mlx-dev` skill not required for a file probe | N/A |
| n–q | No eval | N/A |
| r | Prediction-vs-measurement table present in PAPER.md | PASS |
| s | Math claims (interference ratio operationalization, K1589 threshold) supported by Finding #132 citation; honest UNMEASURABLE call | PASS |

**17/17 PASS or N/A.** No blocking issues.

## Assumptions logged

- Treating "KILLED + already in DB" as "no double-complete needed" — running `experiment complete` again would duplicate evidence; only `finding-add` is required.
- Cohort claim-queue churn is an orchestrator-level issue, not a research-quality issue, and is already escalated by analyst iters 2/3/4. Reviewer does not re-escalate via REVIEW.md.
