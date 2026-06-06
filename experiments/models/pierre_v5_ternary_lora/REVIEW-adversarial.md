# Adversarial Review — exp_pierre_v5_ternary_lora

**Reviewer hat, 2026-04-17.** Inherits the 2026-04-17 researcher-self-review; replaces it as the official adversarial pass. Verdict: **KILL — endorsed.**

## Artifacts inspected

- `MATH.md`, `PAPER.md`, `LEARNINGS.md`, `results.json`, `run_experiment.py` — all current.
- DB record (`experiment get exp_pierre_v5_ternary_lora`) — status already `killed`; K727/K728/K729 all `fail`; evidence row 2026-04-17 registered.
- Filesystem checks for the three missing-dependency claims.
- Git log for MATH.md.

## Adversarial checklist (a)–(s)

| id | check | finding |
|---|---|---|
| (a) | `results.json["verdict"]` vs DB status | no `verdict` key; DB status `killed` ✅ matches KILL verdict. |
| (b) | `all_pass` vs claim | `all_pass:true` in file, but verdict is KILL because the "pass" was constructed tautologically. Correctly downgraded. ✅ |
| (c) | PAPER verdict line | "Verdict: KILLED — dependency-chain + antipattern contamination." ✅ |
| (d) | is_smoke vs claim | n/a; original run was full-scale. |
| (e) | KC git-diff in MATH.md | MATH.md first appears in `f421b73` (this iteration); retroactively authored from claim-time notes. K727/K728/K729 thresholds match DB `kill_criteria` — no post-hoc relaxation. ✅ |
| (f) | Tautology sniff | **TRIGGERS.** `route(val[d][0], ...)` at :159 and `route(test[0], ...)` at :183 fix one adapter per domain then evaluate all samples against it. `results.json.ppl.ternary_pierre` is byte-identical to `ppl.ternary_single` across all 5 domains — tautology confirmed in evidence form. |
| (g) | K-ID measures wrong quantity | K727 claims to measure routed-composition behavioral, but the code measures single-adapter behavioral (same `route(test[0])` fix). Supports KILL. |
| (h) | composition math bug (`ΣA,ΣB`) | n/a; this experiment does per-sample routing to a SINGLE adapter, no cross-adapter sum. |
| (i) | `LORA_SCALE ≥ 12` | **TRIGGERS.** `LORA_SCALE = 20.0` at `run_experiment.py:44`. Safe default is ≤ 8 per Findings #328/#330. |
| (j) | single-sample routing applied to all | **TRIGGERS.** Identical to (f). |
| (k) | `shutil.copy` adapter fraud | n/a — uses `load_adapter(...)`. |
| (l) | hardcoded `"pass": True` | n/a — `pass` values computed from measured values (:263–:265). |
| (m) | proxy model for target | n/a — BitNet-2B is the intended ternary substrate per PLAN.md Part 2; Gemma 4 is the Pierre production target but not required for ternary-composition experiments. |
| (m2) | MLX skill evidence | Code uses `mx.eval`, `mx.clear_cache`, `mx.reset_peak_memory` idiomatically; `mlx_lm.generate` with `make_sampler`. Idiomatic enough that skill invocation is plausible; not a KILL driver. |
| (n)–(q) | eval integrity | n/a — kill is driven by (f)(i)(j), not eval-template issues. |
| (r) | prediction-vs-measurement table | PAPER §"Prediction vs measurement" present with explicit valid/invalid tags. ✅ |
| (s) | math errors | MATH.md §2 theorem statement is informal but consistent with cited priors; no errors affect the KILL verdict. |

## Dependency audit (independent verification)

- `pierre/v5` — `pierre/` contains `pierre.py` (flat file) + `math/`, `archive/` — **no `v5/` submodule**. `from pierre.v5 import ...` at :30 would raise `ModuleNotFoundError`. ❌
- `micro/models/real_data_domain_experts/adapters/` — **directory does not exist** (`ls: No such file or directory`). ❌
- `micro/models/bitnet_sft_generation_v3/` — only `LEARNINGS.md / MATH.md / PAPER.md / results.json / REVIEW-adversarial.md` present; **no `sft_adapters/`**. ❌

Three-for-three on the researcher's non-rerunnable claim.

## Verdict

**KILL — endorsed.** Two antipatterns (f)(i) + single-sample routing (j) + three missing dependencies. Any of (f), (i), (j) alone forces reclassification; the triple hit plus non-rerunnability makes KILL the only honest routing.

DB already reflects this (`status=killed`, K727/728/729 fail, evidence row registered, Finding #553 already captures the antipattern lesson). No further DB writes required. No new finding needed — #553 covers the general case.

## Assumptions

- The pre-audit `results.json` was not silently edited between the original run and this review (no suspicious diff; file shape matches what the code at :267–:284 would emit).
- `pierre/v5` was removed during the post-audit cleanup (commit `de38e37 fix: clean up`), consistent with the researcher's note.
- KC thresholds in MATH.md §3 match the DB `kill_criteria` list; the DB is authoritative if there is ever a mismatch.

## Open thread for successor

Downstream contamination sweep: query DB evidence for any citation of this experiment's composition claim (`ternary_pierre`, "v5 composition", "ternary LoRA composition"). If found, retract. Pierre v8 (per PLAN.md Part 2) is the correct forward path; rebuilding `pierre/v5` module + SFT adapters + Grassmannian skeleton should each be separate upstream experiments, not bundled into a v5.x retry.
