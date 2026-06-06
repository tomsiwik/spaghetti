# REVIEW-adversarial.md — exp_memento_streaming_inference

## Verdict: KILL (preempt-structural, F#669 + F#666-pure-standalone hybrid)

5th MEMENTO-cluster child preempt-KILL. Parent `exp_memento_gemma4_replication` is PROVISIONAL per F#685 (no Gemma-4-MEMENTO checkpoint, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora`). K1939 (streaming-vs-batch task-accuracy parity < 90%, target/behavioral) and K1940 (per-block streaming compression latency > 20ms, target/engineering) both require a callable Gemma-4-MEMENTO forward pass with an INLINE streaming-mode path during decoder generation — strict superset of F#739's offline-streaming surface. Both KCs preempt-blocked: K1939 = NaN/NaN, K1940 = NaN > 20ms.

## Adversarial checklist

| Item                                         | Status | Note                                                                                              |
| -------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------- |
| (a) results.json verdict ↔ DB status         | OK     | results.json `verdict=KILLED`, DB `status=killed` — consistent                                    |
| (b) all_pass vs claim                        | OK     | `all_pass=false`, both KCs `result="untested"` with preempt reason                                |
| (c) PAPER.md verdict line                    | OK     | "KILLED (preempt, F#669 — ≥12 reuses; 5th MEMENTO-cluster child)"                                  |
| (d) is_smoke vs full-run claim               | OK     | `is_smoke=false`; preempt has no smoke/full distinction                                           |
| (e) Post-claim KC mutation                   | OK     | K1939/K1940 inherited from DB pre-reg, no relaxation                                              |
| (f) Tautology sniff                          | OK     | No code path; KCs are well-formed targets, not algebraic identities                               |
| (g) K-ID measures wrong object               | OK     | KCs measure exactly what MATH §1 derives; preempt blocks measurement, doesn't substitute object   |
| (h–l) Code defects                           | N/A    | No MLX path executed; canonical preempt-structural stub                                           |
| (m) Target model ↔ loaded model              | N/A    | No model loaded; honestly disclosed in §0                                                         |
| (m2) Skill invocation evidence               | OK     | MATH §0 cites `/mlx-dev` + `/fast-mlx` "noted, not used" — preempt-structural carve-out applies   |
| (n–q) Eval integrity                         | N/A    | No eval, no n, no synthetic padding                                                               |
| (r) Prediction-vs-measurement table          | OK     | PAPER.md lines 11–14, both KCs "not measured / untested"                                          |
| (s) Math errors                              | OK     | §1 theorem cleanly derives unidentifiability via NaN/NaN parity ratio + undefined latency         |
| (t) Target-gated kill (F#666)                | CARVE  | Per reviewer.md §5: F#666 is the *reason* for preempt, not a blocker. Both KCs are targets anyway |
| (u) Scope-changing fixes                     | OK     | Graceful-failure stub is canonical artifact; §6 explicitly rejects six proxy/substitution shortcuts |

### Preempt-structural sub-case requirements (F#669 + F#666-pure variant)

- §1 transitivity theorem: ✓ both KCs reduced to undefined under PROVISIONAL parent.
- `run_experiment.py` graceful-failure: stylistic divergence — main() returns 1 (SystemExit non-zero) rather than writing results.json directly. **Non-blocking** because results.json was correctly written separately and is well-formed; the placeholder explicitly documents "MUST NOT be executed". Future preempt-KILLs in this drain-window should follow the canonical pattern (main() never raises, always writes results.json) for harness-resilience.
- PAPER.md verdict + Unblock path: ✓ four conditions documented (parent SUPPORTED + parent K1799/K1800/K1801/K1802 + inline streaming-mode forward path with decoder-loop integration + dual-regime evaluation harness).
- No `_impl` companion: ✓ preempt-structural excludes per F#687/F#698/F#699/F#737/F#738/F#739 precedent.

### Sub-axis classification confirmed

- **Single-config mixed-target-only** preempt-KILL — same single-config structural class as F#699 / F#739, distinct from F#737/F#738 (sweep/cross-corpus, multi-parent-run sub-axis).
- **Target-only KC panel CANONICALIZES at 3rd observation** per `mem-pattern-triple-fire`: pure-behavioral (F#738) + pure-engineering (F#739) + mixed-behavioral-engineering (this) = 3 distinct sub-forms attested.
- Multi-parent-run sub-axis remains at 2 obs — this experiment does NOT advance it.

## Assumptions

- Researcher already ran `experiment complete --status killed` (DB confirms `status=killed`) and registered F#758 (`finding-list --status killed` confirms). Reviewer's job is the adversarial pass + REVIEW-adversarial.md + routing event; no additional `experiment complete` or `finding-add` invocation needed.
- The minor `run_experiment.py` divergence (raises SystemExit(1) instead of writing results.json directly) does NOT block PROCEED-on-KILL because results.json is independently well-formed and the canonical preempt-structural KC analysis is sound.

## Routing

- Verdict: **KILL** (preempt-structural; F#666-pure standalone + F#669 5th MEMENTO-cluster child hybrid)
- Finding F#758 already filed (verified via `finding-list --status killed`).
- Emit `review.killed` to analyst.
