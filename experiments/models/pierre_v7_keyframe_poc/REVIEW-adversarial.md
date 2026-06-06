# REVIEW-adversarial — Pierre v7 Keyframe POC

## Verdict: PROCEED-WITH-KILL

K#745 FAIL is real, reproducible, and explained by the MATH.md theorem. DB
status (killed), results.json (all_pass=false), and PAPER verdict agree. The
reviewer ratifies the kill and flags one antipattern on a non-blocking KC.

## Consistency checks

- results.json verdict=KILLED ↔ DB status=killed ↔ PAPER verdict=KILLED.
- all_pass=false ↔ K#745 pass=false ↔ DB kill_criteria[#745].result=fail.
- is_smoke ∉ results.json; total_time_s = 60.7 and TRAIN_STEPS = 500 (real run).
- actual_steps (500) matches MATH.md pre-registration.

## Adversarial checklist

(a) Consistency — PASS. See above.
(b) KC integrity — PASS. K#745/K#746/K#747 match DB; no post-hoc relaxation.
(c) Tautology — PASS for K#745 (accuracy vs threshold, genuine metric). FAIL-
    ANTIPATTERN for K#746 (see §Antipattern). Flagged but does not affect verdict.
(d) KCs measure claimed phenomenon — PASS for K#745/K#747. K#746 does not.
(e) Code ↔ math — PASS. `ternary_ste` in run_experiment.py:174-177 matches
    MATH.md STE_q definition; rank=16 matches; BCE loss matches; 500 Adam steps
    matches; data generator produces balanced positive/negative labels.
(f) No shutil.copy / no hardcoded pass / no single-sample routing / no
    smoke-scale short-circuit — all clean.
(g) MLX discipline — `mx.eval` after each forward, `mx.clear_cache` in cleanup,
    memory-limit set with 8 GiB headroom, no float64, no Conv2d NHWC concerns.
(h) Eval integrity — N_TEST = 500 held-out balanced split; test seed 43
    differs from train seed 42.
(i) Prediction-vs-measurement — PAPER §"Prediction vs. Measurement" includes all
    six entries and indicates which were pre-registered in MATH.md.
(j) Skills — code dates to 2026-04-05; /mlx-dev discipline on memory and STE
    is observable (mx.eval/mx.clear_cache/stop_gradient all used correctly).

## Antipattern: K#746 is a ghost-composition tautology

Phase 5 composition test constructs `domain_only` and `domain_plus_verifier` PPL
by running the SAME injection code on both branches — `inject_precomputed(m,
skeleton, adapter, di, LORA_SCALE)` only, with no verifier path added. The
reported degradations (0.01%, -0.01%, -0.00%) are model-load noise.

- Severity: **non-blocking** — K#745 delivers the kill regardless.
- Disposition: flag in PAPER, do NOT re-open the experiment. The hypothesis is
  already refuted by K#745; a real composition test would still land in a killed
  state because the classifier itself is collapsed.
- F#157 family (ghost composition / verifier-bypass) — this is a fresh instance
  rather than a new sub-variant; not registering a new finding.

## Observations (non-blocking)

- The base model (Phase 6) is itself 80% correct on this arithmetic set. That
  means the feature "is the supplied answer correct?" is in principle decodable
  given the right signal (compare base prediction to supplied answer). A
  verifier that looks at `(base_prediction, supplied_answer)` rather than a
  single terminal hidden state is the right construction. The ternary-probe
  hypothesis as tested is refuted.
- The collapse-to-majority pattern (pos_acc = 0%, neg_acc = 100%, L ≈ log 2)
  is a strong a-priori tripwire for "balanced BCE over uninformative features".
  Future verifier KCs should include a class-collapse tripwire: `min(pos_acc,
  neg_acc) ≥ 20%`, failing loudly instead of silently balancing.

## Disposition

Mark experiment as KILLED. Do not schedule a follow-up: the immediate
successor "verifier that compares base prediction against supplied answer"
is a distinct design that should live as a new, separately-specified
experiment if prioritised later.
