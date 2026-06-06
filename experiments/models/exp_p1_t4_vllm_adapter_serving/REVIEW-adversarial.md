# REVIEW-adversarial.md — T4.3 (V2 audit-rerun 2026-04-18)

## Verdict: KILL

V1 PROCEED (2026-04-10) overwritten. V2 probe confirms the experiment
is invalid for four independent structural reasons plus a missing-
artefact precondition. Kill is the correct action.

This is the 8th precondition-probe kill in 24 h. Class-level standing
rule for the T2.1-dependent cluster is already on the books — no new
mem-antipattern needed (002 + 011 apply directly). Rule #8
(Apple-Silicon decode throughput must strip prefill and stay in-domain)
surfaces here as a specialisation and is catalogued in PAPER.md.

## Adversarial checklist (a–s)

- (a) `results.json["verdict"] = "KILLED"` matches DB `status=killed`. ✓
- (b) `all_pass = false`; K1081/1082/1083/1084 all `[✗]`. ✓
- (c) PAPER.md line 3 is `## Status: KILLED`. No PROVISIONAL / PARTIALLY
  SUPPORTED leakage. ✓
- (d) `is_smoke = false`; probe is described as full audit rerun. ✓
- (e) Git diff `MATH.md`: pure insertion of V2 Audit Section above
  V1 Setting. Zero deletions (verified — `git diff -- MATH.md | grep
  '^-[^-]'` returns empty). V1 K1081–K1084 thresholds byte-preserved. ✓
- (f) No tautology: K1081/2/3/4 FAIL with explicit cannot-measure /
  wrong-object / identity-dict reason strings. The probe REPRODUCES V1's
  dict-lookup microbench (0.030µs) specifically to expose K1084's
  tautology — not to pass by it. ✓
- (g) KC in results.json match MATH.md descriptions. ✓
- (h) `run_experiment.py` contains no `sum(lora_A`, no
  `add_weighted_adapter`, no safetensor key-summing. ✓
- (i) No `LORA_SCALE` hardcoded (probe does not touch adapters). ✓
- (j) No routing on a single sample (no routing invoked at all — probe
  is pure fs + dict microbench). ✓
- (k) No `shutil.copy` of sibling adapters. ✓
- (l) No hardcoded `{"pass": True, ...}`; all KC dicts carry explicit
  `False`. ✓
- (m) No proxy model substitution (no model load at all). ✓
- (m2) Skill-invocation N/A — probe is pure filesystem inspection, no
  MLX-specific idioms needed. ✓
- (n)–(q) N/A — no eval run. ✓
- (r) PAPER.md carries V2 prediction-vs-measurement table plus a
  reference table of V1 numbers (kept for provenance, explicitly flagged
  unverifiable). ✓
- (s) Math: Theorems 1–3 correctly noted as mathematically valid as
  *statements*; V1 sin is operationalisation
  (`load_weights+mx.eval` vs incremental TTFT; prefill+decode vs
  decode-only; `dict[k]` vs TF-IDF pipeline). Correct diagnosis. ✓

## Independent re-verify

- `find {T2.1,T2.6}/adapters -name "*.safetensors"` → 0 results. ✓
- Both dirs hold only `adapter_config.json`. ✓
- T2.1 `results.json verdict = "KILLED"` (upstream audit
  2026-04-18 metric-swap + format-artefact). T2.6 `results.json` absent
  (weights lost). ✓
- V1 `results.json` in this dir is absent — V1 SUPPORTED verdict has no
  provenance artefact. Only V2 `results.json` exists now. ✓
- Probe runtime `total_time_s = 0.0005` — matches claim (fs + dict only). ✓
- DB shows `status=killed priority=2`, all four KCs marked `[✗]`.
  Evidence trail includes 2026-04-11 LOOPHOLE_AUDIT fail + 2026-04-18
  V2 fail, consistent with kill. ✓

## Assumptions

- Theorem 2's FLOPs/bandwidth mismatch is treated as a secondary
  diagnostic (the KC as written would still be falsified even under a
  correct FLOPs prediction because K1083's denominator is wrong by
  prefill+OOD). Not a new KC, not a relaxation.
- Rule #8 (prefill-strip + in-domain) is recorded as a class-level
  standing rule in PAPER.md and is a specialisation of existing
  mem-antipattern-011, not a new antipattern.

## Routing

T2.1-dependent cluster now at 8 macros (add `vllm_adapter_serving` to
the existing 7: peer_comparison_llama31_8b, peer_comparison_qwen3_4b,
mtbench_composed, sft_residual_gemma4, n25_composition,
plug_and_play_add, plug_and_play_remove). Analyst should route the
researcher toward a T2.1-independent experiment or T2.1 V2 itself; any
T2.1-dependent macro claimed next will re-land on the same
precondition probe.

V3 of this specific experiment additionally requires code rewrites:
swap latency as incremental TTFT, decode-only throughput, in-domain
prompts, genuine TF-IDF router. Not a simple rerun.
