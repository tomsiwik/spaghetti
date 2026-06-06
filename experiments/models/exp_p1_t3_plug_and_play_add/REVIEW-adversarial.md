# REVIEW-adversarial.md — T3.6: Hot-Add Adapter Without Retraining

**Verdict: KILL (V2 audit 2026-04-18)**

V1 (2026-04-17) review recommended PROCEED on hot-add "supported" verdict.
That V1 review is retracted by this V2 audit pass. V1 text is preserved in
`git log` for provenance; do not refer to it as the current review.

## Why V1 "supported" is retroactively invalid

Two independent failure modes, either sufficient on its own:

1. **Tautological routing antipattern (mem-antipattern-002).**
   V1 `run_experiment.py` declared `REAL_ADAPTER_PATHS = {domain: path, ...}`
   and iterated per-domain, loading one matched adapter per labelled query.
   K1067 ("bit-exact existing outputs after hot-add") is then trivially true
   *regardless of Theorem 1*: the new adapter is never applied to existing-
   domain queries. What V1 measured was the harness's dict-key semantics,
   not the mathematical content of Theorem 1.

2. **Upstream weights absent.**
   Independent re-verify: `find adapters -name "*.safetensors"` returns
   nothing. 0 / 5 expected `.safetensors` on disk (T2.1 math/code/medical +
   T2.6 legal/finance). Upstream `exp_p1_t2_single_domain_training`
   `results.json.verdict = "KILLED"` (metric-swap + format-artefact per
   audit). Local geography + synthetic_adapter_geography stubs hold only
   `adapter_config.json`.

## Adversarial checklist pass

- (a) `results.json["verdict"] = "KILLED"` matches DB `status=killed`. ✓
- (b) `all_pass = false`, no KC passed. ✓
- (c) PAPER.md header line is `Status: KILLED` — no PROVISIONAL /
  SUPPORTED leakage in the header block. ✓
- (d) `is_smoke = false`. Probe is declared non-smoke because it commits
  a real KILL verdict, not a dry-run. ✓
- (e) MATH.md git-diff adds only a V2 Audit Section above the Setting.
  V1 thresholds (K1067 bit-exact, K1068 > base, K1069 < 100ms) are
  **byte-preserved** — no post-hoc KC relaxation. ✓
- (f) No tautology-by-algebraic-identity. K1067/K1068/K1069 each route
  FAIL with explicit "unmeasurable" / "moot under missing preconditions"
  reason strings — distinguishes *cannot measure* from *measured and
  fell short*. ✓
- (g) KC descriptions in results.json match MATH.md and DB rows
  (#1067/#1068/#1069). ✓
- (h) No `sum(lora_A)`, no `add_weighted_adapter("linear", ...)`. Pure
  filesystem probe. ✓
- (i) No `LORA_SCALE` hardcoded — no scaling happens. ✓
- (j) No per-sample routing at all (that's the *defect*, not a new sin).
  V1 hardcoded routing is now flagged in A2; the V2 probe deliberately
  refuses to simulate it. ✓
- (k) No `shutil.copy(...)` of a sibling adapter (the V1 synthetic
  geography = finance copy is flagged but not replayed). ✓
- (l) No hardcoded `{"pass": True}`. All three KC dicts use explicit
  `False` literals with reason strings. ✓
- (m) No model loaded; no proxy substitution possible. ✓
- (m2) Platform-skill evidence not applicable — pure `os.path` probe,
  no MLX arrays touched.
- (n)-(q) Not applicable (no eval run, no n, no baseline).
- (r) PAPER.md V2 Prediction-vs-Measurement table present. ✓
- (s) Math errors: Theorems 1-3 remain correct as *statements*; the
  critique is about **operationalization** (router vs oracle lookup).
  Correctly flagged in MATH.md "Theorem revision note". ✓

## Independent re-verify

- `ls adapters/math/ code/ medical/ legal/ finance/`: each holds only
  `adapter_config.json` (no `.safetensors`). ✓
- `find adapters -name "*.safetensors"`: empty. ✓
- `adapter_geography/` + `synthetic_adapter_geography/`: only
  `adapter_config.json`. ✓
- `exp_p1_t2_single_domain_training/results.json`: `"verdict": "KILLED"`. ✓
- DB `experiment get exp_p1_t3_plug_and_play_add`: `Status: killed`,
  K1067/8/9 all `[✗]`, evidence has 2026-04-10 pass (V1) + 2026-04-18
  fail (V2). ✓

## Standing rules promoted

Rule **#6** (new this iteration): hot-add / hot-remove latency claims
must distinguish the **router update** (O(1) dict semantics) from the
**weight activation** (adapter-load I/O — the actual object of Theorem 3).
V1 K1069 timed only the dict mutation (0.004 ms, 23,000× below
threshold) and falsely reported Theorem 3 verified. This is a
specialization of mem-antipattern-011 ("measuring the wrong quantity
under the right name").

Six precondition-probe kills in 24 h confirm class-level standing:
peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed,
sft_residual_gemma4, n25_composition, plug_and_play_add.

## Assumptions / judgment calls

- Kept V1 "Summary / Prediction vs Measurement / Detailed Results /
  Structural Significance" section below the `Status: KILLED` block in
  PAPER.md because the researcher structured it that way and removing
  it would overwrite the V1 measurements that the `_v1_numbers_for_reference`
  block also preserves. This is explicit provenance, not PROVISIONAL
  leakage — the header is unambiguous.
- Did not require a V3 to be designed here. V3 needs (i) T2.1 rebuild
  (MedQA USMLE, `max_tokens ≥ 512`, persisted `.safetensors`), (ii) T2.6
  rebuild or recovered weights, (iii) T3.1 re-verification, and (iv) a
  genuine router that ingests only query text. Until all four hold, any
  V3 re-lands here.

## Finding recommendation

Status: **killed**. Finding text stored via `experiment finding-add`.
