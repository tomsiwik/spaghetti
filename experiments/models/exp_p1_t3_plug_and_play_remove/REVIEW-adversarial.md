# REVIEW-adversarial — exp_p1_t3_plug_and_play_remove (V2 audit-rerun, 2026-04-18)

**Verdict: KILL**

V1 (2026-04-10, "PROCEED / SUPPORTED") overturned. Three independent
structural causes plus a missing-artefact precondition. KCs byte-preserved;
verdict flips because the V1 apparatus could not have tested the stated
theorems. V1 review preserved in git blame.

## Adversarial checklist (a)–(s) — all clean

- (a) `results.json["verdict"] = "KILLED"` matches DB `status=killed`. ✓
- (b) `all_pass = false` consistent with KILLED. ✓
- (c) PAPER.md leads with `Status: KILLED (V2 audit-rerun 2026-04-18)`. No
      PROVISIONAL / PARTIALLY SUPPORTED leakage. ✓
- (d) `is_smoke = false`. ✓
- (e) MATH.md git-diff: only adds V2 Audit Section above the V1 Setting; the
      original Theorems 1–3 and KC table (K1070 = 0/N diffs, K1071 > 4% base,
      K1072 < 10ms) are byte-preserved. No threshold relaxation. ✓
- (f) Tautology sniff: K1070/K1071/K1072 all FAIL with explicit
      "cannot-measure" reasons (structural tautology, adapter copy forgery,
      wrong-object benchmark). The probe does not pass any KC by algebraic
      identity. ✓
- (g) K-IDs in code measure the quantities MATH.md / DB describe (filesystem
      preconditions + dict-mutation micro-bench used only as the V1-bench
      reproduction). KCs are routed FAIL with reason strings, not faked PASS. ✓
- (h) No `sum(lora_A)` / `add_weighted_adapter("linear")` — no model load. ✓
- (i) No `LORA_SCALE` — no training in probe. ✓
- (j) No per-sample-routing-on-one-sample violation — no routing at all. ✓
- (k) No `shutil.copy(...)` of sibling adapter; the probe identifies V1's
      `shutil.copy(finance, history)` as a kill cause (mem-antipattern-009). ✓
- (l) No hardcoded `{"pass": True}`; KCs use explicit `pass: False` literals. ✓
- (m) No proxy model substitution — no model loaded. ✓
- (m2) No platform code requiring `/mlx-dev` / `/fast-mlx`; pure stdlib probe. ✓
- (n)–(q) Eval integrity items not applicable (no eval). ✓
- (r) PAPER.md V2 prediction-vs-measurement table present. ✓
- (s) Math: Theorems 1–3 are correct as statements; V1's sin is
      operationalisation, correctly diagnosed in PAPER.md "Theorem
      Correctness Note". ✓

## Independent re-verify

- `find adapters -name "*.safetensors"` over upstream T2.1 + T2.6 dirs and
  local `adapter_geography/` + `adapter_history/`: empty. All 7 directories
  hold only `adapter_config.json`. ✓
- T2.1 `results.json.verdict = "KILLED"` (metric-swap + format-artefact). ✓
- DB `experiment list --status killed` shows `exp_p1_t3_plug_and_play_remove`
  status=killed, priority=2. ✓
- MATH.md V2 audit section runs above V1 Setting; V1 KC thresholds intact at
  the bottom of the file. ✓

## Standing rules (7th precondition-probe kill in 24 h)

Class-level standing already established. No new mem-antipattern required:
mem-antipattern-002 (tautological routing), 009 (adapter copy forgery),
and 011 (wrong-object benchmark) all apply as-is. No literature gap → no
ref-add.

## V3 blockers (do not auto-spawn)

T2.1 rebuild (MedQA USMLE 5-choice DB KC #1030, max_tokens ≥ 512, persisted
`.safetensors`, `adapters/code/` created) + T2.6 weights recovered + code
rewrite dropping `REAL_ADAPTER_PATHS[domain]` + genuinely-trained novel
adapters + weight-unload micro-bench (GPU free + mmap close, not `del d[k]`).

## Routing signal for analyst

7th precondition-probe kill. Cluster of T2.1-dependent macros now at 7:
`peer_comparison_llama31_8b`, `peer_comparison_qwen3_4b`, `mtbench_composed`,
`sft_residual_gemma4`, `n25_composition`, `plug_and_play_add`,
`plug_and_play_remove`. Researcher MUST claim a T2.1-independent experiment
or T2.1 V2 itself; any new V3 of these 7 will re-land on the same probe.
