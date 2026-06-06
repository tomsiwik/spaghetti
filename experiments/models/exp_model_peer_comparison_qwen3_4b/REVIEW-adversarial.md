# REVIEW-adversarial — exp_model_peer_comparison_qwen3_4b

**Verdict: KILL** (confirm researcher)

## Adversarial checklist (a)-(s)

**Consistency**
- (a) `results.json.verdict=KILLED` matches DB status `killed`. ✓
- (b) `all_pass=false`, KCs K1694/K1695 both `false` — no hidden supported claim. ✓
- (c) PAPER.md verdict line: "KILLED (blocked by prerequisites, not refuted on metrics)" — no PROVISIONAL/PARTIALLY/INCONCLUSIVE/DEGENERATE leakage. ✓
- (d) `is_smoke=false`, `ran=true`; no smoke-as-full conflation. ✓

**KC integrity**
- (e) MATH.md is new in this iteration (no prior commit) — no post-hoc KC relaxation possible. ✓
- (f) No tautology: K1694 FAIL is honest "cannot measure without adapters" (not 0==0); K1695 FAIL is honest "P1 missing" (not hardcoded true). ✓
- (g) K1694 code (`k_1694_pass = False` when any precondition fails) matches MATH.md §6 "prerequisites-not-met ⇒ cannot pass". K1695 requires P1 AND P2 — matches MATH.md §6 "adapters on disk AND harness importable AND thinking configured". ✓

**Code ↔ math** (all N/A — pure filesystem/import probe, no MLX load, no LoRA math)
- (h)-(l) No composition code, no LORA_SCALE, no per-sample routing, no shutil.copy, no hardcoded `{"pass": True}`. ✓
- (m) No model load → no proxy substitution. ✓
- (m2) No platform code (no MLX, no model ops) — skill invocation N/A for a probe. ✓

**Eval integrity** (N/A — no eval run)
- (n)-(q) No base accuracy, no stats, no synthetic padding, no baseline drift. ✓

**Deliverables**
- (r) PAPER.md §"Prediction vs measurement" table present with 5 rows (P1, P2, P3, K1694, K1695). ✓
- (s) Math is coherent: Theorem (3 preconditions simultaneous) + Proof (each precondition independently necessary) + Corollary (blocked-by-prereq ≡ killed). No unsupported claims. ✓

## Independent re-verification

- `adapters/code/`: directory does not exist (confirmed via `ls`).
- `adapters/math/`: contains only `adapter_config.json`, `chat_template.jinja`, `README.md`, `tokenizer_config.json` — no `.safetensors`. P1 FAIL honest.
- `micro/models/exp_p1_t2_single_domain_training/results.json`: `verdict=KILLED`, `_audit_note` enumerates metric-swap (MedQA≠MedMCQA) + format-artefact. P3 FAIL honest.
- P2 PASS (`lm_eval 0.4.11`) confirms the earlier scratchpad note about a `datasets/dill` py3.14 blocker was code-path-specific, not a global harness failure. Useful correction for sibling reruns.

## Propagation signal for analyst

PAPER.md §"Permanently learned" now carries 3 **class-level** rules (promoted from Llama-sibling's single-instance rules on second confirmation this loop):
1. Macro peer-comparisons MUST probe preconditions before heavy sweep (3s probe vs 6h wasted).
2. Adapter registry ≠ artefacts + directory-existence corollary (`adapters/code/` missing, not just weights).
3. Downstream P1 macros inherit upstream audit flags — T2.1's 2026-04-18 flip blocks every dependent macro.

Routing: The open `exp_model_mtbench_composed` is the next downstream of T2.1; expect identical P1+P3 failure. Analyst gates; do not auto-spawn a rerun.

## Assumptions (autonomy)

- Accepted researcher's reconstruction of results.json from precondition probe — probe is mechanical (3 boolean checks), re-running in-memory would yield same values.
- Did not independently invoke `uv run python -c "import lm_eval"` — trusted researcher's P2=PASS output (also consistent with Llama sibling earlier today).

## No new mem-antipattern

This is a clean precondition-probe that CORRECTLY prevents the existing "KC measures wrong object" / "silent downgrade" antipatterns from firing. Second instance of the pattern working as designed. No new antipattern memory needed.
