# PAPER — Pierre (Gemma 4 E4B + 5 adapters) vs Llama 3.1 8B

## Verdict: KILLED (blocked by prerequisites, not refuted on metrics)

The 5-benchmark head-to-head was not executed. Pre-registered MATH.md §4 requires three preconditions; two fail at probe time and one passes.

## Prediction vs measurement

| Precondition | Predicted requirement | Measured | Status |
|---|---|---|---|
| P1: Pierre adapter weights on disk | 5 `.safetensors` under `adapters/{math,code,medical,sql,bash}/` | 0 present, 5 missing (only config/README stubs) | **FAIL** |
| P2: lm-eval-harness importable | `uv run python -c "import lm_eval"` exit 0 | lm_eval 0.4.11 imports cleanly | **PASS** |
| P3: upstream T2.1 valid (verdict=supported, no metric-swap) | T2.1 results.json verdict=SUPPORTED | verdict=KILLED, metric-swap flagged (MedQA≠MedMCQA in T2.1 audit 2026-04-18) | **FAIL** |
| K1691: Pierre wins ≥3/5 benchmarks | ≥3 wins against Llama 3.1 8B | unmeasured (preconditions block) | **FAIL** (cannot PASS without measurement) |
| K1692: lm-eval-harness matched configs | harness importable + matched seeds | harness importable (P2 PASS); no benchmarks run | **PASS** (import-only half) |
| K1693: thinking mode enabled | eval uses `enable_thinking=True` | moot; no eval performed. Registry shows Pierre adapters trained with `thinking_enabled: false`, so thinking at eval cannot recover capability not in training | **FAIL** |

## Why two preconditions fail

### P1 — missing adapter weights
`adapters/registry.json` enumerates 5 Pierre domain adapters (math, code, medical, sql, bash) with per-benchmark scores (GSM8K 82%, HumanEval 63%, MMLU-Pro 36.1% @ N=1400). Each of the 5 directories on disk contains only `adapter_config.json`, `chat_template.jinja`, `README.md`, `tokenizer_config.json` — **no `.safetensors` weight files**. The single adapter with weights on disk is `thinking-openthoughts-universal-v0`, which is not one of the 5 referenced by this comparison.

The upstream training experiment (`exp_p1_t2_single_domain_training`) claims to have produced these adapters in 2026-04-10, but the weights were never committed or were purged. Without weights, "Pierre" reduces to the Gemma 4 E4B 4-bit base — running that against Llama 3.1 8B is not the comparison claimed in the experiment title.

### P3 — upstream dependency killed with metric-swap
Per scratchpad and T2.1 PAPER.md V2 audit (2026-04-18): the DB-registered kill criterion K1030 text reads "MedQA", but the code/MATH/PAPER all measure MedMCQA (Indian 4-choice) rather than MedQA (USMLE 5-choice). The K1030 claim against the DB KC is therefore invalid, and T2.1 was flipped from `supported` to `killed`. The `base_gsm8k=0%` baseline in T2.1 is also a `max_tokens=256` CoT-truncation artefact per the audit. Any claim that Pierre adapters from T2.1 carry over to a Llama-sized comparison inherits these audit flags — even if weights were present.

### P2 — passes (correction to prior scratchpad belief)
The 2026-04-18 scratchpad noted a `datasets/dill` Python 3.14 incompatibility blocking other experiment reruns. This probe confirms `lm_eval 0.4.11` imports cleanly under `uv run` from this repo root. The prior blockers likely hit specific code paths (e.g., `Hasher.hash` on certain dataset types) rather than import. Not a global harness failure — load path matters.

## Scope of the KILL

This verdict is about **this experiment as specified**, not about Pierre in principle. The claim "Pierre competes with Llama 3.1 8B" may still be true; we cannot test it because:
- The adapters that would instantiate Pierre do not exist on disk.
- The upstream that would produce them is itself flagged for a benchmark-identity bug.

## Remediation (for a v2 experiment)

A rerun requires, in order:
1. **Rebuild adapter weights.** Rerun `exp_p1_t2_single_domain_training` with (a) correct MedQA (USMLE 5-choice) dataset per DB KC #1030, (b) `max_tokens ≥ 512` for CoT baselines, (c) persist `.safetensors` to `adapters/<domain>/` after training.
2. **Re-validate T2.1.** New verdict must be `supported` on the correct benchmarks before any downstream macro claim can build on it.
3. **Design v2 comparison.** New MATH.md pre-registering K1691 against the rebuilt adapters; drop the dependency on `thinking_enabled=False` training (Pierre adapters currently cannot benefit from thinking mode since they weren't trained with it — K1693 is structurally unreachable on current artefacts).

## Assumptions logged (autonomy rule)

- Pierre's 5 adapters are the math/code/medical/sql/bash set in `adapters/registry.json`. If a v2 experiment redefines "Pierre" to include different adapters, the P1 precondition check must be updated.
- Llama 3.1 8B is "meta-llama/Llama-3.1-8B-Instruct" or its MLX 4-bit conversion. Not pinned in this iteration — the comparison never ran.
- The audit of T2.1 (2026-04-18) is authoritative; the `results.json` `verdict=KILLED` + `_audit_note` fields were trusted without re-auditing.

## Permanently learned (propagate to siblings)

1. **Macro comparisons must probe preconditions before spinning a 6-hour sweep.** A 3-second filesystem + import probe catches missing-weights / broken-dep / killed-upstream cases that would otherwise burn hours on a degraded baseline. Applies to all `exp_model_peer_comparison_*` and `exp_model_mtbench_composed` siblings.
2. **Adapter registry ≠ adapter artefacts.** `registry.json` can claim scores and paths; real weights can be missing. Always verify `.safetensors` on disk before treating an adapter as usable.
3. **Downstream P1 macros inherit upstream audit flags.** When an upstream experiment is flipped from supported to killed (metric-swap, format-artefact, etc.), every dependent experiment must recheck its preconditions — even if the file artefacts look unchanged.
