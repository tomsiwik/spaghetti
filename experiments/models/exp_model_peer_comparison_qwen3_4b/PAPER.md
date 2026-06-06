# PAPER — Pierre (Gemma 4 E4B + 5 adapters) vs Qwen 3 4B

## Verdict: KILLED (blocked by prerequisites, not refuted on metrics)

The 5-benchmark head-to-head was not executed. Pre-registered MATH.md §4 requires three preconditions; two fail at probe time and one passes. Identical precondition structure to the Llama 3.1 8B sibling (`exp_model_peer_comparison_llama31_8b`, KILLED 2026-04-18).

## Prediction vs measurement

| Precondition / KC | Predicted requirement | Measured | Status |
|---|---|---|---|
| P1: Pierre adapter weights on disk | 5 `.safetensors` under `adapters/{math,code,medical,sql,bash}/` | 0 present, 5 missing (math/medical/sql/bash have only config/README stubs; code/ directory does not exist) | **FAIL** |
| P2: lm-eval-harness importable | `uv run python -c "import lm_eval"` exit 0 | lm_eval 0.4.11 imports cleanly | **PASS** |
| P3: upstream T2.1 valid (verdict=supported, no metric-swap) | T2.1 results.json verdict=SUPPORTED | verdict=KILLED, metric-swap flagged (MedQA≠MedMCQA in T2.1 audit 2026-04-18) | **FAIL** |
| K1694: Pierre wins ≥3/5 benchmarks | ≥3 wins against Qwen 3 4B | unmeasured (preconditions block) | **FAIL** (cannot PASS without measurement) |
| K1695: thinking mode + matched harness | adapters live AND harness importable AND thinking enabled | harness PASS, adapters absent → thinking-on-adapters is unreachable | **FAIL** |

## Why two preconditions fail

### P1 — missing adapter weights
`adapters/registry.json` enumerates 5 Pierre domain adapters (math, code, medical, sql, bash) with per-benchmark scores (GSM8K 82%, HumanEval 63%, MMLU-Pro 36.1% @ N=1400). Filesystem probe:
- `adapters/math/`, `adapters/medical/`, `adapters/sql/`, `adapters/bash/`: contain only `adapter_config.json`, `chat_template.jinja`, `README.md`, `tokenizer_config.json` — **no `.safetensors` weight files**.
- `adapters/code/`: **directory does not exist**.

The only adapter with weights on disk is `thinking-openthoughts-universal-v0`, which is not one of the 5 referenced by this comparison. The upstream training experiment (`exp_p1_t2_single_domain_training`, T2.1) claims to have produced these adapters on 2026-04-10, but the weights were never committed or were purged. Without weights, "Pierre" reduces to the Gemma 4 E4B 4-bit base — running that against Qwen 3 4B is not the comparison claimed in the experiment title.

### P3 — upstream dependency killed with metric-swap
Per T2.1 PAPER.md V2 audit (2026-04-18): the DB-registered kill criterion K1030 text reads "MedQA", but the code / MATH / PAPER all measure MedMCQA (Indian 4-choice) rather than MedQA (USMLE 5-choice). The K1030 claim against the DB KC is therefore invalid, and T2.1 was flipped from `supported` to `killed`. T2.1 also suffers a `base_gsm8k=0%` baseline that is a `max_tokens=256` CoT-truncation artefact. Any claim that Pierre adapters from T2.1 carry over to a Qwen-sized comparison inherits these audit flags — even if weights were present.

### P2 — passes (confirms Llama-sibling correction)
`lm_eval 0.4.11` imports cleanly under `uv run` from repo root. The 2026-04-18 scratchpad's note about a `datasets/dill` Python 3.14 blocker was code-path-specific (hit `Hasher.hash` on certain dataset types), not a global harness failure.

## Scope of the KILL

This verdict is about **this experiment as specified**, not about Pierre in principle or about Qwen 3 4B. The claim "Pierre competes with Qwen 3 4B" may still be true; we cannot test it because:
- The adapters that would instantiate Pierre do not exist on disk.
- The upstream that would produce them is itself flagged for a benchmark-identity bug.

Qwen 3 4B was chosen precisely because it is the closest matched-param open competitor with first-class thinking mode — so the bar is higher than the Llama 3.1 8B sibling in two ways: (a) matched params removes the "8B > 4B" excuse, (b) thinking-on mode on both sides would test Pierre's runtime-routing advantage, not just adapter content. Both levers are unreachable until P1 and P3 are resolved.

## Remediation (for a v2 experiment)

A rerun requires, in order:
1. **Rebuild adapter weights.** Rerun `exp_p1_t2_single_domain_training` with (a) correct MedQA (USMLE 5-choice) dataset per DB KC #1030, (b) `max_tokens ≥ 512` for CoT baselines, (c) persist `.safetensors` to `adapters/<domain>/` after training, (d) create `adapters/code/` if retaining code as a domain.
2. **Re-validate T2.1.** New verdict must be `supported` on the correct benchmarks before any downstream macro claim can build on it.
3. **Design v2 comparison.** New MATH.md pre-registering K1694/K1695 against the rebuilt adapters; drop the "thinking at eval recovers thinking=False training" fiction — Pierre adapters must be retrained with `thinking_enabled: true` if we want to honestly compare at Qwen's thinking-on bar.

## Assumptions logged (autonomy rule)

- Pierre's 5 adapters are the math/code/medical/sql/bash set in `adapters/registry.json`. If a v2 experiment redefines "Pierre" to include different adapters (e.g., the `math-s1k-reasoning-v0` / `math-star-r1-v0` adapters that DO have weights), the P1 precondition check must be updated.
- Qwen 3 4B is `Qwen/Qwen3-4B` (thinking variant). Not pinned in this iteration — the comparison never ran.
- The audit of T2.1 (2026-04-18) is authoritative; the `results.json` `verdict=KILLED` + `_audit_note` fields were trusted without re-auditing.

## Permanently learned (propagate to siblings)

1. **Macro comparisons must probe preconditions before spinning a heavy sweep.** A 3-second filesystem + import probe catches missing-weights / broken-dep / killed-upstream cases that would otherwise burn hours on a degraded baseline. This is now the **second** confirmation of the Llama sibling's rule — promoting to standing class-level rule for every `exp_model_peer_comparison_*` and `exp_model_mtbench_composed`.
2. **Adapter registry ≠ adapter artefacts.** `registry.json` claims scores and paths; real weights can be missing. Always verify `.safetensors` on disk before treating an adapter as usable. Corollary confirmed here: registry can also reference a domain (`code`) whose directory **does not exist** — directory-existence must be checked before glob.
3. **Downstream P1 macros inherit upstream audit flags.** When an upstream experiment is flipped from supported to killed (metric-swap, format-artefact, etc.), every dependent experiment must recheck its preconditions — even if the file artefacts look unchanged. T2.1's 2026-04-18 flip now blocks at least two P1 macros (Llama sibling + this one). The open `exp_model_mtbench_composed` is the next candidate.

## Why KILLING is the honest verdict

A naive rerunner without the probe would have loaded Gemma 4 base, seen `adapters/math/` exists, loaded only `adapter_config.json`, and produced a measurement that is secretly base-model-only but labeled "Pierre". Per mem-antipattern class "KC measures wrong object", that would be a silent downgrade. The probe catches the load-bearing artefact absence before any benchmark runs, and the KILL verdict is the mechanical consequence of the decision rule.
