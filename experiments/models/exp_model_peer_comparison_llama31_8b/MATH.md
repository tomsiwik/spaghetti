# MATH — Pierre (Gemma 4 E4B + 5 adapters) vs Llama 3.1 8B on 5 benchmarks

## 1. Claim
Pierre (Gemma 4 E4B 4-bit + N=5 domain adapters) meets or exceeds Llama 3.1 8B on ≥3/5 of MMLU-Pro, GSM8K, HumanEval, MATH, IFEval under matched lm-eval-harness configs.

## 2. Failure mode
"Competes with giants at similar-param budget" is the rhetorical claim. The falsifying failure mode: Pierre cannot be honestly instantiated because (a) adapter weights are missing, (b) matched-config harness is broken, or (c) upstream T2.1 (the adapter training run) is not validly `supported`. Any of these ≡ verdict KILLED; the claim is unconfirmed, not refuted — but for a P1 head-to-head we require positive evidence, not silence.

## 3. Prior math / grounding
- Gemma 4 E4B: reported MMLU-Pro 62.1% with thinking (Google cards, cited in PLAN.md Part 2).
- Llama 3.1 8B IT: reported GSM8K 84.5%, MMLU 69.4%, HumanEval 72.6% (Meta model card). No official MMLU-Pro, but community harness runs cluster 35–40%.
- Pierre v3 behavioral score: 0.41 (project memory `project_pierre_v5_architecture`).
- T2.1 single-domain training (`exp_p1_t2_single_domain_training`) is the upstream adapter producer. Its status is load-bearing for every claim about Pierre's domain specialization.

## 4. Theorem (operational)
**Theorem.** For Pierre (Gemma 4 E4B + 5 adapters) to match Llama 3.1 8B on a benchmark `b`, three operational preconditions must simultaneously hold:
1. `adapter_weights(i).safetensors` is present on disk for each of the 5 adapters referenced in `adapters/registry.json` (math, code, medical, sql, bash).
2. The evaluation harness resolves its dataset dependencies without error (lm-eval-harness → `datasets` → `dill` must be importable under the platform Python).
3. The upstream training experiment that produced those adapters has verdict=supported with matching benchmark protocol (same dataset definition as evaluated here — cf. the MedQA ≠ MedMCQA metric-swap in T2.1 audit).

**Proof.** (1) Without weights, there is no "Pierre" — only a base model. (2) Without a working harness, there is no matched-config measurement, so KC #1692 cannot be met. (3) Without valid upstream, any observed gain could be artefactual (e.g., base=0% from eval-template truncation inflating reported delta, as in the T2.1 audit). QED.

**Corollary.** If any of (1), (2), (3) fail, the experiment result is "blocked by prerequisites", recorded as `killed` against K1691 (cannot be measured positively). This is **not** a refutation of Pierre's capability — it is a refutation of the claim "this experiment, as specified, reports a meaningful comparison at time of running."

## 5. Predictions (pre-registered, locked before run)

| Precondition | Expected | If measured |
|---|---|---|
| P1: adapter weights present | Pierre 5 adapters in `adapters/{math,code,medical,sql,bash}/adapters.safetensors` | skipped if P1 fails |
| P2: harness imports | `uv run lm_eval --help` returns 0 | skipped if P2 fails |
| P3: upstream valid | T2.1 verdict=supported AND no metric-swap audit flag | skipped if P3 fails |
| Pierre MMLU-Pro | 62.1% ± 2 (Gemma 4 baseline; adapters trained with `thinking=False` per registry, so no uplift expected) | only if P1–P3 |
| Pierre GSM8K | 82.0% ± 3 (from registry math adapter self-reported score) | only if P1–P3 |
| Pierre HumanEval | 63.0% ± 3 (from registry code adapter self-reported) | only if P1–P3 |
| Pierre MATH | unknown, no prior measurement; expect ≤ 35% | only if P1–P3 |
| Pierre IFEval | unknown; expect ~70% (Gemma 4 IT baseline) | only if P1–P3 |
| Llama 3.1 8B MMLU-Pro | 35–40% (community) | only if P1–P3 |
| Llama 3.1 8B GSM8K | 84.5% (model card) | only if P1–P3 |
| Llama 3.1 8B HumanEval | 72.6% (model card) | only if P1–P3 |
| Llama 3.1 8B MATH | ~51% (model card) | only if P1–P3 |
| Llama 3.1 8B IFEval | ~80% (model card) | only if P1–P3 |

## 6. Kill criteria (pre-registered; DO NOT edit post-run)

- **K1691 (DB id 1691): Pierre meets or exceeds Llama 3.1 8B on ≥3 of 5** — PASS requires honest measurement on all 5 benchmarks with Pierre's 5-adapter stack live. Prerequisites-not-met ⇒ **cannot pass**.
- **K1692 (DB id 1692): Evals run via lm-eval-harness with matched configs** — PASS requires harness importable and both models scored under identical (shots, temperature, seed, max_tokens) config. Harness-broken ⇒ **FAIL**.
- **K1693 (DB id 1693): Thinking mode enabled for Pierre where applicable** — PASS requires `enable_thinking=True` in Gemma 4 chat template at eval time. Registry shows adapters trained with `thinking_enabled: false`, so thinking-mode Pierre is untrained; we mark this as PASS iff harness runs with thinking enabled (regardless of whether it helps), FAIL otherwise.

## 7. Decision rule
- All three preconditions hold AND Pierre wins ≥3/5 ⇒ **supported**.
- All three preconditions hold AND Pierre wins < 3/5 ⇒ **killed** (honest refutation).
- Any precondition fails ⇒ **killed** (blocked; future-rerun tag).

## 8. Notes
- No KC edited between this file and completion. If a future rerun changes KCs, design a v2 experiment.
- `mlx-lm` version and Llama 3.1 8B MLX conversion pinned in `run_experiment.py` when/if the blockers are resolved.
