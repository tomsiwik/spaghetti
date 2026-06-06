# MATH — Pierre (Gemma 4 E4B + 5 adapters) vs Qwen 3 4B on 5 benchmarks

## 1. Claim
Pierre (Gemma 4 E4B 4-bit + N=5 domain adapters) meets or exceeds Qwen 3 4B on ≥3/5 of MMLU-Pro, GSM8K, HumanEval, MATH, IFEval under matched lm-eval-harness configs with thinking mode enabled where supported (K1694, K1695).

Qwen 3 4B is the closest matched-parameter open competitor with first-class thinking mode, so this head-to-head is the most honest apples-to-apples comparison in the peer set (registry note on the experiment DB row).

## 2. Failure mode
"Pierre competes with a matched-param thinking model" is the rhetorical claim. The falsifying failure mode: Pierre cannot be honestly instantiated because (a) adapter weights are missing, (b) matched-config harness is broken, or (c) upstream T2.1 (the adapter training run) is not validly `supported`. Any of these ≡ verdict KILLED; the claim is unconfirmed, not refuted — but for a P1 head-to-head we require positive evidence, not silence.

## 3. Prior math / grounding
- Gemma 4 E4B: reported MMLU-Pro 62.1% with thinking (Google model card, cited in PLAN.md Part 2).
- Qwen 3 4B (thinking): Qwen team reports MMLU-Pro ≈56.7%, GSM8K ≈85%, HumanEval ≈72%, MATH ≈69%, IFEval ≈78% on thinking-on mode (Qwen3 Technical Report; community harness runs within ±3pp).
- Pierre v3 behavioral score: 0.41 (project memory `project_pierre_v5_architecture`).
- T2.1 single-domain training (`exp_p1_t2_single_domain_training`) is the upstream adapter producer. Its status is load-bearing for every claim about Pierre's domain specialization.
- Sibling `exp_model_peer_comparison_llama31_8b` (completed 2026-04-18, verdict KILLED) established the precondition-probe template and found P1+P3 fail on the current repo state.

## 4. Theorem (operational)
**Theorem.** For Pierre (Gemma 4 E4B + 5 adapters) to match Qwen 3 4B on a benchmark `b`, three operational preconditions must simultaneously hold:
1. `adapter_weights(i).safetensors` is present on disk for each of the 5 adapters referenced in `adapters/registry.json` (math, code, medical, sql, bash).
2. The evaluation harness resolves its dataset dependencies without error (lm-eval-harness → `datasets` → `dill` must be importable under the platform Python).
3. The upstream training experiment that produced those adapters has verdict=supported with matching benchmark protocol (same dataset definition as evaluated here — cf. the MedQA ≠ MedMCQA metric-swap in T2.1 audit).

**Proof.** (1) Without weights, there is no "Pierre" — only a base model. (2) Without a working harness, there is no matched-config measurement, so K1695 cannot be met. (3) Without valid upstream, any observed gain could be artefactual (e.g., base=0% from eval-template truncation inflating reported delta, as in the T2.1 audit). QED.

**Corollary.** If any of (1), (2), (3) fail, the experiment result is "blocked by prerequisites", recorded as `killed` against K1694 (cannot be measured positively). This is **not** a refutation of Pierre's capability — it is a refutation of the claim "this experiment, as specified, reports a meaningful comparison at time of running."

## 5. Predictions (pre-registered, locked before run)

| Precondition / Measurement | Expected | If measured |
|---|---|---|
| P1: adapter weights present | Pierre 5 adapters in `adapters/{math,code,medical,sql,bash}/*.safetensors` | skipped if P1 fails |
| P2: harness imports | `uv run python -c "import lm_eval"` returns 0 | skipped if P2 fails |
| P3: upstream valid | T2.1 verdict=supported AND no metric-swap audit flag | skipped if P3 fails |
| Pierre MMLU-Pro | 62.1% ± 2 (Gemma 4 baseline; adapters trained with `thinking=False` per registry, so no uplift expected at adapter layer) | only if P1–P3 |
| Pierre GSM8K | 82.0% ± 3 (from registry math adapter self-reported score) | only if P1–P3 |
| Pierre HumanEval | 63.0% ± 3 (from registry code adapter self-reported) | only if P1–P3 |
| Pierre MATH | unknown, no prior measurement; expect ≤ 35% | only if P1–P3 |
| Pierre IFEval | unknown; expect ~70% (Gemma 4 IT baseline) | only if P1–P3 |
| Qwen 3 4B MMLU-Pro | 56.7% ± 3 (thinking-on) | only if P1–P3 |
| Qwen 3 4B GSM8K | 85.1% ± 3 | only if P1–P3 |
| Qwen 3 4B HumanEval | 72.6% ± 3 | only if P1–P3 |
| Qwen 3 4B MATH | 69.2% ± 3 | only if P1–P3 |
| Qwen 3 4B IFEval | 78.0% ± 3 | only if P1–P3 |

## 6. Kill criteria (pre-registered; DO NOT edit post-run)

- **K1694 (DB id 1694): Pierre meets or exceeds Qwen 3 4B on ≥3 of 5** — PASS requires honest measurement on all 5 benchmarks with Pierre's 5-adapter stack live. Prerequisites-not-met ⇒ **cannot pass**.
- **K1695 (DB id 1695): Both use thinking mode where supported AND matched eval harness configs** — PASS requires (a) `enable_thinking=True` at eval time for both models, (b) harness importable with identical (shots, temperature, seed, max_tokens) per benchmark. Harness-broken ⇒ **FAIL**. Registry shows Pierre adapters trained with `thinking_enabled: false`, so thinking-mode Pierre is structurally untrained; even if harness runs, the comparison's "thinking" leg for Pierre is vestigial. We mark K1695 as PASS iff harness importable AND adapters on disk AND thinking-at-eval configured (all three). Absent adapters, PASS is unreachable regardless of harness.

## 7. Decision rule
- All three preconditions hold AND Pierre wins ≥3/5 ⇒ **supported**.
- All three preconditions hold AND Pierre wins < 3/5 ⇒ **killed** (honest refutation).
- Any precondition fails ⇒ **killed** (blocked-by-prerequisites; future-rerun tag).

## 8. Notes
- No KC edited between this file and completion. If a future rerun changes KCs, design a v2 experiment.
- `mlx-lm` version and Qwen 3 4B MLX conversion would be pinned in `run_experiment.py` when/if the blockers are resolved; currently out of scope because the probe blocks before any model load.
- This MATH.md intentionally mirrors the Llama 3.1 8B sibling's structure (same preconditions, same theorem) because the failure mode is load-bearing for the entire `exp_model_peer_comparison_*` class. Divergence limited to the DB KC ids and the target-model reference numbers.
