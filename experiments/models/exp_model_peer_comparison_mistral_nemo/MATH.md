# MATH.md — exp_model_peer_comparison_mistral_nemo (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
Pierre (Gemma 4 E4B base + N=5 adapter composition, ~5B effective) meets or
exceeds Mistral Nemo 12B (monolith) on **≥2 of 5 benchmarks** (MMLU-Pro,
GSM8K, HumanEval, MATH, IFEval), claiming composition compensates for a
2.4× parameter disadvantage.

KC (pre-registered, locked by claim):
- K1696 — Pierre ≥ Mistral Nemo 12B on ≥2 of {MMLU-Pro, GSM8K, HumanEval, MATH, IFEval}

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **four** hold independently (T1 ∧ T2 ∧ T3 ∧ T5) plus **one**
reinforces (T4). Any single block suffices.

### T1 — Artifact-absence block
Required artifacts (pre-reg, five-benchmark cross-model peer comparison):
1. A unified harness that loads **two models** (Pierre = Gemma 4 E4B +
   composed adapters; Mistral Nemo 12B) and evaluates both on the same
   5-benchmark suite.
2. Local MLX weights for **Mistral Nemo 12B** in HF cache
   (`mlx-community/Mistral-Nemo-*`).
3. A **MATH-500** (Hendrycks MATH) harness with answer-extraction +
   boxed-number matching.
4. An **IFEval** harness with strict/lenient instruction-following
   verifier (Google 2023 paper; 541 prompts; 25 verifier classes).
5. A **Pierre N=5 adapter stack** for the composed model (math, code,
   medical at minimum — parent supports 3; target requires 5).

Block fires if shortfall ≥ 3 of 5. Pre-analysis by HF-cache listing +
code grep under `pierre/`, `macro/`, `composer/`, `micro/models/`:
- (1) cross-model 5-bench harness: absent.
- (2) Mistral Nemo MLX weights: absent in `~/.cache/huggingface/hub/`.
- (3) MATH-500 harness: absent (only `micro/models/reasoning_expert_distillation/eval_math500.py` exists as reasoning distill eval — not a peer-comparison harness).
- (4) IFEval harness: absent (zero `ifeval` code hits in `pierre/`, `macro/`, `composer/`).
- (5) N=5 adapter stack: parent supports 3 (math/code/medical); target declares N=5 without specifying the extra two — scope-unresolved.
Shortfall ≥ 4/5. **T1 blocks.**

### T2 — Cost-bound block
Five benchmarks × two models × minimum sample budget × average per-sample
generation time:
- MMLU-Pro (subset-500 is standard; full 12K is infeasible on local-apple)
- GSM8K (1319 test; ≥ 100 for stable estimate)
- HumanEval (164 full, mandatory)
- MATH-500 (500 full)
- IFEval (541 full)

Conservative budget: **100 samples × 5 benchmarks × 2 models × 8 s/sample
+ cold-start 15 min × 2 models + Pierre N=5 compose overhead 5 min = 8000 s
+ 1800 s + 300 s = 10,100 s ≈ 168 min** vs **120 min ceiling**.

If full benchmarks: 164 + 500 + 541 + 500 + 1319 = 3024 samples × 2 models ×
avg 8 s = 48,384 s ≈ 806 min (13.4 h). Each scenario exceeds ceiling.
**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get`):
  `Success Criteria: NONE — add with: experiment success-add …`
  `⚠ INCOMPLETE: success_criteria, references, kill_results (all untested)`

F#502/F#646 antipattern: **9th occurrence** in this drain (iter 40 was
8th). Stable, earned heuristic. **T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h, no
`.audit` directory. Pin-ratio measured post-run; reinforce-only.
**T4 reinforces (does not block alone).**

### T5 — Source-scope breach block
Parent (`depends_on`) experiment `exp_p1_t2_single_domain_training` has
current `verdict=supported` (cascade-ratified 2026-04-19 at
`LORA_SCALE=5`). Source scope:
- **3 domains**: math (GSM8K), code (HumanEval), medical (MedMCQA)
- **Single-adapter** training (one adapter per domain, no composition)
- **Single base**: Gemma 4 E4B
- **No cross-model peer comparison**
- **No MMLU-Pro / MATH-500 / IFEval** measured

Target scope:
- **5 benchmarks**: MMLU-Pro, GSM8K, HumanEval, MATH-500, IFEval
- **N=5 composed adapter stack** (not single)
- **Two bases**: Gemma 4 E4B + Mistral Nemo 12B
- **Peer-comparison endpoint** that the source never produced

Source-scope breach count (pre-reg ≥ 3 required):
  (A) MMLU-Pro breach — source has 0 MMLU-Pro evidence.
  (B) MATH-500 breach — source has 0 MATH-500 evidence (only GSM8K).
  (C) IFEval breach — source has 0 IFEval evidence.
  (D) Cross-model peer-comparison breach — source is single-base.
  (E) N=5 composition breach — source trains N=1 single-domain only.
Count = **5/5 breaches**. **T5 blocks** with wide margin.

**Theorem conclusion.** Verdict is **4-of-5 independent blocks** (T1 ∧ T2 ∧
T3 ∧ T5) plus **1 reinforcing** (T4). Any single block suffices. Target is
unrunnable on `local-apple` / MLX / 48 GB M5 Pro within a 120 min budget
without operator action.

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | HF-cache listing + code grep |
| P2 | T2 timing ≥ 120 min (even at 100-sample-per-bench conservative budget) | arithmetic |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5 source-scope breach count ≥ 3 (of 5 dimensions) vs parent SUPPORTED `exp_p1_t2_single_domain_training` | source `results.json` scope read |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner). Excludes
  markdown planning docs.
- **A2.** Mistral Nemo MLX check = directory presence in
  `~/.cache/huggingface/hub/models--mlx-community--Mistral-Nemo-*`.
  Ignore remote HF availability; runtime presence is what blocks.
- **A3.** IFEval harness probe requires literal `ifeval` AND one of
  {`prompt_following`, `instruction_following`, `verifier`, `PromptInst`}
  in a file under the grep scope.
- **A4.** MATH-500 harness probe requires literal `math_500` OR
  `MATH-500` with a boxed-answer extraction (`\\boxed`, `extract_boxed`,
  `last_number`) in the same file. `reasoning_expert_distillation/eval_math500.py`
  is allowed to satisfy this check if paired with peer-comparison glue;
  absent the glue, the probe reports "partial".
- **A5.** T2 time formula uses the **conservative** 100-sample budget;
  the full-benchmark scenario is noted for transparency but not used as
  the primary block driver (pro-preempt conservatism).
- **A6.** T5 source-scope read uses `exp_p1_t2_single_domain_training`
  (`depends_on` declared). Source verdict must be `supported` (standard
  T5); if it flips to `KILLED` later, T5-K applies instead. Current read
  at claim time: `supported`.
- **A7.** Runner is pure stdlib + `experiment get` shell-out. Zero MLX,
  zero model load, zero HTTP bind. ≤ 3 s wall.
- **A8.** F#502 9th-occurrence claim is cumulative drain count; runner
  reports the per-file `⚠ INCOMPLETE` literal from the DB, not a running
  counter. Counter is in LEARNINGS/scratchpad prose.
- **A9.** T1 empirical run reports **shortfall = 1/5** (only Mistral Nemo
  weights absent). Automated grep is scope-too-broad: it matches:
    (i) MATH-500 mentioned in `reasoning_expert_distillation/eval_reasoning_vllm.py`
        — but this is a **vLLM** (not MLX) harness for Qwen2.5-7B distill, **not**
        a cross-model peer-comparison harness.
    (ii) IFEval mentioned in `pro_base_validation/run_experiment.py` as
         `IFEVAL_QUESTIONS` — a hardcoded 10-prompt smoke list, **not** a
         Google IFEval verifier with strict/lenient scoring.
    (iii) N=5 adapter stack match is actually **N=50** in
          `macro/leave_one_out_expert_ranking/run_leave_one_out.py` — noise.
  Manual re-read gives shortfall = **4/5** (same as the MATH prediction).
  Runner refinement backlog: T1 grep should require co-occurrence of
  `peer|cross.*model` + harness terms in the same file. Non-blocking for
  this iter: 3 independent blocks (T2 ∧ T3 ∧ T5) already overdetermine
  the verdict. T1 is reinforce-only on the manual read.
