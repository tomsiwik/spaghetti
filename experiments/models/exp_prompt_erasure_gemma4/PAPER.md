# PAPER.md — exp_prompt_erasure_gemma4

## Title
Orca-2 Prompt Erasure on Gemma-4-E4B-it-4bit: method injection works on-domain
but collapses out-of-domain knowledge.

## Verdict
**KILLED (smoke; DB status=killed).** K1722 unambiguously falsified by
−30pp MMLU drop; K1721 fails pooled; K1723 passes (structural).
`results.json` verdict = `PROVISIONAL` (smoke), CLI `--status killed`.
`is_smoke=true`, `reuse=true` from `exp_knowledge_disentanglement_control`.

## Abstract
We replicate the Orca-2 Explanation-Tuning + Prompt-Erasure recipe
(Mitra et al., 2311.11045) on Gemma-4-E4B-it-4bit using a rank-16 LoRA
adapter trained (by the sibling experiment
`exp_knowledge_disentanglement_control`) on 25 MMLU-Pro questions with a
4-phase method prompt, then evaluated *without* any system prompt. The
recipe produces a strongly heterogeneous effect: the trained method
signature jumps from 10% → 70% on MMLU (+60pp, on-domain), stays flat
at 75% on GSM8K (already saturated in Gemma-4 native thinking mode),
and *drops* from 75% → 30% on TriviaQA (−45pp, out-of-domain
interference). Pooled invocation rate moves only +5pp (53.3% → 58.3%),
well short of the +20pp bound. In parallel, MMLU accuracy collapses
from 90% → 60% (−30pp, K1722 falsified). The recipe therefore *does*
inject a method on-domain but violates the orthogonality precondition
of LoRA low-rank updates w.r.t. factual circuits — method and content
are not disentangled at rank-16 / N=25.

## Pre-registered predictions vs measurements

| KC | Prediction | Measurement | Verdict |
|---|---|---|---|
| **K1721** | method invocation ≥ 50% AND ≥ base + 20pp | pooled: adapter 58.3% vs base 53.3%, Δ = +5.0pp | **FAIL** (pooled) |
| **K1721** (per-bench diagnostic) | — | MMLU: 70% vs 10% (+60pp); GSM8K: 75% vs 75% (0pp); TriviaQA: 30% vs 75% (−45pp) | mixed |
| **K1722** | \|Δ MMLU\| ≤ 2pp | base 90% → adapter 60%, Δ = −30.0pp | **FAIL** |
| **K1723** | recipe fidelity 3/3 | (a)✓ teacher template; (b)✓ student no-system; (c)✓ eval no-system | **PASS** |
| **all_pass** | — | false | KILLED |

## Evidence table

| Metric | Base (prompt-erased) | Adapter (prompt-erased) | Δ (pp) |
|---|---|---|---|
| Method-invocation rate (pooled, n=60) | 53.3% | 58.3% | +5.0 |
| Method-invocation rate (MMLU, n=20) | 10.0% | 70.0% | +60.0 |
| Method-invocation rate (GSM8K, n=20) | 75.0% | 75.0% | 0.0 |
| Method-invocation rate (TriviaQA, n=20) | 75.0% | 30.0% | −45.0 |
| MMLU accuracy (n=20) | 90.0% | 60.0% | −30.0 |
| GSM8K accuracy (n=20) | 80.0% | 75.0% | −5.0 |
| TriviaQA accuracy (n=20) | 35.0% | 30.0% | −5.0 |

Training loss (sibling): final 0.516, mean 0.760 over 60 steps, 25 seqs
(median ~450 tokens).

## Method
- **Reuse**: this is a re-analysis of the adapter and eval responses
  produced by `exp_knowledge_disentanglement_control` (sibling,
  completed 2026-04-18). The sibling's training pipeline is an exact
  Orca-2 recipe realisation: teacher under `METHOD_SYSTEM_PROMPT` (4
  phases: Restate → Identify → Evaluate → Pick) with
  `enable_thinking=True`; student trained on `{user, assistant}`
  message pairs with the system prompt erased; eval inference uses
  `[{"role":"user", ...}]` only. `K1723` verifies these three
  properties structurally.
- **New metric (K1721)**: method invocation = `≥1 lexicon hit` (regex
  on 6 method-specific tokens: *restate*, *identify relevant*,
  *evaluate option*, *thinking process*, *step 1*, *subgoal*) AND
  `≥2 numbered-step lines` (`^[1-4][.)]` at line start).
- **Signature parse**: computed on the 300-char `resp_prefix` stored
  in the sibling's `data/eval_responses.jsonl` (60 base + 60 adapter).
- **K1722**: reused directly from sibling `results.json`
  (`eval_base.mmlu.acc` vs `eval_adapter.mmlu.acc`).

## Failure analysis
1. **Native base method rate (F3 guard).** Gemma-4's thinking mode
   already emits a numbered-step rationale on ~53% of queries
   (averaged across benches). This eats almost half of the 50%
   absolute bound. K1721 was set with this in mind and required
   +20pp over base to exceed the native rate; pooled +5pp fails
   decisively.
2. **On-domain method injection does work.** MMLU invocation goes from
   10% → 70% (+60pp). The adapter clearly internalises the
   subgoal-decomposition template when the surface form matches the
   training distribution.
3. **Out-of-domain suppression.** TriviaQA invocation *drops* from 75%
   → 30% (−45pp). Training on MCQ + method destabilises the
   numbered-step prior on short-form factual queries. This looks like
   a format-transfer failure: the adapter associates "method" with
   the MCQ question template, not with reasoning in general.
4. **Knowledge collapse (K1722).** MMLU accuracy −30pp under the same
   prompt-erased inference. LoRA rank-16 update is not orthogonal to
   MMLU factual circuits at N=25 / 60 steps. The method signal leaks
   into the content representation.

## Why the recipe failed here
Orca-2 was trained on 817K ChatML rationale traces and evaluated on a
base Mistral-7B that lacked a native method prior. The smoke here is
~30 000× smaller (25 vs 817K), evaluates on a base model with a
*strong* native method prior (Gemma-4 thinking mode), and restricts to
rank-16 on two attention projections. Under these constraints the
adapter simultaneously (i) shifts surface template on-domain, (ii)
fails to shift it off-domain, (iii) degrades factual recall. None of
the three is consistent with the orthogonal-method hypothesis.

## Implications for future experiments
- **Orca-2 recipe on Gemma-4 requires a base-method subtraction.** Any
  K measuring "method invocation" must compare adapter rate to the
  base's native thinking-mode rate, domain-by-domain.
- **Pooled rates hide per-domain effects.** Retire pooled rates as
  primary KCs for behavioural signature tests; use per-domain with
  explicit train-distribution marker.
- **Method injection ≠ method generalisation.** Even a +60pp MMLU
  lift does not imply a generalised method; TriviaQA shows the
  opposite.
- **Rank-16 / N=25 on v_proj+o_proj is not knowledge-neutral.**
  Consistent with `exp_knowledge_disentanglement_control` K1734 fail
  and with the broader `method-adapter-damages-knowledge` antipattern.

## Limitations
- Smoke scale (N_STEPS=60, n_train=25, eval n=20 per bench).
- Method signature measured on 300-char response prefixes; late-
  occurring numbered steps missed. However, teacher template is
  front-loaded, so prefix sampling is likely adequate for this
  signature.
- Single seed (seed=42). Sampling error on method rate at n=20 per
  bench is ≈ ±20pp at 95% CI.
- cais/mmlu proxy for MMLU-Pro; MMLU-Pro reruns required before any
  strong knowledge-preservation claim.

## Full-scale rerun plan (deferred)
If a future `exp_prompt_erasure_gemma4_v2` is scheduled:
1. Retrain at n_train ≥ 500 per category, N_STEPS ≥ 1500, seeds {42, 43, 44}.
2. Store full-length responses (not 300-char prefixes) and recompute
   signature.
3. Eval matrix: in-domain (MMLU-Pro + MATH), out-of-domain (TriviaQA
   + MuSR), base-subtracted per-domain invocation rates.
4. Compare to a `base-method-subtracted` adapter trained only on
   queries where base's native method rate < 20% (reduces F3
   contamination).

## Assumptions (from MATH.md A1–A5)
- A1 ✓ sibling implements Orca-2 recipe faithfully.
- A2 ✓ method signature = lexicon + ≥2 numbered-step lines.
- A3 ✗ 300-char prefix may miss late method markers; only the 60
  MMLU + 60 non-MMLU prefixes were inspected.
- A4 ✓ cais/mmlu at n=20 is a noisy but unbiased MMLU-Pro proxy;
  −30pp is outside the 95% CI.
- A5 ✓ smoke budget declared; full-rerun plan pre-registered above.

## Antipattern pre-flight (all clean except the deliberate reuse)
- composition bug — N/A.
- tautological routing — N/A.
- unsafe LoRA scale — scale=4.0 (< 8).
- KC-swap-after-failure — K1721/2/3 verbatim from DB.
- verdict-DB mismatch — results.json=PROVISIONAL, CLI --status killed
  (K1722 falsified unambiguously); the CLI pointer is the mapping
  convention for smoke falsifications.
- smoke-as-full — declared `is_smoke=true`.
- thinking-mode truncation — N/A; both arms use identical chat
  templates.
- proxy-model-substituted — no proxy model.
- shutil.copy-as-new-adapter — reuse is explicit, not adapter-
  cloning. `sibling_adapter_path` is a provenance pointer.
- hardcoded pass: True — all KC pass flags are computed.
- copy-paste scaffolding — no sibling code copied; we import no
  sibling modules.

## Artefacts
- `MATH.md` (frozen KCs 1721/1722/1723).
- `run_experiment.py` (re-analysis, no model load).
- `results.json` (this run).
- `PAPER.md` (this file).
- Sibling-referenced artefacts remain in
  `micro/models/exp_knowledge_disentanglement_control/`.
