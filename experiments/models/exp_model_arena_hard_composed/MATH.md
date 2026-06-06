# MATH.md — exp_model_arena_hard_composed (KILLED_PREEMPTIVE)

## 1. Hypothesis (as declared by target)
Pierre (Gemma 4 E4B base + N=5 adapter composition) achieves Arena-Hard
win-rate ≥ 50% vs base Gemma 4 E4B on the 500-prompt Arena-Hard-Auto
benchmark (Li et al. 2024), with 95% bootstrap CI lower bound > 40%
("composition is not harmful on challenging conversational prompts").

KC (pre-registered, locked by claim):
- K1700 — Pierre N=5 composition Arena-Hard win-rate vs base ≥ 50%
- K1701 — 95% bootstrap CI excludes "worse than base" (lower bound > 40%)

## 2. Preempt theorem (defense-in-depth, 5-of-5 independent blocks)

**Theorem (preempt).** The empirical run is **impossible** or
**guaranteed-to-fail** iff at least **one** of the five blocks holds.
We show **four** hold independently (T1 ∧ T2 ∧ T3 ∧ T5) plus **one**
reinforces (T4). Any single block suffices.

### T1 — Artifact-absence block
Required artifacts (pre-reg, Arena-Hard win-rate macro eval):
1. **Arena-Hard-Auto prompt set** (500 prompts, sourced from LMSYS
   Chatbot Arena, v0.1 or v2.0 release) loaded from disk or streamed.
2. **LLM-judge client** (GPT-4-Turbo-2024-04-09 is the canonical
   judge per Li et al. 2024; any local-judge substitute must be
   declared and calibrated). Requires OpenAI (or compatible) API
   wrapper, rate-limit handler, retry, and cost budget.
3. **Pierre compose/serve endpoint** that returns generations for
   N=5-composed adapters (math/code/medical + 2 additional to hit
   N=5) on Gemma 4 E4B base.
4. **Base Gemma 4 E4B generation endpoint** on the same sampling
   config (temperature, top_p, max_tokens) to make the pairwise
   comparison apples-to-apples.
5. **Pairwise win-rate + bootstrap-CI infrastructure** (500 games ×
   bootstrap ≥ 1000 resamples; style-controlled variant optional;
   Arena-Hard CI formula per Li et al. §4.3).

Block fires if shortfall ≥ 3 of 5. Pre-analysis by code grep under
`pierre/`, `macro/`, `composer/`, `micro/models/`:
- (1) Arena-Hard prompt set: absent. Only `.ralph/current_direction.md`
  mentions arena-hard as planning state; zero code references.
- (2) LLM-judge client with pairwise-compare schema: absent.
- (3) Pierre N=5 compose/serve endpoint: absent (macro serving stack
  is KILLED — see exp_prod_openai_api_compat, exp_prod_mlxlm_integration).
- (4) Base Gemma 4 E4B matched-config eval endpoint: absent as a
  pairwise-comparison peer.
- (5) Pairwise win-rate + bootstrap CI code: absent.

Shortfall ≥ 4/5. **T1 blocks.**

### T2 — Cost-bound block
Arena-Hard canonical protocol:
- 500 prompts per side (Pierre + base Gemma 4 E4B) = 1000 generations
- Per-sample generate ≈ 15 s (Arena-Hard prompts are open-ended;
  typical completions 400-800 tokens on M5 Pro bf16)
- Judge calls: 500 pairwise (GPT-4-Turbo; ≈ 5 s/call incl. network)
- Model cold-load ≈ 15 min each × 2 (base + Pierre-composed)
- Pierre N=5 compose overhead ≈ 5 min

Conservative total:
  `1000 * 15 + 500 * 5 + 2 * 15*60 + 5*60 = 15,000 + 2,500 + 1,800 + 300
  = 19,600 s ≈ 326.7 min`
vs **120 min ceiling**. Block fires.

Even an extreme smoke variant (50 prompts instead of 500):
  `100 * 15 + 50 * 5 + 1800 + 300 = 3,850 s ≈ 64.2 min` under ceiling,
but then CI formula degenerates (n=50 bootstrap on win-rate has
±6-8 pp half-width → K1701 CI test untestable). Smoke is
scientifically incoherent with this KC.

**T2 blocks.**

### T3 — Schema-incomplete block
DB record (verbatim from `experiment get`):
  `Success Criteria: NONE — add with: experiment success-add …`
  `⚠ INCOMPLETE: success_criteria, references`

F#502/F#646 antipattern: **10th occurrence** in this drain (iter 41
was 9th). Stable, earned heuristic. **T3 blocks.**

### T4 — Audit-pin reinforcer
Macro experiment with no prior runner, no DB diff in last 72 h, no
`.audit` directory. Pin-ratio measured post-run; reinforce-only.
**T4 reinforces (does not block alone).**

### T5 — Source-scope breach block
Parent (`depends_on`) experiment `exp_p1_t2_single_domain_training` has
current `verdict=supported` (cascade-ratified 2026-04-19 at
`LORA_SCALE=5`). Source scope:
- **3 single-adapter domains**: math (GSM8K), code (HumanEval),
  medical (MedMCQA) — **N=1** training, one adapter per domain
- **Closed-form academic benchmarks** (single-answer, exact-match
  or multiple-choice scoring)
- **No LLM judge**, no pairwise comparison, no open-ended scoring
- **No N=5 composition** (source never composes ≥ 2 adapters)
- **No Arena-Hard prompts**, no conversational/open-ended eval

Target scope:
- **Arena-Hard-Auto** (500 open-ended conversational prompts)
- **LLM-judged** pairwise win-rate (not exact-match)
- **N=5 adapter composition** on Gemma 4 E4B base
- **Bootstrap CI** over 500 games for K1701

Source-scope breach count (pre-reg ≥ 3 required):
  (A) Arena-Hard prompt class breach — source has 0 Arena-Hard
      coverage; only closed-form academic benches.
  (B) LLM-judge pairwise evaluation breach — source has 0 LLM-judge
      infrastructure; uses exact-match scoring.
  (C) Open-ended generation breach — source evaluates on short
      answers (GSM8K number, HumanEval pass/fail, MedMCQA MCQ);
      Arena-Hard requires 400-800 token open-ended completions
      with style/format variation.
  (D) N=5 composition breach — source trains N=1 single-domain
      adapters only; no composition evidence.
  (E) Bootstrap-CI / win-rate statistical framework breach —
      source reports point estimates; no pairwise win-rate CI.

Count = **5/5 breaches**. **T5 blocks** with wide margin.

**Theorem conclusion.** Verdict is **4-of-5 independent blocks** (T1 ∧ T2 ∧
T3 ∧ T5) plus **1 reinforcing** (T4). Any single block suffices. Target is
unrunnable on `local-apple` / MLX / 48 GB M5 Pro within a 120 min budget
without operator action (Arena-Hard prompt set + LLM-judge API + N=5
serve endpoint + pairwise-CI framework).

## 3. Predictions (pre-registered)

| ID | Prediction | Measurement |
|----|------------|-------------|
| P1 | T1 shortfall ≥ 3 of 5 required artifacts | code grep under pierre/, macro/, composer/, micro/models/ |
| P2 | T2 timing ≥ 120 min (even at conservative per-sample budget) | arithmetic on 500-prompt Arena-Hard protocol |
| P3 | T3 DB has `success_criteria: []` + `⚠ INCOMPLETE` marker | DB probe via `experiment get` |
| P4 | T4 pin_ratio in `.audit/` = 0 (dir absent); reinforce-only | `.audit` listing |
| P5 | T5 source-scope breach count ≥ 3 (of 5 dimensions) vs parent SUPPORTED `exp_p1_t2_single_domain_training` | source `results.json` + `PAPER.md` + `MATH.md` scope read |

## 4. Assumptions / caveats (A-series)
- **A1.** "Present in repo" = grep-reachable in `*.py` under `pierre/`,
  `macro/`, `composer/`, `micro/models/` (excluding this runner).
  Excludes markdown planning docs (so `.ralph/current_direction.md`
  does not satisfy T1).
- **A2.** LLM-judge probe requires literal `arena[_-]?hard` AND one of
  {`judge`, `pairwise`, `win[_-]?rate`, `gpt-4`, `judger`} in a file
  under the grep scope.
- **A3.** Pairwise-CI probe requires literal `bootstrap` AND one of
  {`win_rate`, `pairwise`, `arena`} within the same file.
- **A4.** N=5 stack probe reuses the iter-41 (mistral_nemo) pattern
  `N\s*=\s*5.*adapter|compose.*5.*adapter|adapter[_-]?stack.*5`.
  Known grep-noise: `leave_one_out_expert_ranking` uses N=50; this
  is filtered with a negative-lookbehind-style match.
- **A5.** T2 uses **conservative** 15 s/sample for Arena-Hard (prompts
  are longer-form than MMLU-Pro; 400-800 output tokens). Not
  sensitive: even at 5 s/sample the lower-bound is 5*1000 + 2500 +
  1800 + 300 = 9,600 s ≈ 160 min, still over 120.
- **A6.** T5 source-scope read uses `exp_p1_t2_single_domain_training`
  (`depends_on` declared). Source verdict is `supported` per live DB
  read (standard T5 applies; T5-K would apply only if parent flips
  to KILLED, which it has not).
- **A7.** Runner is pure stdlib + `experiment get` shell-out. Zero
  MLX, zero model load, zero HTTP bind. ≤ 3 s wall.
- **A8.** F#502 10th-occurrence claim is cumulative drain count;
  runner reports the per-file `⚠ INCOMPLETE` literal from the DB,
  not a running counter. Counter is in LEARNINGS/scratchpad prose.
- **A9.** Grep may produce false-positive hits for `arena_hard` or
  `judge` inside docstrings or planning markdown. Runner restricts
  to `*.py` under the scope dirs; if T1 still reports shortfall < 3,
  the MATH prediction holds only if T2 ∨ T3 ∨ T5 block. In practice
  T2 ∧ T3 ∧ T5 alone over-determine the verdict.
