# MATH.md — exp_knowledge_disentanglement_control

Experiment ID: `exp_knowledge_disentanglement_control`
Platform: Apple M5 Pro 48 GB, `mlx-community/gemma-4-e4b-it-4bit`, `mlx-lm==0.31.2`.
Base: `π_0` = Gemma-4-E4B-it-4bit.
Adapter: rank-16 LoRA on `{self_attn.v_proj, self_attn.o_proj}`, `LORA_SCALE=4.0`, top 16 layers.

## Claim (Theorem — Method/Knowledge Disentanglement)

Let `π_0` be the frozen base and `π_θ^M = π_0 + Δ_r(θ)` the same base
after SFT on `(q, r(q))` pairs where `r(q)` is a teacher trace produced
by `π_0` under an explicit *method* (subgoal-decomposition) system
prompt over a domain-diverse set `D_M = ⋃_{d∈D_train} D_d`, trained with
prompt erasure (Orca 2, Mitra et al. `arxiv:2311.11045`).

**Theorem (informal).** Under the assumptions (A1)–(A3) below, the
learned low-rank delta `Δ_r(θ)` lives approximately in a subspace
`S_method ⊂ ℝ^{d×d}` that encodes *procedural* behaviour (how to
decompose a problem) and is approximately orthogonal to the
*knowledge* subspace `S_knowledge` that encodes factual recall. In
particular:

1. **K1733 (reasoning gain).** On a reasoning-heavy benchmark
   `B_R = BBH_proxy` (reasoning proxy — see §Proxy), accuracy
   `acc(π_θ^M, B_R) − acc(π_0, B_R) ≥ +5 pp`.
2. **K1734 (knowledge neutrality).** On a general-knowledge benchmark
   `B_K = MMLU-Pro (balanced held-out cats)`,
   `|acc(π_θ^M, B_K) − acc(π_0, B_K)| < 1 pp`.
3. **K1735 (factual neutrality).** On a pure factual-recall benchmark
   `B_F = TriviaQA validation subset`,
   `|acc(π_θ^M, B_F) − acc(π_0, B_F)| < 1 pp`.
4. **K1736 (robustness).** The three KCs above hold for **each of 3
   independent training seeds** with inter-seed coefficient of
   variation < 10 %.

If all four hold the claim is supported; any single failure kills the
claim.

## Mechanism (why it should work)

1. **Task arithmetic (Ilharco et al., `arxiv:2212.04089`).** Fine-tuning
   deltas in weight space are approximately additive for single-task
   training. Their result is full-rank; we extend it to LoRA rank 16
   by hypothesising the method direction is ≤ 16-dim.
2. **Function vectors (Todd et al., `arxiv:2310.15213`).** High-level
   behavioural *functions* (like "translate", "classify", "decompose")
   are encoded as low-dimensional vectors in activation space that
   generalise across examples. Our hypothesis lifts this to **weight
   space**: the low-rank LoRA parameters can store the function
   vector of the method itself.
3. **Orca 2 explanation-tuning (Mitra et al., `arxiv:2311.11045`).**
   Training on teacher rationale traces with **prompt erasure** induces
   the student to emit the reasoning style without the system prompt.
   Orca 2 used full-rank tuning; we test the rank-16 analogue.
4. **Gradient decomposition under domain diversity.** For a LoRA delta
   `Δ = BA` with `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`, the per-example
   gradient decomposes (sketch) into a method direction `e_M` plus
   a domain-content direction `e_d`:
   `∂L/∂Δ ≈ e_M + e_d(q)`.
   Balanced sampling across `|D_train| = 5` categories makes the
   domain-content term contract as `(1/5)·Σ_d e_d`, whose norm is
   bounded by the Welch-bound cosine `cos_max ≤ √((n−r)/(r(n−1)))
   ≈ 0.45` for `n=5`, `r=1` — so the method term dominates.
5. **Why knowledge is untouched.** Knowledge lives primarily in MLP
   layers and `q_proj`/`k_proj` in attention (see Meng et al., ROME,
   `arxiv:2202.05262`, which localised factual recall in mid-layer
   MLPs). We only adapt `v_proj+o_proj` — attention's output
   projection — which controls **how** attention routes information
   rather than **what** facts are retrieved. If the Meng localisation
   is real, then rank-16 on `{v_proj, o_proj}` cannot rewrite
   factual content; therefore `Δ acc on knowledge tasks ≈ 0`.

## Assumptions (logged per guardrail 1007)

- **(A1) Method-subspace low-rank.** The procedural "decompose into
  subgoals" skill has intrinsic dimensionality ≤ 16 (the LoRA rank).
- **(A2) Knowledge-skill localisation.** Factual recall is localised
  in layers/projections not adapted here (primarily MLPs and
  `q_proj/k_proj`; ROME evidence).
- **(A3) Prompt-erasure works at rank 16.** Training on student-side
  `(q, r(q))` without the method system prompt is sufficient for the
  adapter to emit the method at inference time without the prompt
  (Orca-2 transfer at rank 16).
- **(A4) Proxy equivalence.** Since BBH (`lukaemon/bbh`) is not
  cached locally and this iteration is network-isolated to remote
  DBs only, I substitute **GSM8K (gsm8k-test)** as the reasoning
  proxy. GSM8K measures multi-step numerical reasoning where
  subgoal decomposition directly helps; a method adapter that
  cannot lift GSM8K is unlikely to lift BBH. **This is a
  stipulated proxy limitation**: if the smoke produces a
  supported verdict, full-scale rerun MUST swap GSM8K → BBH.
- **(A5) Budget parity and KC-lock.** Kill criteria K1733–K1736
  are frozen at commit-time; no adjustment after results.
- **(A6) Smoke discipline.** The smoke run reports
  `is_smoke=true`, verdict `PROVISIONAL`, and completes with
  `--status provisional`. Upgrades to `supported` require a full
  rerun with 3 seeds and full benchmark sizes.

## Kill criteria (pre-registered, DB IDs 1733–1736)

| KC    | Threshold                                                           | Source          |
| ----- | ------------------------------------------------------------------- | --------------- |
| K1733 | `acc(adapter, R) − acc(base, R) ≥ +5 pp` on reasoning proxy         | DB id 1733      |
| K1734 | `|acc(adapter, MMLU) − acc(base, MMLU)| < 1 pp`                     | DB id 1734      |
| K1735 | `|acc(adapter, TriviaQA) − acc(base, TriviaQA)| < 1 pp`             | DB id 1735      |
| K1736 | K1733 ∧ K1734 ∧ K1735 hold on **each** of 3 seeds, CV < 10 %        | DB id 1736      |

At smoke scale (1 seed, reduced n), K1736 is marked **inconclusive**
by construction. Smoke verdict is `PROVISIONAL` regardless of K1733–
K1735 outcomes.

## Predictions (with quantitative numbers)

- **Reasoning (GSM8K, n=30 smoke).** Base Gemma-4-E4B-it-4bit with
  thinking: literature reports ≈ 70–80 % on GSM8K; we expect ≈ 70 %
  at n=30. Adapter: expect +5–+15 pp (75–85 %) if the method subspace
  hypothesis is correct; expect 0 ± 5 pp if the adapter just adds
  noise; expect − if the adapter damages reasoning.
- **Knowledge (MMLU subset, n=30 smoke).** Base ≈ 60–65 %; adapter
  within ±1 pp if (A2) holds; outside ±1 pp if the adapter leaks into
  knowledge (failure mode).
- **Factual (TriviaQA, n=30 smoke).** Base ≈ 40–55 % (thinking
  models often lose on factual recall); adapter within ±1 pp if (A2)
  holds.

At smoke n=30 per benchmark, statistical noise is ~±9 pp. A
PASS on K1734/K1735 at smoke scale therefore *includes* the
possibility of a 1-question swing; the informative signal is in
K1733 (+5 pp ≫ 1/30 ≈ 3 pp quantum of a single answer) and in
whether the adapter preserves thinking-mode output.

## Antipattern scan (pre-flight)

- **composition-bug:** N/A (single adapter, no composition).
- **tautological-routing:** N/A (single adapter, no routing).
- **lora-scale (unsafe ≥ 8):** scale=4.0. ✓
- **thinking-truncation:** eval `max_tokens=2048`; teacher
  `max_tokens=1024` — both sufficient for Gemma-4 thinking traces.
  `strip_thinking` is fortified per predecessor's Issue 2 (strips
  `<|channel>thought` even when the `<channel|>` close tag is
  missing, up to next blank line or `Answer:`/final answer line).
- **hardcoded-pass:** KC booleans derived from measurements.
- **proxy model:** Same `mlx-community/gemma-4-e4b-it-4bit` weights
  in all arms; adapter attached via `load(..., adapter_path=...)`.
- **smoke-as-full:** `is_smoke=true` explicitly flagged when
  `SMOKE_TEST=1` (default); full rerun requires `SMOKE_TEST=0`.
- **KC swap after data:** frozen at commit. Any downstream need to
  relax ⇒ v2 experiment.
- **copy-paste scaffolding silently propagating bugs:** we *copy*
  `parse_mcq_answer` and `count_subgoal_markers` helpers from
  `exp_method_vs_domain_adapter/run_experiment.py` verbatim, with
  the **v2 fortification of `strip_thinking`** applied here (missing
  close tag case). Noted so reviewer can diff.

## Kill-criteria lock

Kill criteria above are the DB pre-registered IDs. No modification
is permitted after the commit of this file; v2 experiment required
if changes are needed.
