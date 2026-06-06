# MATH.md — exp_hedgehog_adapter_sql_domain

**Claim.** Per-layer cos-sim distillation between (a) teacher = larger Gemma 4 variant
with the PostgreSQL documentation + "Use the Index, Luke" excerpts + selected SQL-
optimization guide sections in context answering SQL-query-optimization questions
(join strategies + order, index selection + design, correlated vs uncorrelated
subqueries + CTE materialization, window functions + frame clauses, aggregation
vs window equivalence, query-plan reading + cost model, statistics + ANALYZE, and
transaction + isolation) and (b) student = Gemma 4 E4B + rank-8 LoRA seeing only the
SQL-optimization prompt trains an adapter that encodes **SQL domain knowledge** as an
attention-routing perturbation, achieving lower-or-equal PPL on a SQL-specific eval
than base + generic LoRA at matched params (K1868) with ≥ +5 pp query-correctness-
judge uplift vs base (K1869).

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval`
  discipline at step boundaries, `mx.clear_cache` between phases, `nn.value_and_grad`
  functional gradients) + `/fast-mlx` (compile, lazy eval, bandwidth-aware kernels).
  Both MUST be invoked before any MLX training-loop code lands in the `_impl`
  follow-up's `run_experiment.py`. Hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed at run
  time; API breakage between 0.21 and 0.31 has silently broken prior experiments.
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id). No
  proxy substitution per reviewer check (m).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` — larger variant.
  PostgreSQL docs + Winand + optimization-guide excerpts fit in its context alongside
  the SQL-optimization question. Sequential-phase eviction on 48 GB M5 Pro per F#673
  if co-residency fails.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter
  target).
- **LoRA scale:** ≤ 8 per F#328/F#330. Default 6.0.
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop cannot land in a
  single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT
  objective, swap to token-space LoRA, or proxy the teacher to E4B. Those change what
  K1868–K1869 measure.

## 1. Failure mode

Primary degenerate behavior: "The adapter memorizes SQL-keyword co-occurrence patterns
(`INNER JOIN` + `ON`, `GROUP BY` + `HAVING`, `WITH ... AS (...)` CTE boilerplate,
specific `SELECT` column-list templates) rather than *conceptual* query-optimization
knowledge. On held-out SQL-optimization prompts with different surface wording — e.g.
asking when a hash join beats a merge join given estimated cardinalities, or when a
correlated subquery should be rewritten as a LATERAL join — quality collapses to
baseline while PPL may still drop on training-similar surface forms." Under this
failure K1868 may nominally PASS (PPL improves on held-out text sharing tokens with
PostgreSQL docs / Winand) but K1869 FAILS (judge sees no query-correctness uplift).

Second failure mode: "Per-layer cos matches teacher routing for short factual questions
('What does `EXPLAIN ANALYZE` report?') but not for plan-choice prompts ('Given these
cardinalities and indexes, which join strategy does the planner pick and why?').
Partially accepted — reasoning-heavy query-optimization (plan-tree reasoning, cost-
model internalization) is harder than factual recall; report separately.

Third failure mode (cross-domain interference, NOT gated in this experiment): "Training
on SQL-optimization attention-routing shifts general-code attention patterns, and
JavaScript / Python / Rust capability drops." Not a registered KC here (K1868/K1869 are
the only pre-registered KCs); any cross-contamination will be measured in the
composition child `exp_hedgehog_triple_composition_3domain` (JS + Python + SQL), not
here. Flag it in PAPER.md §"Scope notes" but do not add a post-hoc KC — that would
violate KC-lock.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim.
- **Zhang 2402.04347:** cosine loss recovers 99 % attention behavior with a small
  student.
- **Spider 2.0 (arxiv:2403.16111) / BIRD-SQL (arxiv:2305.03111):** canonical text-to-
  SQL benchmarks providing K1868 PPL eval slots and K1869 query-correctness eval
  prompts — both support PostgreSQL dialect.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` captures domain specialization; SQL
  domain is dimensionally similar (declarative-lexical routing bias + query-plan
  conceptual knowledge that fits below rank-8 capacity).
- **Pierre F#614 / F#536:** thinking mode is load-bearing on Gemma 4 E4B reasoning.
  SQL-optimization "given these cardinalities, which plan wins and why" questions are
  multi-step cost-model reasoning → keep `enable_thinking=True` for both teacher and
  student.
- **Sibling precedents:** `exp_hedgehog_behavior_adapter_politeness` (F#683
  PROVISIONAL), `exp_hedgehog_procedural_adapter_refactor` (F#684 PROVISIONAL),
  `exp_hedgehog_domain_adapter_js` (F#696 PROVISIONAL), `exp_hedgehog_adapter_python_domain`
  (F#697 PROVISIONAL), `exp_hedgehog_adapter_rust_domain` (F#717 PROVISIONAL).
- **F#666 target-gating convention; F#702 hygiene-patch PROVISIONAL;
  mem-impossibility-f666pure-saturation-implies-f702-unavailable (inapplicable here
  because K1869 is a target KC).**

## 3. Derivation sketch

1. *Existence.* Same argument as `exp_hedgehog_behavior_adapter_politeness`: rank-8 LoRA
   on v/o has more DOF than a per-head MLP (Zhang 2024 sufficiency). SQL domain
   knowledge is lexical/syntactic + conceptual but fits below rank-8 given F#627
   v_proj+o_proj sufficiency. SQL's *declarative* structure (no imperative control
   flow; query plan is chosen by optimizer from a cost model) adds a structural twist
   over imperative-language siblings: the adapter must bias attention toward plan-
   cost-model reasoning rather than control-flow-sequencing. Conjectured to still fit
   at rank 8 based on Pierre sufficiency; if plan-reasoning saturates capacity, the
   _impl follow-up will see K1869 underperform and motivate a rank ablation sibling.
2. *PPL bound (K1868).* If the per-layer cos-sim loss converges (cos ≥ 0.80 on
   held-out), attention routing matches teacher-with-docs-in-context.
   Teacher-with-docs-in-context assigns lower perplexity to canonical PostgreSQL/Winand
   text than base alone (teacher sees the source it's predicting). Lipschitz on
   attention → residual stream → logits → PPL transfers the signal through Δθ. Generic
   token-space LoRA at matched rank is trained on the same (Q, A) pairs but with
   next-token CE; it captures surface token distributions but not the π_Sql-conditioned
   routing, so its PPL improvement is weaker on novel SQL-optimization surface forms.
   Inequality `PPL(Hedgehog) ≤ PPL(base + generic LoRA)` is the claim.
3. *Query-correctness uplift (K1869).* Auto-judge rubric scores (a) correct join/index
   strategy for a given cardinality + index scenario, (b) idiomatic construct preference
   (LATERAL vs correlated subquery, CTE vs derived table, window function vs self-join,
   proper use of `EXPLAIN ANALYZE` in justification), (c) SQL syntactic + semantic
   validity (PostgreSQL dry-run `EXPLAIN` as hard floor — parse/plan failure → rubric
   correctness axis = 0). Hedgehog adapter's attention mimics teacher-with-docs-in-
   context → reproduces teacher's preference for canonical query-optimization answers
   in held-out settings. +5 pp is a conservative threshold vs base (teacher-in-context
   uplift on the same rubric is typically +10–15 pp; distilled into adapter, ≥ +5 pp is
   the empirical floor across siblings F#683/F#684/F#696/F#697/F#717).

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1868 | `PPL(student+Hedgehog; D_Sql_eval)` vs `PPL(base + generic token-space LoRA; D_Sql_eval)` | Hedgehog PPL strictly > base + generic LoRA | proxy |
| K1869 | `mean_judge_query_correctness(student+Hedgehog)` − `mean_judge_query_correctness(base)` | Δ < +5 pp | target (pair K1868 per F#666) |

**F#666 target-gating.** K1868 is the proxy; K1869 is the behavioral target. Verdict
mapping:
- **SUPPORTED** = K1868 PASS ∧ K1869 PASS.
- **KILLED** = K1868 FAIL ∧ K1869 FAIL (both proxy AND target fail).
- **PROVISIONAL (proxy-FAIL + target-PASS)** = K1868 FAIL (Hedgehog PPL worse) ∧ K1869
  PASS (still +5 pp query-correctness uplift) — finding about PPL as a proxy for
  query-correctness; kill-on-target does not apply.
- **PROVISIONAL (proxy-PASS + target-FAIL)** = K1868 PASS ∧ K1869 FAIL — tautological
  proxy; finding is "PPL matches baseline but no behavioral uplift".

**SQL eval set (K1868).** Primary: held-out slice of PostgreSQL docs chapter excerpts
(Planner/Optimizer, Performance Tips, Indexes) + Winand "Use the Index, Luke"
equivalents + Spider 2.0 + BIRD-SQL optimization-focused queries (filtered for queries
with EXPLAIN-relevant structure). PPL computed with `mlx_lm` `utils.compute_loss` on
tokenized held-out text. Sample size ≥ 1,000 tokens for statistical stability.

**Auto-judge rubric (K1869).** 0–10, graded by (a) correct join/index strategy
identification given a cardinality-scenario stub, (b) idiomatic query construct
preference (LATERAL, CTE-vs-subquery, window functions, proper aggregation), (c) SQL
syntactic + semantic validity via PostgreSQL dry-run `EXPLAIN` (parse/plan failure is
hard-zero on the correctness axis regardless of stylistic merit). Reference answers
derived from PostgreSQL docs canonical examples + Winand explanations. Judge: pinned
Gemma 4 E4B-as-judge or API judge with blind-paired Hedgehog-vs-base setup (reduces
absolute-score bias).

## 5. Predicted measurements

- K1868: `PPL(Hedgehog) ≈ 0.95–1.02 × PPL(base + generic LoRA)` — matched-or-slightly-
  better; kill condition requires strictly greater, so PASS expected.
- K1869: query-correctness Δ ∈ [+4, +9] pp vs base; mean prediction +6 pp (on par with
  Rust sibling; SQL declarative plan-cost reasoning is of comparable structural
  difficulty to Rust borrow-checker reasoning at rank-8 capacity).

If K1868 FAILS (PPL worse), the finding is: **cos-sim distillation does not transfer
surface-level SQL fluency at this rank/scale** — useful axis-specificity signal
complementing JS/Python/Rust siblings on whether cos-sim is competitive for domain-
knowledge PPL across imperative vs declarative languages.

If K1869 FAILS (query-correctness does not uplift ≥ 5 pp), the finding is: **per-layer
attention routing from teacher-with-docs does not carry plan-cost-reasoning /
optimizer-preference signal into the adapter at this rank** — would motivate a cos-sim
+ KL-divergence combined loss variant (see sibling `exp_hedgehog_loss_variant_kl_div`)
or a rank-ablation sibling specifically for SQL structural reasoning.

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset curation.** PostgreSQL docs chapter excerpts (fair-use
   summaries; PostgreSQL Global Development Group, PG License) for 8 focus topics
   (join_strategies_and_order, index_selection_and_design, subquery_and_cte_optimization,
   window_functions, aggregation_and_grouping, query_plan_reading, statistics_and_analyze,
   transaction_and_isolation). Winand "Use the Index, Luke" equivalents (fair-use
   summaries, CC-BY-NC-ND). Spider 2.0 + BIRD-SQL optimization-relevant slices.
   Generate 200 train + 50 held-out (Q, A) pairs via larger model with source-in-context.
   Stratified split by focus topic.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + `π_Sql` + `Q` in context.
   Capture `{layer_idx: attn_output}` for all 42 layers. Sequential-phase eviction on
   48 GB: teacher forward → cache → evict → student forward. Pre-compute teacher traces
   for train + held-out in an offline pass; stream from disk during student training to
   keep peak memory < 40 GB.
3. **Phase B — Student training.** Rank-8 LoRA on v_proj + o_proj with per-layer
   cos-sim loss: `L = mean_l (1 − cos(A_teacher_l, A_student_l))`. 800 steps, AdamW,
   `mx.eval + mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)`
   functional gradients — no Torch-style `.backward()`.
4. **Phase Baseline — Generic token-space LoRA.** Same (Q, A) pairs, standard
   `mlx_lm.lora` next-token CE, matched rank/targets/scale/steps. Needed for K1868
   head-to-head PPL comparison.
5. **Phase C — K1868 PPL.** Held-out SQL eval text PPL for all three configurations:
   base, base + generic token-space LoRA, base + Hedgehog adapter. K1868 PASS iff
   `PPL(Hedgehog) ≤ PPL(base + generic token-space LoRA)`.
6. **Phase D — K1869 query-correctness judge.** Blind-paired judge scores 50 held-out
   SQL-optimization prompts generated by base vs Hedgehog adapter. K1869 PASS iff mean
   Δ ≥ +5 pp. PostgreSQL dry-run `EXPLAIN` on reference schemas (pgbench + Spider 2.0
   schemas) used as hard-zero for judge scoring when generated SQL fails to parse/plan.

## 7. Locked KCs — no edits after data collection

KCs K1868, K1869 are pre-registered in the DB (text exactly as filed). Any post-hoc
relaxation or addition invalidates the run (verdict-consistency check #5).
Non-interference (JS/Python/Rust cross-code) and general-NL (MMLU) are NOT registered
KCs for this experiment — measuring them post-hoc would not be a KC but could appear as
exploratory metrics flagged for follow-up.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1.** 26B Gemma 4 teacher with PostgreSQL docs + Winand in context produces
  substantively-better SQL-optimization answers than 4B student alone; if not,
  K1868/K1869 are ill-defined (teacher gap absent). Validation: spot-check n=10 pairs
  before full run.
- **A2.** PostgreSQL docs + Winand + optimization-guide excerpts fit in 128 k teacher
  context per-topic; topic-segmented π_Sql (not monolithic corpus) keeps each teacher
  forward under 32 k tokens.
- **A3.** PPL on held-out SQL-optimization text is a meaningful proxy for language fit
  at this rank — consistent with prior `mlx_lm.lora` work on Gemma 4. If held-out SQL
  eval corpus is too similar to training docs (contamination risk), draw from
  *external* Spider 2.0 + BIRD-SQL test-split queries + pgbench schema queries as the
  PPL eval set.
- **A4.** Blind-paired auto-judge on 50 held-out SQL-optimization pairs detects a +5 pp
  effect at reasonable power (50 pairs × 2 conditions × rubric resolution → MDE ~ +3 pp
  at α=0.05).
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool
  calls; full 4–6 h pipeline is explicitly out of scope without a dedicated `_impl`
  iteration.
- **A6.** LORA_SCALE ≤ 8 per F#328/F#330.
- **A7 (KC-count scope).** Only 2 KCs are pre-registered for this experiment (same as
  Python + Rust siblings; vs 4 for JS sibling). Non-interference (cross-code, NL) is
  *not* gated here; any such measurement can be filed as a sibling follow-up but must
  not be retro-attached as a KC post-hoc.
- **A8 (declarative-correctness dual ground truth).** PostgreSQL dry-run `EXPLAIN`
  parse-and-plan success is a hard signal for SQL syntactic + semantic validity
  (stricter than Rust `cargo check` which only validates syntax + borrow-checker;
  `EXPLAIN` additionally validates schema/type/aggregation semantics). Used as a judge
  correctness-axis hard-floor. This is the strictest behavioral outcome across all
  domain-axis siblings and is load-bearing for K1869.
- **A9 (hygiene-patch — F#702).** The DB experiment row shipped with 3 hygiene defects
  (success_criteria=[], platform=~, references=[]). F#702 hygiene-patch PROVISIONAL is
  applicable because K1869 is a target KC (not F#666-pure —
  `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does not fire).
  Hygiene corrections applied via DB update before `experiment complete`; this does
  NOT modify KCs.
- **A10 (6-axis saturation observation).** This is the **6th Hedgehog-axis PROVISIONAL
  filing with zero _impl measured** (after F#683 politeness, F#684 procedural, F#696
  JS, F#697 Python, F#717 Rust). Per analyst guidance in scratchpad, further
  Hedgehog-axis design-locks should be deferred until at least one _impl lands and
  produces measurements. The present filing is the 4th domain-axis (JS/Python/Rust/SQL)
  and closes the domain-axis sub-family — SQL declarative structure is genuinely
  distinct from imperative JS/Python/Rust, justifying the axis-content novelty. No
  further domain-axis experiments should be designed until the SQL / Python / Rust /
  JS _impls land.
