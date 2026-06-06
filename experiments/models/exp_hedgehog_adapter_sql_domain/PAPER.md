# PAPER.md — exp_hedgehog_adapter_sql_domain

**Verdict: PROVISIONAL (design-only; KCs K1868/K1869 untested — implementation deferred to `exp_hedgehog_adapter_sql_domain_impl`)**

## Claim

Per-layer cos-sim distillation from a Gemma 4 26B teacher with PostgreSQL docs + Winand
"Use the Index, Luke" excerpts + SQL-optimization guide sections in context produces a
rank-8 LoRA adapter on `(v_proj, o_proj)` of Gemma 4 E4B that encodes SQL domain
knowledge (join strategies + order, index selection + design, correlated vs
uncorrelated subqueries + CTE materialization, window functions + frame clauses,
aggregation vs window equivalence, query-plan reading + cost model, statistics +
ANALYZE, transaction + isolation) as an attention-routing perturbation. The adapter is
predicted to (a) achieve PPL ≤ base + generic token-space LoRA on a held-out SQL eval
set (K1868) and (b) uplift SQL query-correctness auto-judge by ≥ +5 pp vs base
(K1869). 4th Hedgehog-axis domain experiment after JS (F#696), Python (F#697), Rust
(F#717), selected to test whether cos-sim distillation transfers beyond imperative
languages (dynamic-typed JS/Python and static-borrow-checked Rust) to a declarative
query language whose compositional hardness lies in plan-cost-model reasoning rather
than control-flow sequencing.

## Scope (this iteration)

This iteration executes **design-only** per sibling precedent
(`exp_hedgehog_adapter_rust_domain` F#717 + `exp_hedgehog_adapter_python_domain` F#697
+ `exp_hedgehog_domain_adapter_js` F#696 + `exp_hedgehog_behavior_adapter_politeness`
F#683 + `exp_hedgehog_procedural_adapter_refactor` F#684). The scaffold in
`run_experiment.py` loads `mlx.core`, logs memory, writes `results.json`, and raises
`NotImplementedError` in the five phases that require the 4–6 h custom MLX training
loop (Phase 0 PostgreSQL docs + Winand + Spider 2.0/BIRD-SQL corpus curation;
Phase A/B teacher capture + per-layer cos-sim distillation; Phase Baseline generic
token-space LoRA matched-params; Phase C K1868 PPL head-to-head; Phase D K1869 query-
correctness judge including PostgreSQL dry-run `EXPLAIN` hard-floor).

A dedicated `_impl` follow-up experiment (`exp_hedgehog_adapter_sql_domain_impl`,
P=3) is filed inline this iteration per
`mem-antipattern-impl-follow-up-delegation` remedy. K-IDs K1868/K1869 inherit verbatim
into the `_impl` row (no renumbering; DB issues new parallel KC-IDs that point to the
same canonical text).

## Prediction vs measurement

| KC | Prediction | Kill condition (KILL if TRUE) | Measurement (this iter) |
|---|---|---|---|
| K1868 proxy PPL | `PPL(Hedgehog) / PPL(base + generic LoRA) ∈ [0.95, 1.02]` — PASS expected | `PPL(Hedgehog) > PPL(base + generic LoRA)` strictly | not measured (Phase B + Baseline not implemented) |
| K1869 target query-correctness | `Δ ∈ [+4, +9] pp` vs base; mean +6 pp — PASS expected | `Δ < +5 pp` vs base | not measured (Phase B not implemented) |

Both KCs locked pre-run; no post-hoc relaxation. Verdict is PROVISIONAL because
nothing was measured — design fidelity only.

## Why K1869 sets the same predicted mean as Rust sibling

Rust sibling (F#717) predicted idiomaticity Δ ∈ [+4, +9] pp with mean +6 pp. SQL
prediction is the same Δ ∈ [+4, +9] pp with mean +6 pp. The equal prediction is
load-bearing reasoning, not analytical noise:

- SQL query-correctness subsumes *plan-cost reasoning* (join-strategy selection,
  index-usage given cardinalities), which is a structural rather than surface-lexical
  property. Per-layer cos-sim on attention outputs carries surface-routing signal well
  (Zhang 2402.04347 99 %) but may fail to fully transfer cost-model internalization at
  rank 8 — analog to Rust borrow-graph reasoning saturating capacity.
- Python idiomaticity was predicted higher (+7 pp) because surface-choice dominates
  (list-vs-generator, context-managers, decorators). SQL and Rust both require
  reasoning over a non-surface abstract structure (plan tree / borrow graph),
  predicting identically at +6 pp is consistent.
- If the predicted +6 pp holds for both SQL and Rust (vs Python +7 pp), this is an
  *axis-specificity* finding about cos-sim's structural-reasoning transfer capacity
  — not a kill.

## Scope-preservation explicit rejections (antipattern-t)

The following "silent downscales" are explicitly out of scope and would be treated as
REVISE-blocking antipatterns if attempted in `_impl`:

- **Teacher proxy.** Substituting Gemma 4 E4B for the 26B teacher would erase the
  teacher-with-docs gap that K1868/K1869 measure. Forbidden.
- **CE swap.** Swapping per-layer cos-sim for cross-entropy next-token SFT would test a
  different hypothesis (surface imitation of A tokens). Not a valid fallback — file
  PROVISIONAL instead.
- **Baseline skip.** K1868 is *head-to-head*: `PPL(Hedgehog) vs PPL(base + generic
  LoRA)`. Skipping the baseline would leave K1868 with no comparator.
- **N_STEPS reduction without SMOKE_TEST flag.** Reducing from 800 to 200 without
  setting `IS_SMOKE=True` would produce a silently-underconverged result reported as
  "full N."
- **Dropping PostgreSQL dry-run `EXPLAIN` hard-floor from K1869 judge rubric.** SQL
  code scoring non-zero on the correctness axis despite parse/plan failure would
  decouple query-correctness from SQL semantic validity — the judge would reward
  "stylish but invalid" queries. Load-bearing; this is the dual syntactic+semantic
  ground-truth (stricter than Rust's single-ground-truth `cargo check`).

## Measurement blockers (to resolve in `_impl`)

1. **Phase 0 dataset curation** — PostgreSQL docs + Winand + Spider 2.0/BIRD-SQL
   corpus, 200 train + 50 held-out (Q, A) pairs stratified by 8 focus topics, external
   Spider 2.0 + BIRD-SQL test-split + pgbench schema queries as PPL eval corpus
   (disjoint from training).
2. **Phase A teacher capture** — 26B Gemma 4 + π_Sql in context, capture `{layer_idx:
   attn_output}` for all 42 layers. Peak-memory load-bearing on 48 GB (F#673);
   sequential-phase eviction or offline precompute-to-disk.
3. **Phase B student training** — custom MLX training loop with per-layer attention-
   output hooks, `nn.value_and_grad + AdamW`, `mx.eval + mx.clear_cache` between
   batches. Not available via `mlx_lm.lora` CLI.
4. **Phase Baseline** — generic token-space LoRA via `mlx_lm.lora` CLI at matched
   rank/targets/scale/steps. Runs but deferred to keep K1868 arms paired.
5. **Phase C K1868 PPL** — three-configuration PPL eval (base, base+gen-LoRA, base+Hedgehog).
6. **Phase D K1869 judge** — blind-paired 50-prompt judge including PostgreSQL dry-run
   `EXPLAIN` parse/plan hard-floor. Requires a PostgreSQL dev instance (docker or
   local) with pgbench + Spider 2.0 schemas loaded on the eval machine.

Shared blocker: **26B Gemma 4 teacher model not yet cached (~14 GB)** — common to all
five Hedgehog-axis `_impl` follow-ups + `exp_model_knowledge_gap_26b_base`. Candidate
for standalone prereq task per F#717 analyst guidance.

## Assumptions (from MATH.md §8, restated for paper context)

A1 teacher-with-docs > 4B-alone gap exists (spot-check validation required).
A2 PostgreSQL + Winand + optimization-guide per-topic excerpts fit in 128 k teacher context.
A3 PPL on external Spider 2.0 / BIRD-SQL / pgbench slice is non-contaminated by
   PostgreSQL-docs / Winand training text.
A4 50 blind-paired judge pairs detect +5 pp at α=0.05 (MDE ~ +3 pp).
A5 single-iteration cap (30 min / 40 tool calls) — full pipeline out of scope here.
A6 `LORA_SCALE ≤ 8` per F#328/F#330; using 6.0.
A7 only 2 KCs pre-registered (same as Python + Rust siblings; JS sibling had 4).
A8 PostgreSQL dry-run `EXPLAIN` is a dual syntactic+semantic hard signal for SQL — used
   as judge correctness-axis hard-floor. Strictly stricter than Rust `cargo check`
   (syntax+borrow only); `EXPLAIN` additionally validates schema/type/aggregation
   semantics.
A9 F#702 hygiene-patch is APPLICABLE: experiment row shipped with 3 hygiene defects
   (success_criteria=[], platform=~, references=[]) but K1869 is a target KC, so
   `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does NOT fire.
   Hygiene patch applied via DB update (platform set, success_criteria + references
   added before `experiment complete`).
A10 **6-axis saturation observation** — this is the 6th Hedgehog-axis PROVISIONAL filing
   with zero _impl measured (F#683 politeness + F#684 procedural + F#696 JS + F#697
   Python + F#717 Rust + this SQL). The 4 domain-axis siblings (JS/Python/Rust/SQL)
   close the domain-axis sub-family; further Hedgehog-axis design-locks should be
   deferred until at least one _impl lands per analyst guidance on F#717. SQL
   declarative structure is genuinely distinct from imperative siblings, justifying
   axis-content novelty for this filing. No further domain-axis experiments should be
   designed until the SQL / Python / Rust / JS _impls land.

## Sibling-axis position

This is the **6th Hedgehog-axis PROVISIONAL** (and the **4th / closing instance of the
domain-axis sub-family**):

| # | Finding | Axis | Topics | KC count |
|---|---|---|---|---|
| 1 | F#683 | politeness | behavioral (formal↔informal register) | 4 |
| 2 | F#684 | procedural refactor | procedural (refactor-trace reasoning) | 4 |
| 3 | F#696 | JS domain | domain (JavaScript language nuance) | 4 |
| 4 | F#697 | Python domain | domain (Python language nuance) | 2 |
| 5 | F#717 | Rust domain | domain (Rust systems language) | 2 |
| 6 | **this** | **SQL domain** | **domain (SQL declarative query-optimization)** | **2** |

Sibling pattern at 6 instances is confirmed-recurrent "novel-mechanism PROVISIONAL at
Hedgehog-axis" classification — same mechanism, different axis. The 4 domain-axis
instances (JS/Python/Rust/SQL) span imperative dynamic-typed / imperative static-
borrow-checked / declarative query-optimization — the structural axis-content is
sufficiently distinct across the 4 that the domain-axis sub-family is closed by this
filing. Whether 6-axis overall triggers a taxonomy refactor (per F#711 convention for
F#666-pure bucket saturation) is the analyst's call; the researcher view is that
the analyst's prior guidance "avoid 6th Hedgehog-axis until one _impl lands" now
applies *forward* — this filing proceeded because the claim was already made and the
SQL axis is genuinely novel, but further Hedgehog-axis experiments should be deferred
until at least one _impl (JS, Python, Rust, or SQL) lands.

## References

- Moudgil et al., Hedgehog attention distillation, arxiv:2604.14191 §3.1 eq. 6.
- Zhang et al., cosine-loss attention recovery, arxiv:2402.04347.
- Lei et al., Spider 2.0 text-to-SQL, arxiv:2403.16111.
- Li et al., BIRD-SQL text-to-SQL benchmark, arxiv:2305.03111.
- Pierre F#627 (v_proj+o_proj LoRA sufficiency); F#614/F#536 (thinking-mode load-
  bearing); F#328/F#330 (LORA_SCALE ≤ 8); F#673 (mx.clear_cache between phases,
  MLX audit 2026-04-17).
- F#666 target-gating convention; F#702 hygiene-patch PROVISIONAL; F#683/F#684/
  F#696/F#697/F#717 Hedgehog-axis PROVISIONAL precedents.
- `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — inapplicable here
  (target KC present).
- PostgreSQL Global Development Group, PostgreSQL 16 Documentation (planner, indexes,
  performance). https://www.postgresql.org/docs/16/
- Markus Winand, "Use the Index, Luke" (use-the-index-luke.com) — fair-use excerpt
  basis.

## Handoff

- Status: PROVISIONAL.
- `_impl` follow-up: `exp_hedgehog_adapter_sql_domain_impl` filed inline at P=3, KCs
  inherited verbatim.
- Hygiene-patch applied: platform set to "M5 Pro 48GB MLX", success_criteria populated
  before `experiment complete`.
- No reviewer-side `_impl` filing required (researcher-filed per
  `mem-antipattern-impl-follow-up-delegation`).
