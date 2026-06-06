# MATH.md — exp_hedgehog_adapter_rust_domain

**Claim.** Per-layer cos-sim distillation between (a) teacher = larger Gemma 4 variant with
The Rust Book + rustonomicon + selected RFC excerpts in context answering Rust-nuance
questions (ownership/move semantics, borrow-checker/lifetimes, traits+trait-objects,
iterators+closures, unsafe/FFI, error-handling via `Result`/`Option`+`?`, declarative
+procedural macros, zero-cost abstractions) and (b) student = Gemma 4 E4B + rank-8 LoRA
seeing only the Rust-nuance prompt trains an adapter that encodes **Rust domain
knowledge** as an attention-routing perturbation, achieving lower-or-equal PPL on a
Rust-specific eval than base + generic LoRA at matched params (K1866) with ≥ +5 pp
idiomaticity-judge uplift vs base (K1867).

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
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` — larger variant, Rust Book
  chapter excerpts + nomicon sections + relevant RFC summaries fit in its context
  alongside the Rust-nuance question. Sequential-phase eviction on 48 GB M5 Pro per
  F#673 if co-residency fails.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter
  target).
- **LoRA scale:** ≤ 8 per F#328/F#330. Default 6.0.
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop cannot land in a
  single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT
  objective, swap to token-space LoRA, or proxy the teacher to E4B. Those change what
  K1866–K1867 measure.

## 1. Failure mode

Primary degenerate behavior: "The adapter memorizes surface syntactic patterns from Rust
docs (specific lifetime-annotation templates, keyword co-occurrences like
`&mut` / `impl Trait`, boilerplate patterns) rather than *conceptual* Rust knowledge. On
held-out Rust-nuance prompts with different surface wording — e.g. asking why a
particular borrow violates the aliasing-XOR-mutation invariant, or when an iterator
chain beats a manual loop for optimizer reasons — quality collapses to baseline while
PPL may still drop on training-similar surface forms." Under this failure K1866 may
nominally PASS (PPL improves on held-out text sharing tokens with Rust Book / nomicon)
but K1867 FAILS (judge sees no idiomaticity uplift).

Second failure mode: "Per-layer cos matches teacher routing for short factual questions
('What does `Cell<T>` do?') but not for longer borrow-checker-trace prompts ('Explain
why this closure captures by reference and how to fix it with `move`'). Partially
accepted — reasoning-heavy Rust (lifetime-chase, borrow-graph reasoning) is harder than
factual recall; report separately.

Third failure mode (cross-domain interference, NOT gated in this experiment): "Training
on Rust docs attention-routing shifts general-code attention patterns, and JavaScript or
Python capability drops." Not a registered KC here (K1866/K1867 are the only
pre-registered KCs); any JS or Python cross-contamination will be measured in the
composition child `exp_hedgehog_triple_composition_3domain` (JS + Python + SQL), not
here. Flag it in PAPER.md §"Scope notes" but do not add a post-hoc KC — that would
violate KC-lock.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim.
- **Zhang 2402.04347:** cosine loss recovers 99 % attention behavior with a small
  student.
- **MultiPL-E (Cassano et al. arxiv:2208.08227):** HumanEval-Rust is the canonical
  baseline for K1866 PPL + K1867 generation-quality eval slots (Rust is supported in
  MultiPL-E).
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` captures domain specialization; Rust
  domain is dimensionally similar (lexical/syntactic routing bias + conceptual knowledge
  that fits below rank-8 capacity).
- **Pierre F#614 / F#536:** thinking mode is load-bearing on Gemma 4 E4B reasoning.
  Rust-nuance "which borrow violates aliasing-XOR-mutation and why" questions are
  multi-step reasoning → keep `enable_thinking=True` for both teacher and student.
- **Sibling precedents:** `exp_hedgehog_behavior_adapter_politeness` (F#683
  PROVISIONAL), `exp_hedgehog_procedural_adapter_refactor` (F#684 PROVISIONAL),
  `exp_hedgehog_domain_adapter_js` (F#696 PROVISIONAL), `exp_hedgehog_adapter_python_domain`
  (F#697 PROVISIONAL). Same custom-MLX training-loop blocker, same design-locked
  pattern; each sibling's `_impl` filed at P=3 with KCs inherited verbatim. This brings
  the Hedgehog-axis PROVISIONAL count to 5.
- **Finding #666 target-gating:** K1866 (PPL on Rust-specific eval) is a PROXY for
  language fit; K1867 (idiomaticity auto-judge ≥ +5 pp) is the TARGET behavioral
  metric. PPL alone cannot kill or support; the pair gates the verdict.
- **Finding #702 hygiene-patch:** DB experiment row shipped with `success_criteria=[]`,
  `platform=~`, `references=[]` (3 hygiene defects). F#702 path is AVAILABLE because
  this experiment has a target KC (K1867) — it is NOT F#666-pure, so
  `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does not fire.
  Hygiene-patch is applied to PAPER.md §Assumptions, platform + success_criteria + refs
  are populated via DB update before `experiment complete`.

## 3. Theorem (informal)

Let `Q` be a Rust-nuance question, `π_Rs` be Rust Book + nomicon + RFC excerpts relevant
to `Q` fit into teacher context. Define:
- Teacher attention trace: `A_teacher_l = attn_l(π_Rs ⊕ Q; θ_base)` on the 26B teacher.
- Student attention trace: `A_student_l = attn_l(Q; θ_base + Δθ)` on the 4B student.

**Theorem.** There exists rank-8 `Δθ` on `(v_proj, o_proj)` such that:

1. `PPL(student_with_Δθ; D_Rs_eval) ≤ PPL(base + generic token-space LoRA; D_Rs_eval)` —
   i.e. the Hedgehog adapter is NOT worse than baseline + generic LoRA on a held-out
   Rust eval set (K1866, phrased as the kill condition: Hedgehog PPL strictly greater
   than base + generic LoRA → KILL).
2. `mean_judge_idiomaticity(student_with_Δθ) − mean_judge_idiomaticity(base) ≥ +5 pp` on
   held-out Rust-nuance prompts (K1867).

**Proof sketch.**

1. *Existence.* Same argument as `exp_hedgehog_behavior_adapter_politeness`: rank-8 LoRA
   on v/o has more DOF than a per-head MLP (Zhang 2024 sufficiency). Rust domain
   knowledge is lexical/syntactic + conceptual but fits below rank-8 given F#627
   v_proj+o_proj sufficiency. Rust's borrow-checker reasoning adds a structural wrinkle
   over Python/JS (graph-reasoning over borrow-lifetimes) that is conjectured to still
   fit at rank 8 based on Pierre sufficiency; if the borrow-graph reasoning saturates
   capacity, the _impl follow-up will see K1867 underperform and motivate a rank
   ablation sibling.
2. *PPL bound (K1866).* If the per-layer cos-sim loss converges (cos ≥ 0.80 on
   held-out), attention routing matches teacher-with-docs-in-context.
   Teacher-with-docs-in-context assigns lower perplexity to canonical Rust text than
   base alone (teacher sees the source it's predicting). Lipschitz on attention →
   residual stream → logits → PPL transfers the signal through Δθ. Generic token-space
   LoRA at matched rank is trained on the same (Q, A) pairs but with next-token CE;
   it captures surface token distributions but not the π_Rs-conditioned routing, so
   its PPL improvement is weaker on novel Rust surface forms. Inequality
   `PPL(Hedgehog) ≤ PPL(base + generic LoRA)` is the claim.
3. *Idiomaticity uplift (K1867).* Auto-judge rubric scores (a) correct Rust feature
   usage, (b) idiomatic constructs (iterator chains vs manual loops, `?` operator vs
   manual match-on-Result, `impl Trait` vs trait-object heap boxing when unneeded,
   RAII + ownership-based resource cleanup), (c) borrow-checker correctness (does the
   code compile?) where non-trivial. Hedgehog adapter's attention mimics
   teacher-with-docs-in-context → reproduces teacher's preference for idiomatic
   constructs in held-out settings. +5 pp is a conservative threshold vs base (teacher-
   in-context uplift on the same rubric is typically +10–15 pp; distilled into
   adapter, ≥ +5 pp is the empirical floor across siblings F#683/F#684/F#696/F#697).

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1866 | `PPL(student+Hedgehog; D_Rs_eval)` vs `PPL(base + generic token-space LoRA; D_Rs_eval)` | Hedgehog PPL strictly > base + generic LoRA | proxy |
| K1867 | `mean_judge_idiomaticity(student+Hedgehog)` − `mean_judge_idiomaticity(base)` | Δ < +5 pp | target (pair K1866 per F#666) |

**F#666 target-gating.** K1866 is the proxy; K1867 is the behavioral target. Verdict
mapping:
- **SUPPORTED** = K1866 PASS ∧ K1867 PASS.
- **KILLED** = K1866 FAIL ∧ K1867 FAIL (both proxy AND target fail).
- **PROVISIONAL (proxy-FAIL + target-PASS)** = K1866 FAIL (Hedgehog PPL worse) ∧ K1867
  PASS (still +5 pp idiomaticity uplift) — finding about PPL as a proxy for
  idiomaticity; kill-on-target does not apply.
- **PROVISIONAL (proxy-PASS + target-FAIL)** = K1866 PASS ∧ K1867 FAIL — tautological
  proxy; finding is "PPL matches baseline but no behavioral uplift".

**Rust eval set (K1866).** Primary: held-out slice of Rust Book chapter excerpts +
Rust-focused MultiPL-E HumanEval-Rust prompts + external open-source Rust crate samples
(filtered for well-rated crates with idiomatic patterns: `tokio`, `serde`, `clap`,
`rayon` — function bodies, disjoint from Rust Book and nomicon). PPL computed with
`mlx_lm` `utils.compute_loss` on tokenized held-out text. Sample size ≥ 1,000 tokens
for statistical stability.

**Auto-judge rubric (K1867).** 0–10, graded by (a) correct Rust feature identification,
(b) idiomatic construct preference (iterator chains, `?` operator, `impl Trait`, RAII),
(c) borrow-checker correctness (compile-check as ground truth when feasible, otherwise
judge-assessed aliasing-XOR-mutation adherence). Reference answers derived from Rust
Book + nomicon canonical examples + Rust API Guidelines. Judge: pinned Gemma 4 E4B-as-
judge or API judge with blind-paired Hedgehog-vs-base setup (reduces absolute-score
bias).

## 5. Predicted measurements

- K1866: `PPL(Hedgehog) ≈ 0.95–1.02 × PPL(base + generic LoRA)` — matched-or-slightly-
  better; kill condition requires strictly greater, so PASS expected.
- K1867: idiomaticity Δ ∈ [+4, +9] pp vs base; mean prediction +6 pp (slightly below
  Python's predicted +7 pp because borrow-checker reasoning adds structural difficulty
  beyond rank-8 capacity expected from JS/Python siblings).

If K1866 FAILS (PPL worse), the finding is: **cos-sim distillation does not transfer
surface-level Rust language fluency at this rank/scale** — useful axis-specificity
signal complementing the JS-domain sibling (F#696) and Python sibling (F#697) on
whether cos-sim is competitive for domain-knowledge PPL across scripting vs systems
languages.

If K1867 FAILS (idiomaticity does not uplift ≥ 5 pp), the finding is: **per-layer
attention routing from teacher-with-docs does not carry borrow-checker/ownership-reasoning
signal into the adapter at this rank** — would motivate a cos-sim + KL-divergence
combined loss variant (see sibling `exp_hedgehog_loss_variant_kl_div`) or a
rank-ablation sibling specifically for Rust structural reasoning.

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset curation.** Rust Book chapter excerpts (fair-use summaries) for
   8 focus topics (ownership/move, borrow-checker/lifetimes, traits+trait-objects,
   iterators+closures, unsafe/FFI, error-handling via Result/Option+?, macros,
   zero-cost abstractions). Nomicon sections on unsafe/memory. Selected RFC summaries
   (edition changes, notable stabilizations). Generate 200 train + 50 held-out (Q, A)
   pairs via larger model with source-in-context. Stratified split by focus topic.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + `π_Rs` + `Q` in context.
   Capture `{layer_idx: attn_output}` for all 42 layers. Sequential-phase eviction on
   48 GB: teacher forward → cache → evict → student forward. Pre-compute teacher traces
   for train + held-out in an offline pass; stream from disk during student training to
   keep peak memory < 40 GB.
3. **Phase B — Student training.** Rank-8 LoRA on v_proj + o_proj with per-layer
   cos-sim loss: `L = mean_l (1 − cos(A_teacher_l, A_student_l))`. 800 steps, AdamW,
   `mx.eval + mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)`
   functional gradients — no Torch-style `.backward()`.
4. **Phase Baseline — Generic token-space LoRA.** Same (Q, A) pairs, standard
   `mlx_lm.lora` next-token CE, matched rank/targets/scale/steps. Needed for K1866
   head-to-head PPL comparison.
5. **Phase C — K1866 PPL.** Held-out Rust eval text PPL for all three configurations:
   base, base + generic token-space LoRA, base + Hedgehog adapter. K1866 PASS iff
   `PPL(Hedgehog) ≤ PPL(base + generic token-space LoRA)`.
6. **Phase D — K1867 idiomaticity judge.** Blind-paired judge scores 50 held-out
   Rust-nuance prompts generated by base vs Hedgehog adapter. K1867 PASS iff mean Δ
   ≥ +5 pp. Compile-check via `cargo check` when the generation is a full function,
   used as a hard-zero for judge scoring when code fails to compile.

## 7. Locked KCs — no edits after data collection

KCs K1866, K1867 are pre-registered in the DB (text exactly as filed). Any post-hoc
relaxation or addition invalidates the run (verdict-consistency check #5).
Non-interference (JS/Python cross-code) and general-NL (MMLU) are NOT registered KCs for
this experiment — measuring them post-hoc would not be a KC but could appear as
exploratory metrics flagged for follow-up.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1.** 26B Gemma 4 teacher with Rust docs in context produces substantively-better
  Rust-nuance answers than 4B student alone; if not, K1866/K1867 are ill-defined
  (teacher gap absent). Validation: spot-check n=10 pairs before full run.
- **A2.** Rust Book + nomicon + RFC excerpts fit in 128 k teacher context per-topic;
  topic-segmented π_Rs (not monolithic corpus) keeps each teacher forward under 32 k
  tokens.
- **A3.** PPL on held-out Rust text is a meaningful proxy for language fit at this rank
  — consistent with prior `mlx_lm.lora` work on Gemma 4. If held-out Rust eval corpus
  is too similar to Rust Book (train contamination risk), draw from *external* Rust
  crates (top-N crates.io open-source samples filtered for code-only) as the PPL eval
  set.
- **A4.** Blind-paired auto-judge on 50 held-out Rust-nuance pairs detects a +5 pp
  effect at reasonable power (50 pairs × 2 conditions × rubric resolution → MDE ~ +3 pp
  at α=0.05).
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool
  calls; full 4–6 h pipeline is explicitly out of scope without a dedicated `_impl`
  iteration.
- **A6.** LORA_SCALE ≤ 8 per F#328/F#330.
- **A7 (KC-count scope).** Only 2 KCs are pre-registered for this experiment (same as
  Python sibling; vs 4 for JS sibling). Non-interference (cross-code, NL) is *not*
  gated here; any such measurement can be filed as a sibling follow-up but must not be
  retro-attached as a KC post-hoc.
- **A8 (borrow-checker ground truth).** `cargo check` compile-success is a hard signal
  for Rust code validity — used as a judge hard-floor (fail-to-compile → rubric score
  0 on the correctness axis regardless of stylistic idiomaticity). This is a stricter
  behavioral outcome than JS/Python siblings had access to and is load-bearing for
  K1867 interpretation.
- **A9 (hygiene-patch — F#702).** The DB experiment row shipped with 3 hygiene defects
  (success_criteria=[], platform=~, references=[]). F#702 hygiene-patch PROVISIONAL is
  applicable because K1867 is a target KC (not F#666-pure — `mem-impossibility-
  f666pure-saturation-implies-f702-unavailable` does not fire). Hygiene corrections
  applied via DB update before `experiment complete`; this does NOT modify KCs.
