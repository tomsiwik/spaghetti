# MATH.md — exp_hedgehog_adapter_python_domain

**Claim.** Per-layer cos-sim distillation between (a) teacher = larger Gemma 4 variant with CPython docs + PEP-8/PEP-20/PEP-257/PEP-484 excerpts in context answering Python-nuance questions (duck-typing, context managers, generators/iterators, decorators, GIL vs asyncio, comprehensions, descriptors) and (b) student = Gemma 4 E4B + rank-8 LoRA seeing only the Python-nuance prompt trains an adapter that encodes **Python domain knowledge** as an attention-routing perturbation, achieving lower-or-equal PPL on a Python-specific eval than base + generic LoRA at matched params (K1844) with ≥ +5 pp idiomaticity-judge uplift vs base (K1845).

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval` discipline at step boundaries, `mx.clear_cache` between phases, `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval, bandwidth-aware kernels). Both MUST be invoked before any MLX training-loop code lands in the `_impl` follow-up's `run_experiment.py`. Hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed at run time; API breakage between 0.21 and 0.31 has silently broken prior experiments.
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id). No proxy substitution per reviewer check (m).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` — larger variant, CPython doc excerpts + relevant PEP summaries fit in its context alongside the Python-nuance question. Sequential-phase eviction on 48 GB M5 Pro per F#673 if co-residency fails.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter target).
- **LoRA scale:** ≤ 8 per F#328/F#330. Default 6.0.
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop cannot land in a single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT objective, swap to token-space LoRA, or proxy the teacher to E4B. Those change what K1844–K1845 measure.

## 1. Failure mode

Primary degenerate behavior: "The adapter memorizes surface syntactic idioms from CPython docs (specific `__dunder__` method names, boilerplate patterns) rather than *conceptual* Python knowledge. On held-out Python-nuance prompts with different surface wording — e.g. asking when a generator expression is preferred over a list comprehension and why — quality collapses to baseline while PPL may still drop on training-similar surface forms." Under this failure K1844 may nominally PASS (PPL improves on held-out text sharing tokens with CPython docs) but K1845 FAILS (judge sees no idiomaticity uplift).

Second failure mode: "Per-layer cos matches teacher routing for short factual questions ('What does `yield` do?') but not for longer code-trace prompts ('Explain why this decorator breaks `self` binding'). Partially accepted — reasoning-heavy Python is harder than factual recall; report separately.

Third failure mode (cross-domain interference, NOT gated in this experiment): "Training on Python docs attention-routing shifts general-code attention patterns, and JavaScript or Rust capability drops." Not a registered KC here (K1844/K1845 are the only pre-registered KCs); any JS or Rust cross-contamination will be measured in the composition child `exp_hedgehog_triple_composition_3domain` (JS + Python + SQL), not here. Flag it in PAPER.md §"Scope notes" but do not add a post-hoc KC — that would violate KC-lock.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim.
- **Zhang 2402.04347:** cosine loss recovers 99 % attention behavior with a small student.
- **MultiPL-E (Cassano et al. arxiv:2208.08227):** HumanEval Python is the canonical baseline for K1844 PPL + K1845 generation-quality eval slots.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` captures domain specialization; Python domain is dimensionally similar (lexical/syntactic routing bias + conceptual knowledge that fits below rank-8 capacity).
- **Pierre F#614 / F#536:** thinking mode is load-bearing on Gemma 4 E4B reasoning. Python-nuance "which of these implementations is idiomatic and why" questions are multi-step reasoning → keep `enable_thinking=True` for both teacher and student.
- **Sibling precedents:** `exp_hedgehog_behavior_adapter_politeness` (F#683 PROVISIONAL), `exp_hedgehog_procedural_adapter_refactor` (F#684 PROVISIONAL), `exp_hedgehog_domain_adapter_js` (F#696 PROVISIONAL). Same custom-MLX training-loop blocker, same design-locked pattern; each sibling's `_impl` filed at P=3 with KCs inherited verbatim.
- **Finding #666 target-gating:** K1844 (PPL on Python-specific eval) is a PROXY for language fit; K1845 (idiomaticity auto-judge ≥ +5 pp) is the TARGET behavioral metric. PPL alone cannot kill or support; the pair gates the verdict.

## 3. Theorem (informal)

Let `Q` be a Python-nuance question, `π_Py` be CPython-doc + PEP excerpts relevant to `Q` fit into teacher context. Define:
- Teacher attention trace: `A_teacher_l = attn_l(π_Py ⊕ Q; θ_base)` on the 26B teacher.
- Student attention trace: `A_student_l = attn_l(Q; θ_base + Δθ)` on the 4B student.

**Theorem.** There exists rank-8 `Δθ` on `(v_proj, o_proj)` such that:

1. `PPL(student_with_Δθ; D_Py_eval) ≤ PPL(base; D_Py_eval) + PPL(base + generic token-space LoRA; D_Py_eval)` — i.e. the Hedgehog adapter is NOT worse than baseline + generic LoRA on a held-out Python eval set (K1844, phrased as the kill condition: Hedgehog PPL strictly greater than base + generic LoRA → KILL).
2. `mean_judge_idiomaticity(student_with_Δθ) − mean_judge_idiomaticity(base) ≥ +5 pp` on held-out Python-nuance prompts (K1845).

**Proof sketch.**

1. *Existence.* Same argument as `exp_hedgehog_behavior_adapter_politeness`: rank-8 LoRA on v/o has more DOF than a per-head MLP (Zhang 2024 sufficiency). Python domain knowledge is lexical/syntactic + conceptual but fits below rank-8 given F#627 v_proj+o_proj sufficiency.
2. *PPL bound (K1844).* If the per-layer cos-sim loss converges (cos ≥ 0.80 on held-out), attention routing matches teacher-with-docs-in-context. Teacher-with-docs-in-context assigns lower perplexity to canonical Python text than base alone (teacher sees the source it's predicting). Lipschitz on attention → residual stream → logits → PPL transfers the signal through Δθ. Generic token-space LoRA at matched rank is trained on the same (Q, A) pairs but with next-token CE; it captures surface token distributions but not the π_Py-conditioned routing, so its PPL improvement is weaker on novel Python surface forms. Inequality `PPL(Hedgehog) ≤ PPL(base + generic LoRA)` is the claim.
3. *Idiomaticity uplift (K1845).* Auto-judge rubric scores (a) correct Python feature usage, (b) idiomatic constructs (list vs generator comprehensions, context-manager vs try/finally, ``@property`` vs explicit getter), (c) PEP-8 conformance where non-trivial. Hedgehog adapter's attention mimics teacher-with-docs-in-context → reproduces teacher's preference for idiomatic constructs in held-out settings. +5 pp is a conservative threshold vs base (teacher-in-context uplift on the same rubric is typically +10–15 pp; distilled into adapter, ≥ +5 pp is the empirical floor).

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1844 | `PPL(student+Hedgehog; D_Py_eval)` vs `PPL(base + generic token-space LoRA; D_Py_eval)` | Hedgehog PPL strictly > base + generic LoRA | proxy |
| K1845 | `mean_judge_idiomaticity(student+Hedgehog)` − `mean_judge_idiomaticity(base)` | Δ < +5 pp | target (pair K1844 per F#666) |

**F#666 target-gating.** K1844 is the proxy; K1845 is the behavioral target. Verdict mapping:
- **SUPPORTED** = K1844 PASS ∧ K1845 PASS.
- **KILLED** = K1844 FAIL ∧ K1845 FAIL (both proxy AND target fail).
- **PROVISIONAL (proxy-FAIL + target-PASS)** = K1844 FAIL (Hedgehog PPL worse) AND K1845 PASS (still +5 pp idiomaticity uplift) — finding about PPL as a proxy for idiomaticity; kill-on-target does not apply.
- **PROVISIONAL (proxy-PASS + target-FAIL)** = K1844 PASS ∧ K1845 FAIL — tautological proxy; finding is "PPL matches baseline but no behavioral uplift".

**Python eval set (K1844).** Primary: held-out slice of CPython doc excerpts + Python-focused HumanEval prompts (164 problems). PPL computed with `mlx_lm` `utils.compute_loss` on tokenized held-out text. Sample size ≥ 1,000 tokens for statistical stability.

**Auto-judge rubric (K1845).** 0–10, graded by (a) correct Python feature identification, (b) idiomatic construct preference (comprehensions, context managers, decorators), (c) PEP-8 conformance where load-bearing. Reference answers derived from CPython doc canonical examples + PEP-20 stylistic principles. Judge: pinned Gemma 4 E4B-as-judge or API judge with blind-paired Hedgehog-vs-base setup (reduces absolute-score bias).

## 5. Predicted measurements

- K1844: `PPL(Hedgehog) ≈ 0.95–1.02 × PPL(base + generic LoRA)` — matched-or-slightly-better; kill condition requires strictly greater, so PASS expected.
- K1845: idiomaticity Δ ∈ [+5, +10] pp vs base; mean prediction +7 pp (between no-uplift 0 pp and teacher-in-context ceiling ~+12 pp).

If K1844 FAILS (PPL worse), the finding is: **cos-sim distillation does not transfer surface-level language fluency at this rank/scale** — useful axis-specificity signal complementing the JS-domain sibling (F#696) on whether cos-sim is competitive for domain-knowledge PPL.

If K1845 FAILS (idiomaticity does not uplift ≥ 5 pp), the finding is: **per-layer attention routing from teacher-with-docs does not carry idiomaticity signal into the adapter at this rank** — would motivate a cos-sim + KL-divergence combined loss variant (see sibling `exp_hedgehog_loss_variant_kl_div`).

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset curation.** CPython doc excerpts (fair-use summaries) for 7 focus topics (duck-typing, context managers, generators/iterators, decorators, GIL vs asyncio, comprehensions, descriptors). PEP-8/PEP-20/PEP-257/PEP-484 canonical text. Generate 200 train + 50 held-out (Q, A) pairs via larger model with source-in-context. Stratified split by focus topic.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + `π_Py` + `Q` in context. Capture `{layer_idx: attn_output}` for all 42 layers. Sequential-phase eviction on 48 GB: teacher forward → cache → evict → student forward. Pre-compute teacher traces for train + held-out in an offline pass; stream from disk during student training to keep peak memory < 40 GB.
3. **Phase B — Student training.** Rank-8 LoRA on v_proj + o_proj with per-layer cos-sim loss: `L = mean_l (1 − cos(A_teacher_l, A_student_l))`. 800 steps, AdamW, `mx.eval + mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)` functional gradients — no Torch-style `.backward()`.
4. **Phase Baseline — Generic token-space LoRA.** Same (Q, A) pairs, standard `mlx_lm.lora` next-token CE, matched rank/targets/scale/steps. Needed for K1844 head-to-head PPL comparison.
5. **Phase C — K1844 PPL.** Held-out Python eval text PPL for all three configurations: base, base + generic token-space LoRA, base + Hedgehog adapter. K1844 PASS iff `PPL(Hedgehog) ≤ PPL(base + generic token-space LoRA)`.
6. **Phase D — K1845 idiomaticity judge.** Blind-paired judge scores 50 held-out Python-nuance prompts generated by base vs Hedgehog adapter. K1845 PASS iff mean Δ ≥ +5 pp.

## 7. Locked KCs — no edits after data collection

KCs K1844, K1845 are pre-registered in the DB (text exactly as filed). Any post-hoc relaxation or addition invalidates the run (verdict-consistency check #5). Non-interference (JS/Rust cross-code) and general-NL (MMLU) are NOT registered KCs for this experiment — measuring them post-hoc would not be a KC but could appear as exploratory metrics flagged for follow-up.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1.** 26B Gemma 4 teacher with Python docs in context produces substantively-better Python-nuance answers than 4B student alone; if not, K1844/K1845 are ill-defined (teacher gap absent). Validation: spot-check n=10 pairs before full run.
- **A2.** CPython doc + PEP excerpts fit in 128 k teacher context per-topic; topic-segmented π_Py (not monolithic corpus) keeps each teacher forward under 32 k tokens.
- **A3.** PPL on held-out Python text is a meaningful proxy for language fit at this rank — consistent with prior `mlx_lm.lora` work on Gemma 4. If held-out Python eval corpus is too similar to CPython docs (train contamination risk), draw from *external* Python repositories (top-N PyPI open-source samples filtered for code-only) as the PPL eval set.
- **A4.** Blind-paired auto-judge on 50 held-out Python-nuance pairs detects a +5 pp effect at reasonable power (50 pairs × 2 conditions × rubric resolution → MDE ~ +3 pp at α=0.05).
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; full 4–6 h pipeline is explicitly out of scope without a dedicated `_impl` iteration.
- **A6.** LORA_SCALE ≤ 8 per F#328/F#330.
- **A7 (KC-count scope).** Only 2 KCs are pre-registered for this experiment (vs 4 for JS sibling). Non-interference (cross-code, NL) is *not* gated here; any such measurement can be filed as a sibling follow-up but must not be retro-attached as a KC post-hoc.
