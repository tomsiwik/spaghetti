# MATH.md — exp_hedgehog_domain_adapter_js

**Claim.** Per-layer cos-sim distillation between (a) teacher = larger model with MDN JavaScript docs + Eloquent JavaScript excerpts in context answering JS-nuance questions (hoisting, closures, event loop, `this`-binding, prototypes) and (b) student = Gemma 4 E4B + rank-8 LoRA seeing only the JS-nuance prompt trains an adapter that encodes **JavaScript domain knowledge** as attention-routing perturbation, matching same-data token-space LoRA on JS benchmarks without degrading unrelated language (Python) or natural-language (MMLU) capability.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval` discipline at step boundaries, `mx.clear_cache` between phases, `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval, bandwidth-aware kernels). Both MUST be invoked before any MLX training-loop code lands in the `_impl` follow-up's `run_experiment.py`. Hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed at run time; API breakage between 0.21 and 0.31 has silently broken prior experiments.
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id). No proxy substitution per reviewer check (m).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` — larger variant, MDN excerpts + Eloquent-JS chapter summaries fit in its context alongside the JS-nuance question. If the 26B variant does not co-reside with the student on 48GB M5 Pro, use a sequential-phase pattern (teacher forward → cache attn traces to disk → evict → student forward) per F#673.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter target).
- **LoRA scale:** ≤ 8 per F#328/F#330. Default 6.0.
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop cannot land in a single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT objective, swap to token-space LoRA, or proxy the teacher to E4B. Those change what K1790–K1793 measure.

## 1. Failure mode

Primary degenerate behavior: "The adapter memorizes lexical idioms from MDN (literal function names, specific API call patterns) rather than *conceptual* JS knowledge. On held-out JS-nuance prompts with different surface wording — e.g. asking why a variable behaves a certain way with `var` vs `let` in a loop closure — answer quality collapses to baseline." Under this failure, K1790 may PASS on familiar patterns (teacher trace is similar to training) while K1791 fails on held-out benchmark.

Second failure mode: "Per-layer cos matches teacher routing for short single-turn factual prompts ('What is hoisting?') but not for longer code-reasoning prompts ('Trace the output of this closure-in-loop snippet'). K1790 PASS, K1791 PASS on factual, K1791 FAIL on reasoning." Partially accepted — reasoning-heavy JS is harder than factual recall; reported separately.

Third failure mode (cross-domain interference): "Training on JS docs attention-routing shifts general-code attention patterns; Python HumanEval drops ≥ 3pp." If K1792 fails, the adapter is NOT domain-narrow and composition with other code adapters would be unsafe.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim.
- **Zhang 2402.04347:** cosine loss recovers 99 % attention behavior with a small student.
- **MultiPL-E (Cassano et al. arxiv:2208.08227):** HumanEval translated to JavaScript and 17 other languages; canonical cross-language benchmark used for K1791.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` captures domain specialization; JS domain specialization is dimensionally similar (a lexical/syntactic routing bias, not a new knowledge store exceeding rank-8 capacity).
- **Pierre F#614 / F#536:** thinking mode is load-bearing on Gemma 4 E4B reasoning. JS-nuance "trace this closure" questions are multi-step reasoning → keep `enable_thinking=True` for both teacher and student.
- **F#263:** composition mechanism degrades MMLU knowledge recall ~5-6 pp regardless of training objective. Single-adapter MMLU drop should stay < 2 pp (K1793); composition behaviour is out-of-scope here (gated by `exp_hedgehog_composition_polite_refactor_js` PREEMPT-KILLED by F#688 pending all 3 parents target-SUPPORTED).
- **Sibling precedent:** `exp_hedgehog_behavior_adapter_politeness` (F#683 PROVISIONAL), `exp_hedgehog_procedural_adapter_refactor` (F#684 PROVISIONAL). Same training-loop blocker, same design-locked pattern.

## 3. Theorem (informal)

Let `Q` be a JS-nuance question, `π_JS` be MDN/Eloquent-JS excerpts relevant to `Q` fit into teacher context. Define:
- Teacher attention trace: `A_teacher_l = attn_l(π_JS ⊕ Q; θ_base)` on the 26B teacher.
- Student attention trace: `A_student_l = attn_l(Q; θ_base + Δθ)` on the 4B student.

**Theorem.** There exists rank-8 `Δθ` on `(v_proj, o_proj)` such that:
1. `E[cos(A_teacher_l, A_student_l)] > 0.80` over held-out `(π_JS, Q)` pairs (K1790),
2. JS-benchmark accuracy ≥ same-data token-space LoRA at matched rank (K1791),
3. HumanEval (Python) pass@1 drop < 3 pp (K1792),
4. MMLU subset drop < 2 pp (K1793).

**Proof sketch.**
1. *Existence.* Same argument as `exp_hedgehog_behavior_adapter_politeness` — rank-8 LoRA on v/o has more DOF than a per-head MLP (Zhang 2024 sufficiency). JS domain knowledge is lexical/syntactic + conceptual but fits below rank-8 given F#627 v_proj+o_proj sufficiency.
2. *Knowledge transfer (K1790 → K1791).* The answer `A` to `Q` is a function of both `Q` and `π_JS`. Per-layer cos-sim forces student to reproduce teacher's attention output *without seeing π_JS*, meaning student must absorb the dependence on π_JS into `Δθ`. If K1790 PASS, the learned routing mimics "I-have-seen-π_JS" behavior, and answer accuracy follows (Lipschitz on attention → residual stream → logits).
3. *K1791 ≥ token-space LoRA baseline.* Token-space LoRA (trained on `(Q, A)` pairs with next-token CE) matches *answer tokens*. Cos-sim LoRA matches *attention routing*. The claim is cos-sim ≥ token-space for domain-knowledge transfer; the rigorous bound is open — reported empirically.
4. *Non-interference (K1792, K1793).* Rank-8 on `v_proj+o_proj` ≪ base rank. Perturbation magnitude is bounded. Per F#627 no collapse pattern on unrelated tasks at this rank. Cross-language (JS → Python) interference is the load-bearing check because JS and Python both activate "code" attention heads; if K1792 fails, the adapter is a general-code perturbation rather than JS-specific.

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1790 | mean per-layer cos on held-out JS-nuance pairs | > 0.80 | structural proxy |
| K1791 | JS-benchmark accuracy vs same-data token-space LoRA baseline | ≥ baseline | target (pair K1790 per F#666) |
| K1792 | HumanEval pass@1 drop vs base | < 3 pp | target non-interference |
| K1793 | MMLU subset drop vs base | < 2 pp | target specificity |

**JS benchmark choice (K1791).** Primary: HumanEval-JS (MultiPL-E arxiv:2208.08227 translation of canonical HumanEval to JavaScript). Secondary (if HumanEval-JS dataset not available): custom 100-item JS-nuance suite targeting hoisting, closures, event loop, `this`-binding, prototype chain — graded by auto-judge against reference answers derived from MDN.

**Auto-judge rubric:** 0–10, graded by (a) syntactic validity (does the code parse?), (b) factual correctness matching MDN canonical answer, (c) idiomatic use of JS-specific constructs where relevant.

## 5. Predicted measurements

- K1790: cos ∈ [0.80, 0.88], mean ≈ 0.83 (comparable to procedural sibling; JS content is more specific than politeness style).
- K1791: auto-judge Δ ∈ [0, +1.5] vs token-space LoRA (roughly matched; cos-sim distillation hypothesis is cos-sim ≥ token-space for knowledge transfer, small but non-zero advantage expected).
- K1792: HumanEval drop ≤ 2 pp (JS-specific adapter should not corrupt Python).
- K1793: MMLU drop ≤ 1 pp (narrow code-domain adapter should not affect general NL).

If K1791 fails (token-space LoRA wins), the finding is: **cos-sim distillation is better for behavior/style than for domain-knowledge transfer at this scale** — a useful axis-specificity finding complementing the F#683/F#684 pair.

If K1792 fails, the finding is: **JS-domain Hedgehog training induces cross-language code-routing collapse** — adapter is not composable with other code-domain adapters; renegotiate composition scope.

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset curation.** MDN excerpts (fair-use summaries) for 6 focus topics (hoisting & TDZ, closures & scope, `this`-binding & arrow functions, event loop & microtasks, prototype chain, async/await & error handling). Eloquent JavaScript (CC-BY-NC 3.0) chapter summaries. Generate 200 train + 50 held-out (Q, A) pairs via larger model with source-in-context. Stratified split by focus topic.

2. **Phase A — Teacher attention capture.** 26B Gemma 4 + `π_JS` (topic-relevant excerpts) + `Q` in context. Capture `{layer_idx: attn_output}` for all 42 layers. Sequential-phase eviction on 48GB: teacher forward → cache → evict → student forward. Precompute all teacher traces for train + held-out in an offline pass; stream from disk during student training to keep peak memory below 40GB.

3. **Phase B — Student training.** Rank-8 LoRA on v_proj + o_proj with per-layer cos-sim loss: `L = mean_l (1 − cos(A_teacher_l, A_student_l))`. 800 steps, AdamW, `mx.eval + mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)` functional gradients — no Torch-style `.backward()`.

4. **Phase Baseline — Token-space LoRA.** Same (Q, A) pairs, standard `mlx_lm.lora` next-token CE, matched rank/targets/scale/steps. Needed for K1791 head-to-head.

5. **Phase C — K1790 eval.** held-out cos-sim (mean over 50 prompts × 42 layers).

6. **Phase D — K1791 / K1792 / K1793 eval.**
   - K1791: generate with both adapters on held-out JS prompts + HumanEval-JS; auto-judge.
   - K1792: HumanEval Python pass@1 (full, 164 problems).
   - K1793: MMLU subset (10 categories × 20 questions = 200 items).

## 7. Locked KCs — no edits after data collection

KCs K1790, K1791, K1792, K1793 are pre-registered in the DB (text exactly as filed). Any post-hoc relaxation or addition invalidates the run (verdict-consistency check #5).

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1.** 26B Gemma 4 teacher with JS docs in context produces substantively-better JS-nuance answers than 4B student alone; if not, K1790 is ill-defined (teacher gap absent). Validation: spot-check n=10 pairs before full run.
- **A2.** MDN + Eloquent-JS excerpts fit in 128k teacher context per-topic; topic-segmented π_JS (not monolithic corpus) keeps each teacher forward under 32k tokens.
- **A3.** HumanEval-JS via MultiPL-E is accepted as the JS benchmark; fallback to custom nuance suite if unavailable.
- **A4.** Auto-judge (Gemma 4 E4B-as-judge or API judge with pinned version) is reliable enough for K1791 head-to-head; blind-paired setup reduces absolute-score bias.
- **A5.** Researcher-hat guardrail 1009 caps single-iteration work at 30 min / 40 tool calls; full 4–6 h pipeline is explicitly out of scope without a dedicated `_impl` iteration.
- **A6.** LORA_SCALE ≤ 8 per F#328/F#330.
