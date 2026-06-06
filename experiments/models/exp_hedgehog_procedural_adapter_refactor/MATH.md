# MATH.md — exp_hedgehog_procedural_adapter_refactor

**Claim:** Per-layer cos-sim distillation between (a) teacher = larger model + Fowler-catalog-entry in context performing refactor and (b) student = Gemma 4 E4B + rank-8 LoRA seeing only the pre-refactor code trains an adapter that encodes **procedural refactoring knowledge** as attention-routing perturbation, matching a same-data token-space LoRA on refactor quality without degrading general code tasks.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval` discipline at step boundaries, `mx.clear_cache` between phases, `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval, bandwidth-aware kernels). Both MUST be invoked before any MLX training-loop code lands in `run_experiment.py`. Hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as the installed version in the pueue venv at run time; API breakage between 0.21 and 0.31 has silently broken prior experiments.
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id). No proxy substitution per reviewer check (m).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` — larger variant, Fowler catalog entry fits in its context alongside code. If the 26B variant does not fit on 48GB M5 Pro alongside the student, use a sequential-phase pattern (teacher forward → cache attn traces → evict → student forward) rather than simultaneous residency.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627 — proven Gemma 4 E4B adapter target).
- **LoRA scale:** ≤ 8 per F#328/F#330. Default 6.0.
- **Scope-preservation (antipattern-t).** If the Hedgehog training loop (Phase B) cannot land in a single iteration, file PROVISIONAL; do NOT silently substitute a cross-entropy SFT objective or swap to token-space LoRA. Doing so changes what K1–K4 measure.

## 1. Failure mode

Degenerate behavior: "The adapter memorizes surface patterns from the Fowler catalog (literal variable names, specific call patterns) rather than procedural transformation rules. On held-out code with structurally-matching refactor opportunities but different identifiers, refactor quality collapses to baseline." Under this failure, K2 fails on held-out, K4 specificity fails.

Second failure: "Per-layer cos matches teacher routing on *seen catalog entries* but not on *compositional refactors* (apply Extract + Rename in sequence). K1 PASS, K2 PASS on single-step, K2 FAIL on multi-step." We partially accept this — multi-step refactor is a known hard case, not a kill, but will be reported.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim.
- **Zhang 2402.04347:** cosine loss recovers 99% attention behavior.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` captures domain specialization; procedural specialization is dimensionally similar (a routing bias, not a new knowledge store).
- **Pierre F#614 / F#536:** thinking mode is load-bearing on Gemma 4 E4B reasoning tasks. Refactor = multi-step reasoning → we keep `enable_thinking=True` for both teacher and student.

## 3. Theorem (informal)

Let `R` be a refactor catalog entry, `c_pre` a code snippet exhibiting the refactor opportunity, `c_post = R(c_pre)` the teacher's target. Define:
- Teacher attention trace: `A_teacher_l = attn_l(π_R ⊕ c_pre; θ_base)` where `π_R` is the catalog entry in context.
- Student attention trace: `A_student_l = attn_l(c_pre; θ_base + Δθ)`.

**Theorem.** There exists rank-8 `Δθ` on `(v_proj, o_proj)` such that:
1. `E[cos(A_teacher_l, A_student_l)] > 0.80` over held-out `(R, c_pre)` pairs (K1),
2. Refactor-quality auto-judge ≥ same-data token-space LoRA at matched rank (K2),
3. HumanEval pass@1 drop < 3pp (K3), non-refactor code-gen drop < 2pp (K4).

**Proof sketch.**
1. *Existence.* Same argument as `exp_hedgehog_behavior_adapter_politeness` — rank-8 LoRA on v/o has more DOF than a per-head MLP (Zhang 2024 sufficiency).
2. *Procedural transfer (K1 → K2).* The refactor operation `R` is a function of both `c_pre` and `π_R`. Per-layer cos-sim forces student to reproduce teacher's attention output *without seeing π_R*, meaning student must absorb the dependence on π_R into `Δθ`. If K1 PASS, the learned routing mimics "I-have-seen-π_R" behavior, and refactor-task accuracy follows (via Lipschitz argument on attention → residual stream → logits).
3. *K2 ≥ token-space LoRA baseline.* Token-space LoRA (trained on refactor pairs with next-token CE) matches *output tokens*. Cos-sim LoRA matches *attention routing*. The claim is cos-sim is at least as good; the rigorous bound is open. We report the head-to-head empirically.
4. *Specificity (K3, K4).* Rank-8 on `v_proj+o_proj` ≪ base rank. Perturbation magnitude is bounded. Per F#627 no collapse pattern on unrelated tasks.

## 4. Kill-criterion map

| KC | Measured quantity | Threshold | Type |
|---|---|---|---|
| K1 | mean per-layer cos on held-out refactor pairs | > 0.80 | structural proxy |
| K2 | refactor quality auto-judge vs same-data token-space LoRA baseline | ≥ baseline | target (pair K1) |
| K3 | HumanEval pass@1 drop vs base | < 3pp | target non-interference |
| K4 | non-refactor code-gen (gen-from-spec) drop vs base | < 2pp | target specificity |

Auto-judge rubric: 0-10, graded by (a) correctness (unit tests pass on post-refactor code), (b) refactor named correctly, (c) semantic equivalence.

## 5. Predicted measurements

- K1: cos ∈ [0.80, 0.88], mean ≈ 0.83 (lower than politeness because procedural content is more specific)
- K2: auto-judge Δ ∈ [0, +1.5] vs token-space LoRA (roughly matched)
- K3: HumanEval drop ≤ 2pp
- K4: non-refactor drop ≤ 1pp

If K2 fails (token-space LoRA wins), the finding is: **cos-sim distillation is better for behavior (style/politeness) than for procedural knowledge at this scale** — a useful axis-specificity finding.
