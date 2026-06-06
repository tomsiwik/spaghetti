# MATH.md — exp_followup_sequential_activation_compose_real

## Background & Audit Motivation

`exp_p3_b4_sequential_activation_compose` (parent) was titled "sequential
activation composition" but implemented pure additive composition
(`ΔW_eff = ΔW_D + ΔW_P`). The audit (killed_12.md) flagged this as
a title/implementation mismatch. This follow-up implements a genuine
sequential pipeline and measures it against the additive baseline of
the parent (style=24.0%, math=15.0%, Finding #464/parent PAPER.md §3).

## Impossibility Structure: Weight-Space Sequential on q_proj

**Theorem 1 (Architectural infeasibility at weight-space / per-layer
activation level).** For two LoRA adapters with `A∈ℝ^{d_in×r}`,
`B∈ℝ^{r×d_out}` trained on the SAME projection `p`, the second-order
cross-term `ΔW_P @ ΔW_D` required by
`h_seq = (I + ΔW_P)(W + ΔW_D)x = Wx + ΔW_D x + ΔW_P W x + ΔW_P ΔW_D x`
requires `d_out(ΔW_D) = d_in(ΔW_P)`.

For both adapters in this study, `p = self_attn.q_proj` with
`d_in = hidden_size = 2560` and `d_out = n_kv_heads × head_dim = 2048`
(Gemma 4 E4B). Because `2048 ≠ 2560`, the cross-term is
non-computable. Consequently:

* Per-layer activation-space sequential is **undefined**
  (`ΔW_P` cannot consume `ΔW_D`'s output).
* Weight-space sequential via `(I+ΔW_P)(W+ΔW_D)` is **undefined**
  for the same reason.

**QED.** (restates the parent MATH.md §Background observation)

This theorem is *verified* in Phase 0 of `run_experiment.py` by
asserting the actual loaded adapter tensor shapes.

## Theorem 2 — Model-Level Sequential Generation is Feasible

**Statement.** Let `f_θ: 𝒱* → 𝒱*` denote the generation function of the
causal LM parameterized by `θ` (base weights + any active adapters).
For adapter sets `θ_D` (domain) and `θ_P` (personal), the composed
function

```
h(x) = f_{θ_P}( f_{θ_D}( x ) )
```

is well-typed: the output of `f_{θ_D}` lies in `𝒱*` (token strings),
which is exactly the domain of `f_{θ_P}`. No dimensional constraint
couples `θ_D` and `θ_P` because the intermediate representation is a
token sequence, not a weight-space or activation-space object.

**QED** (typing argument).

This is the unique interpretation of
`h = personal_forward(domain_forward(base_forward(x)))` in the title
that is both (a) architecturally sound and (b) implementable on the
existing q_proj-only adapters. Both base and domain "forward" map
`x → 𝒱*`; we let `base_forward(x) = x` (the user prompt is already
valid natural-language text, i.e., the base model forward for the
identity-adapter case is the prompt as given to the next stage).

## Kill Criteria (locked pre-run)

KC#1563 (DB): "Sequential activation composition matches or beats
additive composition on P3.B5 domain tasks." Operationalized as two
sub-criteria, locked before any run:

| KC | Comparator | Threshold | Source |
|---|---|---|---|
| K1563a (style) | `seq_style_rate >= 24.0%` | additive baseline (parent PAPER.md §1) | Finding #464 |
| K1563b (math)  | `seq_mcq_acc   >= 15.0%` | additive baseline (parent PAPER.md §1) | Finding #464 |

Both must PASS for K1563 PASS. Either failure → KILLED.

**Smoke-mode caveat.** N_smoke=5 per task. Per PLAN.md §1
verdict-consistency rule 4, `is_smoke=true` runs complete as
`provisional`, never `supported`/`killed`, regardless of KC outcome.

## Predictions (quantitative, pre-registered)

| Quantity | Prediction | Rationale |
|---|---|---|
| Domain-only style rate | ≤ 10% | Math adapter trained on math data, no style signal |
| Personal-only MCQ | ≈ 5–15% | Personal adapter is generic style, only 16 layers; weak math |
| Pipeline style | ≥ 24% | Personal at stage 2 restores style lost at stage 1 |
| Pipeline MCQ | ≥ 15% | Domain at stage 1 provides math, stage 2 preserves |
| Shape check (Thm 1) | `B_D ∈ ℝ^{r_D×2048}`, `A_P ∈ ℝ^{2560×r_P}` | Per adapter configs |

The core prediction is that the pipeline beats both stage-isolated
baselines because each adapter contributes at its own stage.

## Assumptions (logged per rule 1007)

1. Base Gemma-4-E4B generates meaningful chat responses at
   `max_tokens=128` for both style questions and MCQ prompts.
2. The personal adapter can consume an assistant-style completion as
   context and restyle it (tested implicitly — if this fails the
   pipeline will produce long regurgitations and K1563a will FAIL,
   which is an acceptable falsification).
3. MCQ accuracy measured on `cais/mmlu/abstract_algebra` matching
   parent's protocol.

## Protocol

1. **Phase 0 — shape verification (Thm 1).** Load both adapter
   `safetensors`; assert `lb_D.shape[1] = 2048` and
   `la_P.shape[0] = 2560` and `2048 ≠ 2560`.
2. **Phase 1 — stage-isolated baselines** (domain-only style,
   personal-only MCQ). Quantifies how much each adapter *alone*
   contributes to the opposite task.
3. **Phase 2 — model-level sequential pipeline.** For each question Q:
   * `R_D ← generate(adapter=domain, prompt=format(Q))` (≤128 tok)
   * `R_P ← generate(adapter=personal,
     prompt=format(Q) + "\nInitial reply: " + R_D + "\nRefined reply:")`
     (≤128 tok)
   * Evaluate `R_P` for style (contains `PREFERENCE_MARKER`) and MCQ
     accuracy (letter extraction matching parent protocol).
4. **Phase 3 — KC evaluation.** Compare `(pipeline_style, pipeline_mcq)`
   against additive baselines `(24.0%, 15.0%)`.

## Connection to Prior Literature

* **Parent (Finding #464, exp_p3_b4):** pure additive gives
  style=24%, math=15%. This experiment's baselines to beat.
* **Cascade generation** — the stage-1/stage-2 protocol follows the
  common "refinement prompt" pattern used in tool-use and chain-of-
  thought literature (Wei et al., arXiv 2201.11903; Madaan et al.
  "Self-Refine", arXiv 2303.17651). No novelty claimed for the
  protocol itself; novelty is applying distinct LoRA adapters at
  each stage.
* **LoRA (Hu et al., 2110.04367)** — composition is defined at the
  weight level; this experiment operates at the generation-function
  level as an alternative composition mode.

## Tools / Versions

* `mlx-lm` current project pin (see repo `pyproject.toml`).
* Model: `mlx-community/gemma-4-e4b-it-4bit`.
* All adapter configs inherited unchanged from
  `exp_p1_t2_single_domain_training/adapters/math` and
  `exp_p1_t5_user_local_training/personal_adapter` (scale=6.0 math,
  scale=4.0 personal, rank 6/4, 42-layer / layers-26-41 q_proj).
