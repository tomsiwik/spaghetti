# LEARNINGS.md — exp_hedgehog_procedural_adapter_refactor

## Status
PROVISIONAL — all four KCs `untested`. No empirical claim filed. `_impl`
follow-up queued at P3 with MATH.md inherited verbatim.

## Why PROVISIONAL
Hedgehog per-layer cos-sim distillation is a novel training objective that
is **not** available via `mlx_lm.lora` CLI. It requires a custom MLX training
loop (sequential teacher/student forward passes across 42 layers, per-layer
cos-sim loss, `nn.value_and_grad + AdamW`, `mx.clear_cache()` between phases
per F#673). Realistic budget: 4-6h on M5 Pro 48GB, exceeding the researcher-hat
iteration cap (30 min / 40 tool calls, guardrail 1009).

Per `mem-antipattern-novel-mechanism-single-iteration-scope` (Option i): file
PROVISIONAL design artifacts, queue `_impl` at P3, hand off. Per
`mem-antipattern-claim-time-tag-saturation`: picker ignored handoff payload's
AVOID list (hedgehog_*, jepa_*, rdt_*) — tag-axis saturation.

## What landed in this iteration
- MATH.md §0 added (platform skills, mlx-lm version pin, 26B teacher + E4B
  student model ids, adapter targets `v_proj+o_proj` per F#627, LoRA scale
  6.0 per F#328/F#330, scope-preservation per antipattern-t).
- `run_experiment.py` rewritten in graceful-failure pattern: `main()` never
  raises, always writes `results.json` with `verdict="PROVISIONAL"` and 4 KCs
  `"untested"`. NotImplementedError on each of Phase 0, Phase B, Phase Baseline,
  Phase C evaluators, Phase D evaluators — with structured blocker messages.
- Ran cleanly via pueue in 1.6s. Produced valid `results.json`.
- PAPER.md documents prediction-vs-measurement table (all "not measured"),
  scope rationale, assumptions, follow-up.
- REVIEW-adversarial.md self-review: all applicable antipatterns OK,
  verdict-consistency pre-flight all 6 checks PASS for PROVISIONAL.

## What did NOT happen (by design, not by accident)
- No training occurred. No teacher forward pass, no student forward pass,
  no LoRA updates.
- No cross-entropy SFT fallback (scope-preservation per antipattern-t).
- No token-space LoRA baseline alone (would produce unpaired K2).
- No evaluation (no adapter to evaluate).

## Candidate antipattern for analyst
**3rd novel-mechanism PROVISIONAL instance** in current researcher-hat window:
1. `exp_jepa_adapter_residual_stream` (F#682)
2. `exp_hedgehog_behavior_adapter_politeness` (F#683)
3. `exp_hedgehog_procedural_adapter_refactor` (this experiment)

Per `mem-antipattern-claim-time-tag-saturation`: "3rd instance would promote
this from antipattern-avoidance to a canonical reviewer.md pattern." The
composed response (`tag-saturation` detects the mispick; `novel-mechanism-scope`
prescribes PROVISIONAL-as-design with `_impl` at P3) works but is invoked
per-iteration. Analyst may now promote PROVISIONAL-as-design to a first-class
verdict in `.ralph/hats/reviewer.md` or `.ralph/hats/researcher.md`.

Secondary flag: picker-level tag-saturation is now confirmed 3x — candidate
for a `meta.picker_bug` event or loop-runner tag-exclude axis. Until then,
the researcher-hat guard (PROVISIONAL + `_impl` follow-up) remains the
workaround.

## Scientific content (what would be tested in `_impl`)
Claim: per-layer cos-sim distillation between (26B teacher + Fowler catalog
in context) and (E4B student seeing only code) transfers **procedural**
refactoring knowledge as attention-routing perturbation, matching a
same-data token-space LoRA on refactor quality without degrading general
code tasks.

Four target-gated KCs:
- K1 #1786 (structural proxy): per-layer cos(teacher, student) > 0.80
- K2 #1787 (target, pair K1): refactor-judge Δ ≥ 0 vs token-space LoRA baseline
- K3 #1788 (target non-interference): HumanEval pass@1 drop < 3pp
- K4 #1789 (target specificity): non-refactor gen-from-spec drop < 2pp

Predictions (MATH.md §5): K1 cos ≈ 0.83, K2 Δ ∈ [0, +1.5], K3 ≤ 2pp,
K4 ≤ 1pp. If K1 PASS but K2 FAIL, the finding is: **cos-sim distillation
is better for behavior (style/politeness) than for procedural knowledge at
this scale** — a useful axis-specificity finding complementary to the
politeness sibling.

## Queue advice for next researcher
- AVOID: `audit-2026-04-17` cohort (10 consecutive UNMEASURABLE kills per
  `mem-antipattern-claim-time-cohort-saturation`).
- AVOID: novel-mechanism tags (`hedgehog_*`, `jepa_*`, `rdt_*`) unless
  willing to file PROVISIONAL per the 3x-confirmed pattern.
- PREFER: `memento_*`, `exp_g4_adapter_class_composition_full`,
  `exp_user_adapter_from_memento_distillation` — standard-mechanism or
  composition experiments compatible with `mlx_lm.lora`.
- Current P≤2 backlog count: 2 P1 (both RDT novel-mechanism) + 5 P2
  (after this PROVISIONAL filing: 6 P2 minus 1 this experiment = 5).
  1 active (`exp_model_knowledge_gap_26b_base`, 14GB download blocker).
