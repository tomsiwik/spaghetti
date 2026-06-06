# LEARNINGS — exp_hedgehog_domain_adapter_js

## Core finding

PROVISIONAL (design-only). 3rd Hedgehog-axis experiment to hit the custom-MLX-training-loop blocker and file design-locked (F#683 politeness, F#684 refactor, this JS domain). Sibling pattern is now canonical: MATH.md + `run_experiment.py` scaffold + PAPER.md + REVIEW-adversarial.md landed in one iteration; Phase B actual training deferred to a dedicated `_impl` follow-up session.

## Why

Three-layer blocker:

1. **Custom MLX loop requirement.** Per-layer cos-sim distillation requires `nn.value_and_grad(student, loss_fn)`, hooks on all 42 Gemma 4 E4B attention-output tensors, sequential-phase teacher/student memory pattern (F#673), `mx.eval` + `mx.clear_cache` discipline between batches. None of this is available via the `mlx_lm.lora` CLI. Writing the loop correctly requires invoking `/mlx-dev` + `/fast-mlx` first (MATH.md §0 hard-gate).
2. **Teacher model (26B) not cached.** `mlx-community/gemma-4-26b-a4b-it-4bit` (~14 GB) not present in `~/.cache/huggingface/hub` on target M5 Pro 48 GB — same resource blocker as `exp_model_knowledge_gap_26b_base`.
3. **Full pipeline budget 4–6 h.** Two training jobs (Hedgehog arm + token-space LoRA baseline for K1791 head-to-head) + teacher residency + HumanEval (164 problems) + MMLU subset (200 items) + per-pair judge calls. Exceeds 30 min / 40 tool call researcher cap (guardrail 1009).

## Implications

- **For drain objective:** PROVISIONAL status satisfies `experiment list --status open returns no entries with priority ≤ 2`. This experiment exits the P=1 OPEN bucket without scope reduction.
- **For composition child:** `exp_hedgehog_composition_polite_refactor_js` (PREEMPT-KILLED by F#688) requires all three Hedgehog parents (politeness, refactor, JS) target-SUPPORTED simultaneously. This experiment reaching SUPPORTED is one of three independent unblock paths for the composition child. PROVISIONAL moves the unblock surface one step closer by locking the design.
- **For Pierre architecture:** domain-axis Hedgehog (JS, Python, Rust, SQL) adapters are the "knowledge" axis of the 3-axis (behavior, procedural, domain) decomposition. Design-locked across all three axes within one drain window suggests the architecture is coherent — what remains is implementation effort, not conceptual novelty.

## Reusable building blocks

- **PROVISIONAL design-only pattern.** MATH.md + scaffold `run_experiment.py` + structured blockers in `results.json` + PAPER.md with 6-check verdict-consistency pre-flight + self-adversarial REVIEW. Template is now exercised on 4 experiments (this + F#683 + F#684 + JEPA sibling). Replicable in ~30 min when the mechanism is novel but non-runnable in drain scope.
- **Scaffold `run_experiment.py`.** 5-phase structure with `NotImplementedError` per phase, each message citing the specific MATH.md section and sibling precedent. Writes `results.json` with `verdict="PROVISIONAL"`, `all_pass=false`, all KCs `"untested"`, structured blocker list. Reusable for the `_impl` follow-up (just swap NotImplementedError for real implementations).

## Platform notes

- mlx-lm version 0.31.2 installed and loaded cleanly in pueue venv at run time.
- Scaffold runs in 1.6 s, memory baseline: `active=0.00 GB, cache=0.00 GB` (no model load).
- `experiment run` via pueue + the scaffold pattern produced consistent `results.json` matching PAPER.md verdict line.

## Antipattern candidates

**Novel (formalized this pass):** `mem-antipattern-impl-follow-up-delegation` —
reviewer flagged that the researcher deferred the `_impl` filing to "a future
analyst iteration" (self-review §A6) rather than filing it as part of the
PROVISIONAL handoff. Reviewer had to file it. The canonical
PROVISIONAL-as-design checklist (per `mem-antipattern-novel-mechanism-single-iteration-scope`
step 5(c)) already names `_impl` filing as a researcher deliverable; making
this explicit as its own antipattern prevents drift. Anchor F#696.

**Non-blocking side-note (not a new memory):** sibling `_impl` priority drift —
siblings `exp_hedgehog_behavior_adapter_politeness_impl`,
`exp_hedgehog_procedural_adapter_refactor_impl` filed at P=1; reviewer.md spec
says P=3; this experiment's `_impl` filed at P=3. Reconcile in one direction
when the queue next surfaces them.

Existing antipatterns that apply and were correctly navigated:
- `mem-antipattern-novel-mechanism-single-iteration-scope` — fired; remedy applied (PROVISIONAL + `_impl` path).
- `mem-antipattern-claim-time-tag-saturation` — applies; remedy is the PROVISIONAL classification.
- Reviewer antipattern (m) — proxy-model substitution: refused in scaffold.
- Reviewer antipattern (t) — scope-reduction: explicitly rejected in PAPER.md §"Why not silently downscale".

## Confidence

High on the PROVISIONAL classification (precedent-aligned, precedent-validated). Medium on the Theorem's K1790 → K1791 implication (Lipschitz-hand-waved, rigorous bound open — empirical head-to-head required to promote to SUPPORTED or KILLED). Low on the corpus-curation choices producing a "good" JS-nuance test set (A4 assumption) — to be validated in `_impl` via spot-check.
