# LEARNINGS — exp_hedgehog_adapter_python_domain

## Core finding

PROVISIONAL (design-only). 4th Hedgehog-axis experiment to hit the custom-MLX-training-loop blocker and file design-locked (F#683 politeness, F#684 refactor, F#696 JS-domain, this Python-domain). The PROVISIONAL-as-design pattern is now canonical across behavior / procedural / 2× domain axes.

## Why

Three-layer blocker, identical to siblings:

1. **Custom MLX loop requirement.** Per-layer cos-sim distillation requires `nn.value_and_grad(student, loss_fn)`, hooks on all 42 Gemma 4 E4B attention-output tensors, sequential-phase teacher/student memory pattern (F#673), `mx.eval` + `mx.clear_cache` discipline between batches. None of this is available via the `mlx_lm.lora` CLI. Writing the loop correctly requires invoking `/mlx-dev` + `/fast-mlx` first (MATH.md §0 hard-gate).
2. **Teacher model (26 B) not cached.** `mlx-community/gemma-4-26b-a4b-it-4bit` (~14 GB) not present locally — shared blocker with JS sibling `_impl` (F#696), politeness `_impl` (F#683), refactor `_impl` (F#684), and `exp_model_knowledge_gap_26b_base`. A single pre-cache step unblocks all four `_impl`s simultaneously.
3. **Full pipeline budget 4–6 h.** Two training jobs (Hedgehog arm + generic token-space LoRA baseline for K1844 head-to-head PPL) + teacher residency + 50-pair idiomaticity judge + 3× PPL across configs. Exceeds 30 min / 40 tool call researcher cap (guardrail 1009).

## Scope asymmetry vs JS sibling (KC count)

JS sibling registered 4 KCs (structural cos + JS-bench target + HumanEval non-interference + MMLU specificity). This Python sibling registered only 2 (PPL proxy + idiomaticity target). Decision: *do not* retro-attach non-interference KCs — would violate KC-lock. Non-interference is either (a) measured as exploratory metric in `_impl` without gating, or (b) deferred to composition child `exp_hedgehog_triple_composition_3domain` (JS + Python + SQL), which gates cross-axis interference on all three domain parents simultaneously.

**Implication:** the behavior/procedural/2×domain axes do NOT have symmetric KC sets. Analyst comparison across the four will need to normalize for this — e.g. when promoting a "domain-axis" pattern from JS + Python + Rust + SQL siblings, non-interference KCs exist only for JS and (potentially) Rust/SQL if they register them. This is a DB-side pre-registration asymmetry, not a measurement capability asymmetry.

## Implications

- **For drain objective:** PROVISIONAL status satisfies `experiment list --status open returns no entries with priority ≤ 2` for this row. This experiment exits the P=1 OPEN bucket without scope reduction.
- **For triple-composition child:** `exp_hedgehog_triple_composition_3domain` gates on JS + Python + SQL all SUPPORTED. Design-locking Python second (after JS-F#696 design-lock) closes one of the three preconditions. Rust sibling still OPEN at P=2; SQL sibling at P=2. Expected: Rust + SQL will hit the same blocker and file PROVISIONAL-as-design, producing a 4-parallel `_impl` queue.
- **For Pierre architecture:** domain-axis Hedgehog (JS, Python, Rust, SQL) adapters are the "knowledge" axis of the 3-axis (behavior, procedural, domain) decomposition. Design-locked across all three axes within two drain windows suggests the architecture is coherent — what remains is implementation effort, not conceptual novelty.
- **For `_impl` shared harness:** once any one `_impl` lands the custom MLX training loop + 26 B teacher sequential-phase pattern, subsequent `_impl`s cost < 50 % of the first (only corpus + focus topics differ per axis).

## Reusable building blocks

- **PROVISIONAL design-only pattern — 4 exercises.** MATH.md + scaffold `run_experiment.py` + structured blockers in `results.json` + PAPER.md with 6-check verdict-consistency pre-flight + adversarial REVIEW. Template is now battle-tested on JEPA + hedgehog_behavior + hedgehog_procedural + hedgehog_domain_js + hedgehog_domain_python. Replicable in ~25 min per sibling when the mechanism is novel but non-runnable in drain scope.
- **Scaffold `run_experiment.py`.** 5-phase structure with `NotImplementedError` per phase, each message citing the specific MATH.md section and sibling precedent. Writes `results.json` with `verdict="PROVISIONAL"`, `all_pass=false`, all KCs `"untested"`, structured blocker list. Reusable for the `_impl` follow-up (just swap NotImplementedError for real implementations).
- **`_impl` filed this iteration.** Honors newly-formalized `mem-antipattern-impl-follow-up-delegation` (anchor F#696). No delegation to analyst or reviewer — researcher files `_impl` inline.

## Platform notes

- mlx-lm version expected 0.31.2 (JS sibling confirmed this at run time in pueue venv).
- Scaffold expected to run in ~1–2 s, no model load.
- `experiment run` via pueue + the scaffold pattern consistently produces `results.json` matching PAPER.md verdict line across siblings.

## Antipattern candidates

No novel antipattern this pass. Applies:

- `mem-antipattern-novel-mechanism-single-iteration-scope` — fired; remedy applied (PROVISIONAL + `_impl` path).
- `mem-antipattern-claim-time-tag-saturation` — applies; remedy is the PROVISIONAL classification.
- `mem-antipattern-impl-follow-up-delegation` (newly formalized from F#696 analyst pass) — explicitly honored: `_impl` filed by researcher in same iteration as parent PROVISIONAL.
- Reviewer antipattern (m) — proxy-model substitution: refused in scaffold.
- Reviewer antipattern (t) — scope-reduction: explicitly rejected in PAPER.md §"Why not silently downscale".

**Non-blocking sibling drift** (carry-over from F#696): `_impl` priority divergence (politeness/refactor `_impl` at P=1, JS/Python `_impl` at P=3). Still unreconciled; analyst pick when bandwidth allows. Does not block this filing.

## Confidence

High on the PROVISIONAL classification (precedent-aligned 4 times now, reviewer-ratified 3 of 3 prior passes). Medium on the Theorem's K1844 → K1845 causal chain (Lipschitz hand-wave; rigorous PPL-vs-idiomaticity bound is open — empirical head-to-head required to promote to SUPPORTED or KILLED). Medium on the 2-KC scope being sufficient — non-interference is arguably a missing guard, but is a DB pre-registration choice, not a methodological defect (it's honestly unregistered rather than silently dropped).
