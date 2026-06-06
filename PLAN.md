# Plan

> Central working document. Framework principles (stable) + current research focus (iterable). Update here, not in scattered vision files.

---

## Part 1 — Framework (stable)

The experiment process — lifecycle, proof-first discipline, target-gated kill rule (F#666), verdict consistency, the 3-hat loop, antipatterns, SIGREG, and forbidden classes — is now consolidated in **`experiments/GUIDE.md`** (the single canonical "how agents run experiments" doc). Read it there; this section is intentionally a pointer to avoid duplication.

One-line: a structured experiment framework — hypotheses are claimed, run, reviewed, and recorded in a queryable DB; proof-first (`MATH.md` before code); every claim needs a behavioral target-metric KC.

---

## Part 2 — Current focus: Pierre

> Iterable section. Update as research progresses.

### One-line
A coding agent where every conversation trains a composable domain expert (adapter), and shared adapters make the base smarter over time.

### Platform
- **Target hardware**: Apple M5 Pro 48GB. **MLX only — no CUDA, no RunPod, no torch on GPU.** The machine is unified-memory Metal-optimized; MLX is the native path and produces dramatically better code and runtime behaviour on this hardware.
- **Base model**: `mlx-community/gemma-4-e4b-it-4bit` (dev) / `mlx-community/gemma-4-26b-a4b-it-4bit` (prod). BitNet-2B-4T is retained only as a tok/s speed-ceiling reference (165.6 tok/s), not as product base.
- **Adapter approach**: Standard LoRA r=6 via `mlx_lm.lora`. Current trained adapters (math, code, medical) are on `q_proj` per `exp_p1_t2_single_domain_training`. Future adapters should target `v_proj+o_proj` per F#627. Grassmannian A-matrices and PoLAR are research goals, not current reality.
- **Trained adapter weights**: `adapters/{math,python,medical}/adapters.safetensors` (copied from `experiments/models/exp_p1_t2_single_domain_training/adapters/`). Finance and legal are config-only stubs (no trained weights).

### Adapter vocabulary (glossary)
Pierre uses three distinct adapter kinds. They share LoRA shape but train and compose differently. Use the matching tag when filing experiments.

| Kind | Indexes over | Purpose | Trains on | Tag |
|---|---|---|---|---|
| **Domain adapter** | semantic specialty (math, code, medical, legal, finance) | amplify domain-specific factual/behavior patterns already in base | per-domain corpus | (default `p1`) |
| **Method adapter** | procedural skill (decompose subgoals, diagnose differentially, plan-then-solve) | amplify how-to, not what-to — generalizes across domains | cross-domain traces of the same method | `method-adapter` |
| **Loop adapter** | depth iteration t in a recurrent-depth block | differentiate iterations of weight-shared looped transformer (Bae 2024) | reasoning-heavy corpus; only loop-LoRA + LTI injection train | `loop-adapter` / `rdt` |

Pierre-internal shorthand:
- **Room Model** = `W_combined = Σ ΔW_i` pre-summed composition matrix. External literature term: LoRA merging / task arithmetic (Ilharco et al. 2022, arxiv:2212.04089).
- **Grassmannian A** = orthogonal A matrix constructed via partitioned QR; structural orthogonality verified at Gemma 4 native dims (Finding #562).

### Required skills (MUST invoke before writing any MLX code)
The hat loop has specialised skills that enforce idiomatic MLX and catch common mistakes. Invoke them **before coding**, not after.

| Skill | When to use |
|---|---|
| `/mlx-dev` | Any MLX array/nn/training/inference code. Enforces `mx.eval` discipline, lazy evaluation, proper module patterns, memory cleanup. |
| `/fast-mlx` | Performance-sensitive paths (training loops, inference, compile). Enforces `mx.compile`, fast ops, type promotion, bandwidth-aware kernels. |

Skipping these skills is the single biggest cause of broken MLX code in past experiments (wrong `nn.value_and_grad` mutation patterns, missing `mx.eval`, wrong BCE-with-logits handling — all appear in the audit findings). No exception: even quick smoke tests must go through `/mlx-dev`.

### Code conventions (MLX-specific)
- Use the **phased execution pattern** for memory safety: each compute phase in its own function, explicit cleanup between phases (see `/mlx-dev` skill for reference).
- `mx.clear_cache()` after loading large weights; `del` + `gc.collect()` between phases.
- No PyTorch/CUDA fallbacks. If a library requires CUDA, it's not usable here — find an MLX-native alternative or implement the primitive.
- Cite the `mlx-lm` version used in `MATH.md` — API changes between 0.21 and 0.31 have broken prior experiments.

### Working hypothesis
- Frozen base + lightweight adapters can replicate or exceed monolithic fine-tuning when adapters are (a) domain-specialized, (b) structurally orthogonal, (c) routed cleanly.
- Composition must happen in continuous space; ternary/quantized composition requires explicit handling (BitNet foundations).
- Thinking mode (Gemma 4 `<|channel>thought...`) must be preserved during training and eval, or reasoning degrades.

### Deep reference
The verified, DB-reconciled state of truth is **`STATUS.md`** at the repo root (the 2026-06-03 checkpoint). Read it before repeating any headline claim — the old in-repo vision docs drifted a thesis ahead of the data and are frozen in `docs/archive/2026-06-03-superseded/`.

Do **not** fork a new `VISION_*.md`. When direction shifts, edit Part 2 here and append a checkpoint entry to `STATUS.md §7`.

### Current phase (as of 2026-06-03 — Checkpoint 0)
Research checkpoint completed: 937 experiments / 845 findings reconciled against the product docs (see `STATUS.md`). The strategy-transfer / orthogonal-composition thesis is **refuted** by the May-2026 frontier (F#827/837/844/822/823). Verified ceiling: K=7 static Fisher-Rao ≈64–68% avg (+2–4pp over base); solo adapters lift on-domain (+22–62pp) but interfere off-domain (−12..−14pp); M2P/MEMENTO/Hedgehog are dead or never-run. Earlier systemic issues (v1 composition bug, tautological routing v3–v6, LORA_SCALE=20 inflation, thinking-mode truncation) are confirmed and now superseded by the reconciled status.

**Next gate:** pick one Road from `STATUS.md §5` and run one VERIFY→INTEGRATE→MEASURE cycle (recommended: Road 1, single-domain hot-swap MVP — no composition required).

### Active workstreams
- **P11** — reasoning training recipe (s1K, LIMO, GRPO, ThinkPO, Plan-and-Solve).
- **Routing** — TF-IDF+embedding logistic; N=25 at ~89% top-1 accuracy.
- **Thinking preservation** — MCQ adapters must train with `enable_thinking=True`.

### Near-term goals
- Match or beat Gemma-4 MMLU-Pro baseline (62.1% with thinking) using an adapter that doesn't suppress thinking.
- Verify Grassmannian orthogonality claim on real Gemma 4 runs (not Qwen proxy).
- Close composition-bug recovery: identify every headline number derived through the buggy path; rerun or flag.

### Pierre code progression

**Current stable:** `pierre/pierre.py` (265 loc) — runtime composition pipeline. `pierre/bench.py` — benchmarks. `pierre/math/` — predict + theoretical analysis.

**Version protocol** — each new Pierre version is two artifacts:
1. A snapshot under `pierre/archive/vN/` (frozen code).
2. A validation experiment `exp_pierre_vN_*` in the experiment DB (MATH.md + run_experiment.py + PAPER.md).

Promote a version into `pierre/pierre.py` only after its validation experiment reaches `status=supported` AND passes the verdict-consistency checklist (PLAN.md §1).

**Changelog:**

| Version | Hypothesis | Verdict | Experiment | Notes |
|---|---|---|---|---|
| v1 | original pre-merge composition | supported (retroactively flagged) | — | **composition bug**: code summed `lora_A`/`lora_B` safetensors independently → `(ΣB)(ΣA)` cross-product. Fix required in v8. |
| v3 | SFT adapters + BitLinear side-path | supported | `exp_pierre_v3_sft_n5`, `exp_pierre_v3_n24_scaling` | Used tautological routing `route(val[d][0])`. Headline "0% degradation" is an artifact. |
| v4 | ternary premerge (merge LoRA into BitLinear) | killed | `exp_pierre_v4_ternary_premerge` | Ternary has no room for merged deltas (BitNet foundations). |
| v5 | fully ternary LoRA (Grassmannian A + STE B) | supported | `exp_pierre_v5_ternary_lora` | Same tautological routing as v3. |
| v5.1 | LoTA-QAF lossless ternary merge | killed | `exp_pierre_v51_lota_merge` | |
| v5.2 | Bankai-inspired ternary row flips (greedy search) | killed | `exp_pierre_v52_bankai_flip` | `revert_row_flip` not actually reversible after clipping; caught pre-run. |
| v5.3 | lazy bf16 LoRA side-path | killed | `exp_pierre_v53_lazy_sidepath` | |
| v5.4 | `mx.quantized_matmul` 2-bit lazy side-path | killed | `exp_pierre_v54_quantized_matmul` | |
| v6 | precomputed concatenated deltas (attention-factored) | supported | `exp_pierre_v6_precomputed_concat` | Same tautological routing. |
| v6.2 | hybrid precomputed attention + factored MLP | killed | `exp_pierre_v62_hybrid` | |
| v7 | keyframe adapter (deterministic verifier) | killed | `exp_pierre_v7_keyframe_poc` | `phase_composition` accepts `verifier` arg but never uses it (audit finding). |
| v7.1 | keyframe with last-token hidden state | killed | `exp_pierre_v71_keyframe_lasttoken` | Same unused-verifier bug as v7. |

**Next version plan (v8 — working draft):**

Goals driven by audit recovery:
1. **Fix composition math** — `Σ B_i @ A_i` correctly; drop any code path that sums A and B tensors independently (`mem-antipattern-001`).
2. **Per-sample routing** — replace `route(val[d][0])` with per-sample routing; headline PPL must not equal single-adapter PPL by construction (`mem-antipattern-002`).
3. **Drop LORA_SCALE=20** — default to safe scale (≤ 8); if a claim requires higher scale, it's a scale-specific claim, not a general property (`mem-antipattern-003`).
4. **Thinking preservation** — training and eval both use `enable_thinking=True`; strip regex matches Gemma 4's native `<|channel>thought` format (`mem-antipattern-008`).

Validation protocol:
- Snapshot current `pierre/pierre.py` → `pierre/archive/v7/` if not already there (v7 exists, confirm).
- Write `exp_pierre_v8_*` experiment with MATH.md pre-registering KCs: (i) `composed_ppl != single_ppl` across > 1 sample; (ii) per-sample router accuracy ≥ 85% at N=5; (iii) MMLU-Pro with thinking ≥ 62.1%; (iv) no antipattern memory triggers in review.
- Do not promote v8 → `pierre/pierre.py` until `exp_pierre_v8_*` is `supported` AND the audit-flagged v3/v5/v6 numbers have been reproduced cleanly or retracted.

**Where state lives (reference):**
- Per-experiment: experiment DB (Turso) via `experiment` CLI. This is authoritative.
- Short-term ralph loop: `.ralph/current_direction.md`.
- Roadmap + version plan: this file (PLAN.md Part 2).
- Antipatterns: `.ralph/agent/memories.md` (`type: fix`), auto-injected.

---

## Iteration discipline

- **This document is the central piece.** When research direction shifts, edit Part 2 here — don't fork into a new VISION.md.
- Part 1 changes only when a framework principle has been falsified or extended (rare). Treat it as settled unless a finding forces an update.
- Part 2 changes freely. Keep it short; push deep details to parent docs or `experiments/models/<exp>/PAPER.md`.
- New antipatterns → memory entries (`.ralph/agent/memories.md`, `type: fix`), not here.
