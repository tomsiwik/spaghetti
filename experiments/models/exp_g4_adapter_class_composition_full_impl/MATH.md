# exp_g4_adapter_class_composition_full_impl — MATH (PROVISIONAL, marginal-Phase-A executable slice)

## §0 Platform skills and scope preservation (pre-registered, inherited from parent)

**Platform skills invoked for design:** `/mlx-dev`, `/fast-mlx` per PLAN.md Part 2 + parent §0. This `_impl` follow-up to `exp_g4_adapter_class_composition_full` (parent F#686/PROVISIONAL design-only) inherits MATH.md verbatim from the parent and adds an executable **Phase A topology-inspect slice** within the single-iteration researcher budget (30 min / 40 tool calls, guardrail 1009). The full empirical pipeline (B1-B5 below) realistically requires 8-15h — Phase B-E execution is deferred to a follow-up iteration on the same directory at P=3.

**Pinned versions.** `mlx-lm >= 0.22` (DoRA via `--fine-tune-type dora`); MoLoRA still has no `mlx_lm` turn-key — custom `micro/utils/molora.py` is a Phase B deliverable.

**Base model (exact HF repo id):** `mlx-community/gemma-4-e4b-it-4bit` (Gemma 4 E4B 4-bit, per PLAN.md Part 2). Adapter targets: `v_proj + o_proj` per F#627. LoRA scale: default canonical `scale = 6.0`. Never hardcoded 20 per F#328/#330.

**Scope-preservation antipattern-t forbid list (F1-F5, binding for any future Phase B execution on this dir):** verbatim from parent §0.
- (F1) No silent DoRA/MoLoRA → LoRA swap.
- (F2) No silent N=5 → N=3 reduction.
- (F3) No silent v_proj+o_proj → q_proj substitution.
- (F4) No MMLU eval at n<1000 without explicit 95% CI.
- (F5) OOM fix-order: (i) batch+grad-accum, (ii) gradient_checkpointing, (iii) max_length 2048 — never base/adapter-class/benchmark swap.

## §1 Scope (this iteration: marginal Phase A only)

**Inherited claim** (verbatim from parent): on Gemma 4 E4B 4-bit, trained adapters in Class A (LoRA r=6) achieve MMLU-Pro N=5 composed accuracy ≥3pp above max(Class B.1 DoRA, Class B.2 MoLoRA), with 95% CI lower bound > 0.

**This iteration's executable slice (Phase A):**
1. Load Gemma 4 E4B 4-bit via `mlx_lm.load()`; record exact HF repo id.
2. Enumerate `v_proj + o_proj` layer presence and count (per F#627 LoRA target choice).
3. Inspect `mlx-lm` version + DoRA support (look for `dora_layers` symbol or `--fine-tune-type dora` CLI flag).
4. Sketch MoLoRA module signature (file-level pseudocode in PAPER.md §B-deferred — no MLX ops in this iteration).

Phase A is **plumbing-only**, no training, no eval. K1-K4 remain `untested`. The slice produces a topology readout (the "K0 Phase A" line in PAPER.md §5) that future Phase B execution will consume.

**Why this slice is honest signal:**
- Confirms the parent design's F#627 assumption (`v_proj + o_proj` exist on Gemma 4 E4B 4-bit) — without this readout, any future Phase B run risks an F#669-style cascade (training script targets a non-existent module).
- Confirms `mlx-lm` DoRA availability — the parent's §0 assumption ("verify before writing custom DoRA loop") is closed empirically here.
- Aligns with `exp_jepa_adapter_residual_stream_impl` and `exp_memento_gemma4_replication_impl` precedent (Phase A executable slice → PROVISIONAL routing → Phase B-E deferred to same-dir follow-up).

## §2 Theorem (inherited from parent §2; not re-proven this iteration)

See parent MATH.md §2 verbatim. The K1-K4 KCs in §3 below are inherited unchanged; this iteration's Phase A slice does not test them.

## §3 Pre-registered kill criteria (target-gated per F#666; verbatim from parent + Phase A addendum)

Inherited from parent (verbatim):

**K1 (structural):** ≥13/15 class-domain trainings converge (final_loss < 1.1 × min(train_loss) AND < 0.7 × initial_loss). [#1833 in DB]

**K2 (target behavioral):** `acc_A − max(acc_{B.1}, acc_{B.2}) ≥ 0.03` AND 95% CI lower bound (paired bootstrap, 10000 resamples) > 0 on MMLU-Pro n=1000 at N=5 composition. [#1834 in DB]

**K3 (proxy confirmation — geometric):** `median(dev_D) > 10⁻³` on trained DoRA at composition time. [#1835 in DB]

**K4 (rank ablation):** sign of K2 stable at r=6 and r=8. [#1836 in DB]

**This iteration's Phase A K0 readout (NOT a KC, NOT pre-registered, NOT verdict-bearing):**
- A1: Gemma 4 E4B 4-bit base loads via `mlx_lm.load()` without error.
- A2: `v_proj + o_proj` modules exist in ≥1 layer (F#627 target validated).
- A3: `mlx-lm` version pinned ≥ 0.22 + DoRA mode confirmed available.

These A1-A3 readouts are reported in PAPER.md §5 alongside the unchanged K1-K4=untested rows.

## §4 Assumptions (inherited from parent §4, unchanged)

See parent MATH.md §4. No assumption changes this iteration.

## §5 Prediction-vs-measurement table (Phase A only this iter; KCs unchanged)

| Item | Prediction | Measured (Phase A only) | Status |
|---|---|---|---|
| K1 structural | ≥13/15 trainings converge | (Phase B deferred) | untested |
| K2 target | `acc_A − max(acc_{B.j}) ≥ 0.03`, CI LB > 0 | (Phase C deferred) | untested |
| K3 proxy | `median(dev_D) > 10⁻³` | (Phase C deferred) | untested |
| K4 ablation | sign of K2 stable at r=8 | (Phase B+C deferred) | untested |
| A1 base loads | Gemma 4 E4B 4-bit loads via mlx_lm.load() | (filled by run_experiment.py) | (filled at runtime) |
| A2 v_proj+o_proj | ≥1 layer with both modules | (filled by run_experiment.py) | (filled at runtime) |
| A3 DoRA available | mlx-lm ≥ 0.22 + dora support | (filled by run_experiment.py) | (filled at runtime) |

## §6 Deliverables (inherited from parent §6; Phase A produces only readouts)

Phase A (this iter): topology readout in `results.json` (`phase_a_readout` field). All other deliverables (B1 MoLoRA module, B2 15 trainings, B3 5-corpus curation, B4 N=5 harness, B5 bootstrap) deferred to follow-up at P=3 on the same dir.

## §7 Scope fence (verbatim from parent §7, binding)

Phase A topology readout does not claim:
- Any geometric mechanism transfer from F#82 micro-d to Gemma 4.
- Any DoRA/MoLoRA training feasibility (only that the CLI/library accepts the modes).
- Any MMLU-Pro accuracy.

Phase B execution (when claimed at P=3) will close these gaps under the same MATH.md.

## §8 Antipattern scan (12 items, pre-emptive)

| Antipattern | Status |
|---|---|
| `mem-antipattern-novel-mechanism-single-iteration-scope` | applied option (ii) — Phase A executable slice + Phase B-E deferred (same precedent as memento_replication_impl iter ~104, jepa_adapter_residual_stream_impl) |
| `mem-antipattern-proxy-kc-mislabeled-target` | K2 measures MMLU-Pro accuracy (behavioral); K3 measures proxy. Inherited verbatim from parent. |
| `mem-antipattern-preempt-child-parent-target-unverified` | N/A: parent design-only PROVISIONAL is *intentional* deferral; this child is the designated implementation continuation. |
| `mem-antipattern-thinking-mode-truncates-judge-budget` | N/A this iter (no eval/judge). Phase B-E will need `enable_thinking=False` for MMLU eval per F#793/F#795/F#797/F#798 cross-validated mitigation pattern. |
| `mem-antipattern-researcher-prefiles-finding-before-review` | HONORED — no finding pre-fill; reviewer iter ~108 will file canonical F. (4th consecutive observance post-mitigation.) |
| `mem-antipattern-claim-time-tag-saturation` | N/A: claim was direct manual route (last P≤2 drain pick), not algorithm-stale. |
| `mem-antipattern-finding-add-scratchpad-drift` | will verify via finding-list before any future citation. |
| `mem-antipattern-schema-incomplete` | all 4 KCs reference trained-object properties. |
| `mem-antipattern-pueue-env-vars` (mem-1777104328-c05a) | run_experiment.py self-contained; no SMOKE_TEST env-var dependency this iter. |
| `mem-antipattern-claim-algo-prefill` (mem-1777091352-2725) | manual route at last P≤2 drain pick; bypasses claim algo. |
| F#666 target-gating audit | matrix unchanged from parent §3. |
| F#669 cascade-cluster | N/A: this iteration is pure inspection (load + topology), no schema-broken dependencies. |

## §9 No smoke gate (Phase A is plumbing-only, not training)

The smoke-gate convention (5-gate validation) applies to runs that *train* an adapter. Phase A loads the base, inspects modules, and writes a readout — there is no training, so no smoke-gate. Future Phase B (full training) will add a smoke-gate with the F#790-conciseness_full template.

## §10 References

- F#82 (conclusive): composition taxonomy micro-d.
- F#627: Gemma 4 E4B LoRA target choice (v_proj + o_proj).
- F#666: target-gated KC requirement.
- F#673: `mx.clear_cache()` between phase trainings.
- F#679 (provisional): parent geometric proxy.
- F#686 (provisional, parent of this `_impl`): design-only.
- F#793/F#795/F#797/F#798: cross-validated `enable_thinking=False` MMLU mitigation (relevant for future Phase B+C eval).
- F#799 (provisional, prior iter): `exp_memento_gemma4_replication_impl` Phase A precedent.
- arxiv:2402.09353 — DoRA.
- arxiv:2402.11260 — MoLoRA.
- Reviewer.md §5 "PROVISIONAL (novel-mechanism design-only sub-case)" / "Phase A executable slice".
- `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii).
