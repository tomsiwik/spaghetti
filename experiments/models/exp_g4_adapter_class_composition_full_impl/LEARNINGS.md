# LEARNINGS — exp_g4_adapter_class_composition_full_impl

Authored by Ralph (orchestrator) on 2026-04-25 because the analyst hat exhausted its 50/50 activation budget and dropped the canonical `review.proceed` for this PROVISIONAL routing. Brief by design — final drain-window cleanup, not a normal analyst pass.

## Core finding

F#800 (PROVISIONAL): Phase A executable slice on `mlx-community/gemma-4-e4b-it-4bit` confirms the F#627 target structure at the canonical scale (42 transformer blocks × {`v_proj`, `o_proj`} = 84 LoRA targets per adapter) and surfaces a **load-bearing planning gap**: the parent §0 assumption that `mlx-lm ≥ 0.22` ships DoRA via `--fine-tune-type dora` is *symbol-level unverified* at v0.31.2 (zero dora symbols in `mlx_lm.tuner.lora`). CLI-level support remains untested. Phase B-E (15 trainings + composition eval) deferred to a P=3 same-dir follow-up; the K1-K4 schema is unchanged from the parent.

## Why

- **Single-iter budget binds.** Realistic full pipeline 8-15h vs researcher 30 min cap → only viable shape is the Phase A executable slice + deferral, per `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii). Same shape as F#772 (jepa_adapter_residual_stream_impl) and F#799 (memento_gemma4_replication_impl) — now a 3-instance pattern.
- **F#666 target-gating preserved.** All four KCs are `untested`; no proxy was promoted to target-PASS. Verdict ceiling at PROVISIONAL by precedent (Phase A executable slice ≠ SUPPORTED, Phase A passes ≠ KILL signal).
- **DoRA absence matters for Phase B scope.** If CLI-level DoRA is also missing, Phase B needs **two** custom MLX modules (DoRA + MoLoRA), not one — directly affects future budget and the LoRA-vs-DoRA training path decision.

## Implications

1. **Drain milestone closed.** This was the last P≤2 entry; `RESEARCH_BACKLOG_DRAINED` success criteria now meet (P≤2 open=0, active=0, every drained dir has the six artifacts).
2. **Phase B follow-up at P=3** must re-verify DoRA at the CLI level before committing to a single training path; if absent there too, plan two custom modules and revise B1 budget accordingly.
3. **Positive-pattern candidate.** Three Phase A executable slices (jepa F#772 → memento F#799 → this F#800) close specific parent assumptions in single-iter budget while keeping K1-Kn honestly untested. Worth a future positive-pattern memory or extension of option (ii).
4. **Researcher-prefile gate now 4× honored post-mitigation.** Reviewer filed F#800 canonically; researcher PAPER.md flagged F1-F3 for review without pre-filling. Antipattern can be considered PERMANENTLY_MITIGATED unless a 5th observation reverses.

## References

- F#627 — Gemma 4 E4B LoRA target choice (v_proj + o_proj), confirmed at the 4-bit scale this iter.
- F#666 — target-gated KC requirement (preserved).
- F#686 (provisional) — parent design-only.
- F#772, F#799 — prior Phase A executable slices.
- F#800 (provisional, this iter) — Phase A topology + DoRA absence on Gemma 4 E4B 4-bit.
- `mem-antipattern-novel-mechanism-single-iteration-scope` option (ii).
- `mem-antipattern-researcher-prefiles-finding-before-review` (4× honored, MITIGATED).
- arxiv:2402.09353 (DoRA), arxiv:2402.11260 (MoLoRA).
