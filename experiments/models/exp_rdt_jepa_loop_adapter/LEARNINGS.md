# LEARNINGS — `exp_rdt_jepa_loop_adapter`

**Status:** PROVISIONAL (novel-mechanism design-only; empirical verification deferred to `_impl` at P3).

## What this iteration produced

- **MATH.md** (9 sections) with proof-first grounding:
  - §0 Skills invoked (`/mlx-dev` + `/fast-mlx`), scope-preservation lock (F1–F6), pinned versions.
  - §1 Architecture: RDT loop (inherited F#674) + JEPA prediction head P_θ + SIGReg anti-collapse with **novel cross-depth collapse failure mode** (vs. sibling F#682's layer-wise surface).
  - §2 Cited prior math: LeWorldModel, LeJEPA, Bae 2024 RDT, F#627, F#666, F#674, F#1629.
  - §3 Five KCs target-gated per F#666: K#1770 structural-precondition (F#666 carve-out), K#1771↔K#1774 isotropy↔depth-elasticity pair, K#1772↔K#1773 learning-dynamics↔GSM8K pair.
  - §4 Mechanism theorem: auxiliary JEPA loss adds T constraint pairs per step vs parent's single-iterate CE loss; preserves contractive guarantee.
  - §5 Prediction table (P1–P5 with numeric falsifiers).
  - §6 Scope escalation — PROVISIONAL-as-design justification (6–10h pipeline vs. 30 min cap).
  - §7 Antipattern self-audit (11 items checked).
  - §8 Assumptions (A1–A8 including analyst routing C2 justification).
  - §9 QED.
- **run_experiment.py** graceful-failure scaffold: `main()` never raises, always writes `results.json` with `verdict="PROVISIONAL"`, `all_pass=false`, all 5 KCs `"not_measured"` with per-KC unblock pointers. Executed via pueue in 2.11s.
- **results.json** with canonical PROVISIONAL structure (per reviewer.md §5 req #2). `mlx_importable=True` confirmed; no hidden import failures.
- **PAPER.md** with prediction-vs-measurement table (5 rows, all "NOT MEASURED") + explicit scope rationale + F#666 pairing notes + analyst C2 routing citation + verdict-consistency pre-flight (all 6 checks cleared for PROVISIONAL).
- **REVIEW-adversarial.md** researcher-authored placeholder (reviewer hat will overwrite).
- **`_impl` follow-up filed at P3**: `exp_rdt_jepa_loop_adapter_impl` inheriting MATH.md verbatim with all 5 KC IDs (#1770–#1774). Dep-linked to `exp_rdt_loop_kv_cache_impl` (P3, infra unblock) per analyst C2.

## What was *not* produced (honest reporting)

- Custom MLX training loop implementation. Requires ~4–6h of careful MLX engineering (monkey-patch Gemma4TextModel forward + P_θ + cross-depth target construction + SIGReg Epps-Pulley numerical integration + `nn.value_and_grad` training with `mx.eval`/`mx.clear_cache` discipline).
- Any of 500 training steps, SIGReg λ bisection, GSM8K-Hard eval (n=200 × 3 arms), depth-elasticity sweep (6 Ts × 30 prompts × 2 arms).
- K#1770–K#1774 empirical measurements.

Single-iteration researcher-hat budget (30 min / 40 tool calls) is ~12–20× under the required 6–10h end-to-end pipeline. Silently substituting a cheaper objective (e.g. dropping SIGReg, dropping cross-depth prediction, running at T=1 only) would be an antipattern-'t' (scope-preservation) violation; `mem-antipattern-novel-mechanism-single-iteration-scope` mandates PROVISIONAL-as-design with `_impl` at P3 as the honest response.

## Takeaways for the next researcher/analyst iteration

1. **Novel-mechanism drain complete for P≤2 (this-window summary).** 9 P≤2 entries resolved in 2026-04-23 window via reviewer.md §5 canonical clauses:
   - 5 novel-mech PROVISIONAL: F#682 (JEPA layer-wise), F#683 (hedgehog politeness), F#684 (hedgehog refactor), F#685 (MEMENTO Gemma 4), F#686 (g4-adapter-class-full).
   - 3 preempt-KILL: F#687 (single-parent), F#688 (triple-parent), F#689 (dual-parent).
   - 1 macro-scope standard-mech PROVISIONAL-as-design: F#690 (kv-cache).
   - This iteration adds a 10th: novel-mech PROVISIONAL-as-design for `exp_rdt_jepa_loop_adapter`.
   - Expected F#691 filing.

2. **Post-iteration queue:** P≤2 open should be **0** (this was the final P1). `RESEARCH_BACKLOG_DRAINED` is the expected next-iteration researcher verdict per researcher.md step 2 literal reading. Active queue has 1 entry (`exp_model_knowledge_gap_26b_base`, 14GB download blocker out-of-band).

3. **Infra-feasibility vs. behavioral-KC distinction formalized (analyst C2).** This iteration demonstrates a real case where F#669 does NOT apply despite a PROVISIONAL parent: parent's PROVISIONAL status is on infra-feasibility KCs (K1764 bit-exact, K1765 speedup), not behavioral mechanism KCs. Child's behavioral targets depend on parent's infra being *runnable* in `_impl`, not on parent's infra being *SUPPORTED*. This is a testable semantic distinction — analyst should consider whether to promote this as a refinement clause on F#669 via memory edit (or keep as case-by-case routing per `mem-antipattern-preempt-child-parent-target-unverified`).

4. **Novel cross-depth collapse failure mode is a genuine contribution.** MATH.md §1.3 identifies a failure mode not present in sibling F#682 (layer-wise JEPA): `h_d = h_{d+1}` for all d (fixed-point reached at d=1, loop does nothing). SIGReg per-d isotropy (K#1771) rules it out. This is not a rehash of LeWM/LeJEPA — it's an extension to the recurrent-depth axis. The `_impl` iteration will be the first empirical test of this specific collapse mode.

5. **Picker-bug workaround functional (8th iteration running).** Claimed via explicit `--id exp_rdt_jepa_loop_adapter` override per handoff. No mispick cycle this iteration (workaround avoids the claim-picker entirely). Operator-side picker-logic intervention remains pending; this iteration added no new diagnostic signal.

## Not learned / open

- Whether JEPA objective on recurrent depth transfers knowledge into Δ on Gemma 4 E4B (the central claim) — requires `_impl`.
- Whether the novel cross-depth collapse failure mode actually occurs absent SIGReg, or whether parent's CE loss is strong enough to prevent it on its own — requires `_impl` ablation arm.
- Whether λ bisection over {0.0, 0.1, 1.0, 10.0} is wide enough, or whether the optimal λ sits outside this grid — requires `_impl` bisection run.
- Whether T=3 is the right operating point for GSM8K-Hard (parent F#674's K1740-BENCH pre-reg) or whether a deeper T is better — `_impl` depth-elasticity sweep (K#1774) will expose this.

## References

- **LeWorldModel** (Maes/LeCun/Balestriero 2026-03-24, arxiv:2603.19312) — SIGReg stabilized end-to-end JEPA for pixel world-models.
- **LeJEPA** (arxiv:2511.08544) — Thm 1 + Eq. 7 Epps-Pulley statistic.
- **Bae 2024 Relaxed Recursive Transformers** (arxiv:2410.20672) — depth-recurrent shared LoRA framework.
- **Finding #627** — `v_proj + o_proj` is the proven Gemma 4 E4B adapter target.
- **Finding #666** — Target-gated kill rule (proxy + target pairing).
- **Finding #669** — Preempt-structural kill for child-of-unverified-parent (behavioral-KC transitivity axis).
- **Finding #673** — MLX skill discipline (F#673 lineage).
- **Finding #674** — Parent `exp_rdt_loop_lora_gemma4_bench` PROVISIONAL (structural PASS, behavioral underpowered).
- **Finding #682** — Sibling `exp_jepa_adapter_residual_stream` PROVISIONAL (layer-wise JEPA, first JEPA design-only precedent).
- **Finding #690** — Sibling `exp_rdt_loop_kv_cache` PROVISIONAL (macro-scope infra, K1764/K1765 infrastructure axis).
- **Finding #1629** — max_tokens=1024 prevents Gemma 4 CoT truncation on GSM8K.
- `mem-antipattern-novel-mechanism-single-iteration-scope` — governing memory for this filing's PROVISIONAL-as-design route.
