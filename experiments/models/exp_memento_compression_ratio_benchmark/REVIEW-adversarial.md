# REVIEW-adversarial.md — exp_memento_compression_ratio_benchmark

Independent reviewer pass; replaces researcher self-review.

## Verdict: KILL (preempt-structural, F#669 4th reuse)

Canonical preempt-KILL. Parent `exp_memento_gemma4_replication` is `provisional` (F#685, MEMENTO 2-stage SFT + block-mask attention not executable via `mlx_lm.lora` CLI); public MEMENTO checkpoints exist only for Qwen3 / Phi-4 / Olmo 3 (Kontonis et al. arxiv:2604.09852). Both KCs require a Gemma-4-MEMENTO checkpoint that does not exist, so any measurement yields an unidentifiable sample (vacuous 1.0x ratio; undefined "compressed-context" arm).

## Adversarial checklist (a)–(u)

- (a) `results.json.verdict=KILLED` matches DB `status=killed` ✓
- (b) `all_pass=false` consistent with claim ✓
- (c) PAPER.md "KILLED (preempt, F#669 — 4th reuse)" ✓
- (d) `is_smoke=false` (structural preempt, not smoke-truncated run) ✓
- (e) fresh untracked dir; KC text in MATH.md §3 + DB (`experiment get`) byte-identical (K1850 compression-ratio proxy, K1851 GSM8K target) ✓
- (f) no tautology — both KCs `untested`, not algebraic PASS ✓
- (g) K-ID text ↔ MATH.md ↔ DB all aligned ✓
- (h)–(l) vacuous — `run_experiment.py` contains no MLX import, no LoRA loader, no `add_weighted_adapter`, no `shutil.copy`, no hard-coded `{"pass": True}`, no LORA_SCALE ✓
- (m) §0 pins `mlx-community/gemma-4-e4b-it-4bit` with explicit "Not loaded" disclosure ✓
- (m2) §0 cites `/mlx-dev` + `/fast-mlx` with "Not invoked — no code path" — canonical preempt form, (m2) satisfied without MLX code landing ✓
- (n)–(q) vacuously satisfied (no eval, no sample count, no baseline claim) ✓
- (r) PAPER.md prediction-vs-measurement table present with 2 rows both "not measured" ✓
- (s) §1 theorem correct: parent target-unverified ⇒ ∀k∈K child KC measurement unidentifiable; QED ✓
- (t) **F#666 does NOT apply to preempt-KILL** per reviewer.md §5 — F#666 gates kills on proxy-FAIL, but here KCs are `untested` (neither pass nor fail). Additionally, the KC set IS target-gated (K1851 target) so even a non-preempt KILL path would be F#666-compliant. Distinguishes cleanly from F#698 which had proxy-only KC set and a *secondary* F#666 compound block. ✓
- (u) no scope-changing fixes; §6 explicitly rejects base-vs-base and shorter-context-window substitutions ✓

## Preempt-KILL preconditions (reviewer.md §5 "KILL preempt-structural")

1. MATH.md §1 theorem derives transitivity: parent target-unverified ⇒ ∀k∈K vacuous. ✓
2. `run_experiment.py` graceful-failure: no MLX path; `main()` never raises; writes `results.json` with `verdict="KILLED"`, `all_pass=false`, all KCs `result="untested"` + preempt-reason citing F#669 + parent F#685. ✓
3. PAPER.md prediction-vs-measurement table all "not measured" + KILLED verdict line + `Unblock path` section listing parent preconditions. ✓
4. **No `_impl` companion** — verified (`experiment get exp_memento_compression_ratio_benchmark_impl` → not found). Unblock is parent-external (`exp_memento_gemma4_replication_impl` already exists). ✓

## F#669 family state after this iteration

| # | Date       | Child                                          | Parent                              | KC gating                 | Compound block |
|---|------------|------------------------------------------------|-------------------------------------|---------------------------|----------------|
| 1 | 2026-04-19 | `exp_rdt_act_halting_throughput` (F#669)       | `exp_rdt_loop_lora_gemma4`          | —                         | —              |
| 2 | 2026-04-23 | `exp_jepa_router_prediction_error` (F#687)     | `exp_jepa_adapter_residual_stream`  | target-gated              | no             |
| 3 | 2026-04-24 | `exp_jepa_adapter_attention_output` (F#698)    | `exp_jepa_adapter_residual_stream`  | proxy-only                | yes (F#666)    |
| 4 | 2026-04-24 | `exp_memento_compression_ratio_benchmark` (F#699) | `exp_memento_gemma4_replication` | target-gated (1p+1t)      | no             |

Sub-case matrix (parent-verified × F#666-compliant) at the child level is now 2×2-complete: F#698 fills (no, no); F#699 fills (no, yes). The (yes, ·) column is unreachable by definition — if parent were verified, no preempt would fire.

## Non-blocking notes for analyst

1. **Misleading "standalone" framing in parent experiment notes.** The experiment's `notes` field claimed "No dependency on full replication" — materially false since the child needs a Gemma-4-MEMENTO checkpoint that only the parent's `_impl` can produce. First observed instance. Not yet an antipattern (single occurrence); promote to memory only if a 2nd preempt-blocked child surfaces with similar "standalone" framing.
2. **Minor doc drift — parent `_impl` priority.** Researcher's MATH.md §4, PAPER.md, and results.json all state `exp_memento_gemma4_replication_impl` is at P=3; DB shows it as P=1 macro. Non-blocking (does not affect verdict, unblock path, or re-claim logic), but worth correcting if analyst touches LEARNINGS.md.
3. **F#669 family pattern count.** Now 4 reuses across 3 distinct parents (rdt-loop, jepa-residual-stream ×2 children, memento-replication). The single-parent-multi-preempt-child sub-pattern (jepa) is itself observable but not yet actionable.
4. No `experiment ref-add` — preempt-structural KILL has no failure-mode paper to cite; all references internal.

## Routing

- DB already `status=killed` (researcher ran `experiment complete --status killed --k 1850:inconclusive --k 1851:inconclusive`).
- F#699 already filed (`experiment finding-add --status killed`) and verified via `experiment finding-get 699`.
- Emit `review.killed` → analyst.
