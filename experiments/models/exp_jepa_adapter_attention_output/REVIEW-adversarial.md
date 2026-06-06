# REVIEW-adversarial.md — exp_jepa_adapter_attention_output (reviewer independent pass)

## Verdict: KILL (preempt-structural, F#669 3rd reuse)

Researcher's preempt-KILL is correct on all structural grounds. Parent `exp_jepa_adapter_residual_stream` is `provisional` (F#682), so every KC here is transitively unidentifiable per MATH.md §1. The KC set (K1848, K1849) is also proxy-only — F#666 violation independent of parent status. Double-block preempt; either suffices alone. DB already at `status=killed` and F#698 already filed by researcher — this review overwrites the self-review for a canonical independent pass.

## Adversarial checklist (a)–(u)

- (a) `results.json.verdict=KILLED` ↔ DB `status=killed` ↔ PAPER.md verdict line → consistent ✓
- (b) `all_pass=false`; both KCs `result="untested"` (structural, not FAIL) → correct for preempt (no KC measured) ✓
- (c) PAPER.md `KILLED (preempt, F#669 — 3rd reuse)` matches results.json ✓
- (d) `is_smoke=false` — no smoke pass run; no MLX code path → correct (this is preempt, not smoke-PROVISIONAL) ✓
- (e) MATH.md in fresh dir; KC text K1848/K1849 verbatim match DB pre-registration; no post-hoc mutation ✓
- (f) No tautology — no KC measured; "untested" preempt-reasons cite F#669/F#682/F#666, not algebraic identity ✓
- (g) KC IDs #1848/#1849 in results.json ↔ MATH.md §3 table ↔ DB pre-reg verbatim ✓
- (h)–(l) No MLX code path — no composition, no `LORA_SCALE`, no per-sample-route-of-one-sample, no `shutil.copy`, no hardcoded `{"pass": True}` → PASS vacuously (and honestly)
- (m) MATH.md §0 pins intended base model `mlx-community/gemma-4-e4b-it-4bit` (F#627) with explicit "Not loaded" — honest disclosure; no proxy substitution ✓
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` with explicit "**Not invoked** — no MLX code written" → canonical form for design-only / preempt-structural filings per reviewer.md §5 ✓
- (n)–(q) Eval-integrity items vacuous — no eval run, no baseline cited as headline
- (r) PAPER.md prediction-vs-measurement table present; both rows "not measured" + explicit "untested" verdict ✓
- (s) §1 theorem reasoning sound: K1848 references parent's unverified baseline MSE (RHS of inequality); K1849 superficially measurable but loses interpretive anchor without parent's SIGReg-stability claim target-validated (K1767/K1769 in F#682 still untested). §1.1 compounds with F#666 proxy-only gating.
- **(t) Target-gated kill (F#666) DOES NOT APPLY to preempt-KILL** — per reviewer.md §5 preempt-structural KILL clause: "F#666 gates kills on proxy-FAIL; preempt-KILL is a structural verdict where NO KC was measured (proxy or target)." Here F#666 appears only as a *secondary independent block* (future re-claim requires KC-augmentation), not as the primary kill gate. Routing correct. ✓
- (u) No scope-changing fixes — no scope was executed; honest preempt, no silent mechanism swap, no `max_length` reduction, no trackio disable, no smaller-base fallback ✓

All (a)–(u) PASS. No blocking items.

## F#669 reuse ledger (3rd reuse — promotion confirmed)

1. F#669 (2026-04-19) `exp_rdt_act_halting_throughput` over `exp_rdt_loop_lora_gemma4` (4 KCs)
2. F#687 (2026-04-23) `exp_jepa_router_prediction_error` over `exp_jepa_adapter_residual_stream` (4 KCs)
3. **F#698** (2026-04-24, this) `exp_jepa_adapter_attention_output` over `exp_jepa_adapter_residual_stream` (2 KCs, **proxy-only**)

Second-reuse promotion threshold was already hit at F#687; third reuse confirms. reviewer.md §5 has already absorbed the "KILL (preempt-structural sub-case)" clause — promotion is DONE in the hat doc. No further promotion action needed.

## New sub-case element vs F#687

This is the first F#669-family preempt where the **child KC set itself** violates F#666 (proxy-only, no target gate). F#669/F#687 children had target KCs present in pre-registration; here re-claim requires *two* conditions before an F#666-safe re-run:

1. Parent SUPPORTED with K3/K4 SUPPORTED (the F#669/F#687 standard unblock condition).
2. **KC-augmentation**: child must be re-pre-registered with a target-metric KC before claim. Without this, re-run hits F#666 even after parent SUPPORTED.

PAPER.md §Unblock path documents both conditions. This sub-case distinction is non-blocking for the current verdict but worth flagging in F#698's `failure_mode` (already present in the finding text).

## Assumptions

- Parent `exp_jepa_adapter_residual_stream` status at filing time is `provisional` per F#682 and current DB (verified at review time).
- No redesign attempted this iteration (paired attn_output + residual-stream A/B in one experiment would remove parent dependency). Out of scope per drain objective; PAPER.md §Unblock path notes as "Alternative unblock".
- `_impl` companion NOT filed — correct per reviewer.md §5 preempt-KILL clause (unblock is parent-external via existing `exp_jepa_adapter_residual_stream_impl` at P=3).

## Non-blocking notes for analyst

1. Pattern count this drain: **6 novel-mechanism PROVISIONALs** (F#682 JEPA residual, F#683 hedgehog_behavior, F#684 hedgehog_procedural, F#696 hedgehog_domain_js, F#697 hedgehog_domain_python) + **4 preempt-KILLs** in the F#669 family (F#669 RDT, F#671, F#672, F#687 JEPA router, F#698 JEPA attn_output) + **3 SUPPORTED** (budget_forcing_baseline_fix, semantic_router_macro, cayley_riemannian_adam) + **1 KILL** (kv_cache_reuse_honest). The drain is being paced by parent `_impl` landings, not child KC design.
2. No antipattern candidates — clean application of the preempt-KILL (F#669/F#687) clause + honest F#666 secondary-block documentation. Analyst step-5 trigger does not fire.
3. Non-blocking LEARNINGS angle: F#669 family has now preempted 5 distinct child experiments. A meta-observation about "child-of-PROVISIONAL-parent" claim-time guards (e.g. automatic preempt at `experiment claim` if parent in `{provisional, open}`) could be valuable *as a reference memory only* — not a new antipattern, just a tool ergonomics hint.

## Route

Emit `review.killed` with payload: id=exp_jepa_adapter_attention_output, verdict=KILL preempt-structural, F#698 filed, 3rd F#669 reuse + F#666 compound block, no `_impl`.
