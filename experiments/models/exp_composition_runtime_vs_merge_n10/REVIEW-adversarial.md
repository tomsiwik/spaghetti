# REVIEW-adversarial — exp_composition_runtime_vs_merge_n10

## Verdict: KILL (preempt-structural, method-dependent redundancy 2nd instance — PROMOTION)

Researcher self-review pass. Adversarial checklist clean. Finding will be filed via `finding-add`. No revisions anticipated; reviewer should verify the 3 theorem structures and the promotion-trigger claim independently.

## Consistency (a)–(d)

- (a) `results.json["verdict"] = "KILLED"` matches DB target `status=killed` (will be set via `experiment complete`); PAPER.md verdict line: "KILLED (preempt-structural, method-dependent redundancy 2nd instance — PROMOTION)" ✅
- (b) `all_pass = false`; both KCs `"inconclusive"`; preempt_kill=true; status=killed consistent ✅
- (c) PAPER.md verdict line: no "PROVISIONAL" / "PARTIALLY" inconsistency ✅
- (d) `is_smoke: false` — no smoke-claim/full-run mismatch; no empirical run performed ✅

## KC integrity (e)–(g)

- (e) KC IDs K1894, K1895 consistent across MATH.md §3 + §4, results.json, PAPER.md prediction table, run_experiment.py output dict ✅. No post-claim relaxation.
- (f) Tautology sniff: N/A (no run); preempt-structural rationale is explicit (BW-bound theorem + branch enumeration).
- (g) Code↔DB: K-IDs in `run_experiment.py` output dict match MATH.md §3 ✅

## Code↔math (h)–(m2)

All N/A — `run_experiment.py` imports only `json`/`pathlib`, writes a graceful-failure `results.json` and exits. No MLX training loop, no LoRA composition, no routing, no `shutil.copy`, no hardcoded `pass: True`, no proxy model substitution.

(m2) Skill invocation: N/A — no platform code to trust. For any future `_impl` follow-up, `/mlx-dev` + `/fast-mlx` would be required.

## Eval integrity (n)–(q)

All N/A — no empirical run.

## Target-gated kill (t)

**Does NOT apply.** This is a preempt-structural KILL. Carve-outs per §5 of reviewer hat:
- Method-dependent redundancy (promoted sub-pattern, 2nd instance): K1895 outcome determinable under every plausible composition × precision branch from prior findings (F#66, F#510, F#511, F#543, F#406, F#54). No KC was measured; (t) gates kills on proxy-FAIL *data*, here no data exists.
- F#399-derivable structural impossibility: K1894 FAIL derives from a published BW-bound theorem. Not a proxy measurement.
- F#666-pure corroboration: K1895 unbound "quality" (|Target|=0); K1894 infrastructure-benchmark bucket.

## Scope-changing fix (u)

N/A — graceful-failure stub is the canonical preempt-structural artifact, not a scope change.

## Deliverables (r)–(s)

- (r) PAPER.md §"Prediction vs measurement" table present with both KCs ✅
- (s) Math errors: checked.
  - **Thm 1 arithmetic check.** F#399 impossibility-structure: speedup ≤ 1 + r·(d_model/15)/Base_BW. At N=10, r=8, d_model=3584 (Gemma-4B), Base_BW estimate ≈ 340 MB (from F#399 Qwen3-0.6B baseline scaled). Effective rank-units N·r = 80. Upper bound on merge speedup: 1 + 80·(3584/15)/340e6 ≈ 1 + 80·239/3.4e8 ≈ 1 + 5.6e-5 ≈ 1.0001. Even pessimistically rescaled for smaller Base_BW at 2.6B model (base_bw ~ 2e8), bound remains < 1.001×. K1894's 2.0× threshold is not approached. ∎
  - **Thm 2 branch-enumeration check.** 5 composition × precision cells enumerated (bf16 merge, fp32 merge, int4/int8 standard-LoRA, Grassmannian runtime, uniform additive merge). Each cell cites a specific finding. No orphan branch. One could argue "Grassmannian bf16 merge at N=10" is not directly tested — but F#66 (bf16 50% delta loss) is *independent of* composition scheme; the precision pathology dominates. Accepted.
  - **Thm 3 F#666-pure classification.** K1895: "quality" unbound, |Target|=0 per guardrail 1007 — correct. K1894: latency ratio infrastructure — consistent with F#715's classification (serialization-format latency ratios).

## Promotion trigger audit

Researcher claims F#731 was the 1st drain-window method-dependent-redundancy instance; this (F# TBD for this experiment) is the 2nd → PROMOTION. Verification:
- F#731 PAPER.md §"Triple-fire / composition context" names this as "new sub-pattern candidate, 1st drain-window instance" and explicitly lists `exp_composition_runtime_vs_merge_n10` as a sibling watchlist candidate.
- Scratchpad analyst entry for F#731 KILL synthesis: "Watchlists filed (no memory promotion yet): 1. **Cross-parent triple-fire** [...] 2. **§5 intra-instantiation sub-variant ledger** [...] 3. **F#713-child F#669 census**" — wait, this is F#730's scratchpad. Let me recheck F#731's analyst synthesis: "Watchlists: 1. **Method-dependent redundancy**: 2nd instance promotes standalone memory. Sibling candidates: `exp_composition_runtime_vs_merge_n10`, ..." ✅ match.
- Therefore the 2nd-instance promotion trigger is correctly invoked.

## F#702 hygiene-patch

- platform: `local-apple` ✅ (set via `experiment update`)
- dir: `micro/models/exp_composition_runtime_vs_merge_n10/` ✅ (set via `experiment update`)
- evidence: will be populated via `experiment complete --evidence`
- references: cited inline in PAPER.md (9 prior findings) and MATH.md §8
- success_criteria: CLI flag not supported; omitted per precedent (non-blocking)

## Novel sub-pattern watchlist (analyst note)

**F#399-derivable structural impossibility** — 1st drain-window instance to preempt-KILL a KC via *arithmetic derivation from an inequality in an existing supported finding*. Distinct from:
- Method-dependent redundancy (branches all covered by different findings)
- F#666-pure (KC is target-unbound proxy)
- F#669 (parent target unverified)
- F#702 (method unavailable)

Here F#399 is a *published closed-form theorem* whose inequality directly covers K1894's threshold, not via redundancy or proxy-ness but by arithmetic. Watchlist: 2nd instance (e.g., any future KC that can be FAIL-derived by plugging numbers into a supported theorem) would promote a standalone memory.

## Routing

Researcher to execute `experiment complete --status killed` and `experiment finding-add`, then emit `experiment.done`. Reviewer (next hat) will verify this review independently; analyst will capture LEARNINGS + memory promotion.
