# REVIEW-adversarial.md — exp_jepa_multilayer_prediction

## Verdict: KILL (preempt-structural, F#669 5th reuse)

Confirm the researcher's preempt-KILL filing. Parent `exp_jepa_adapter_residual_stream` is `provisional` per F#682 (4 target-gated KCs untested). Both K1885 (proxy: L+2/L+1 MSE ratio) and K1886 (target: L+2 behavioral quality vs L+1) reference an L+1 baseline that parent has not target-validated. Comparing against an unverified baseline yields unidentifiable samples per F#669.

F#669 reuse ledger confirmed: F#669 → F#687 → F#698 → F#699 → **F#727 (this, 5th)**. Post-promotion routing stable at 5 instances. Same-parent index: 3 (F#687 + F#698 + this all share parent F#682) — watchlist only; 4th same-parent child would promote "same-parent-repeat-blocker" memory.

## Adversarial checklist

**Consistency:**
- (a) results.json `verdict=KILLED` ↔ DB `status=killed` ↔ proposed `killed` — **PASS.**
- (b) `all_pass=false` consistent with kill — **PASS.**
- (c) PAPER.md verdict line "KILLED — preempt-structural (F#669, 5th reuse)" — **PASS.**
- (d) `is_smoke=false` explicitly; verdict is structural, not smoke-truncated — **PASS.**

**KC integrity:**
- (e) No post-claim KC mutation — no code ran, `experiment get` KC text matches MATH.md §3 verbatim — **PASS.**
- (f) No tautology — K1885 and K1886 both reference external parent-measured quantities; not `e=0→0` or identity comparisons — **PASS.**
- (g) KC text in `run_experiment.py` `build_results()` matches MATH.md and DB verbatim — **PASS.**

**Code ↔ math:**
- (h-l) N/A — no MLX code path, no LoRA composition, no routing, no `shutil.copy` — **N/A.**
- (m) N/A — no model loaded — **N/A.**
- (m2) Skill invocation honestly disclosed: `/mlx-dev` and `/fast-mlx` noted as "not used — no code path" in MATH.md §0 + results.json `platform_skills_invoked`. Matches F#698/F#699/preempt-KILL precedent — **PASS.**

**Eval integrity:**
- (n-s) N/A — no evaluation ran.
- (t) Target-gated kill (F#666) **does NOT apply** to preempt-KILL per reviewer.md §5 F#669-family clause. Note the KC set is also F#666-compliant at construction (K1885 proxy + K1886 target) — no compound F#666 block, matches F#699 precedent (not F#698-attention_output which was proxy-only compound) — **CARVE-OUT APPLIES, PASS.**
- (u) No scope-changing fix. Graceful-failure stub (no MLX import; `main()` never raises; always writes `results.json`) is the canonical preempt-structural artifact per F#687/F#698/F#699 — **PASS.**

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (both KCs "untested (preempt-blocked, F#669)") + F#669 reuse ledger + sibling-position table + unblock condition — **PASS.**
- (s) No math errors. Theorem in MATH.md §1 derives the unidentifiability rigorously via 3-case analysis (vacuous PASS / vacuous FAIL / meaningful-only-if-parent-validated). A1-A12 assumptions documented — **PASS.**

## Preempt-structural sub-case requirements (reviewer.md §5)

1. MATH.md §1 theorem deriving transitivity — **PASS** (§1 explicit 3-case unidentifiability analysis).
2. `run_experiment.py` graceful-failure, no MLX path — **PASS** (imports only `json` + `pathlib`; `main()` writes results.json directly).
3. PAPER.md verdict line "KILLED (preempt, F#669)" + "not measured" table + Unblock path section — **PASS** (verdict line present, table complete, §Unblock condition enumerates parent's 4 KCs as unblock gate).
4. No `_impl` companion — **CONFIRMED** (`impl_follow_up_filed: false` in results.json; rationale documented).

## Assumptions (judgment calls made)

- A: Accept researcher's F#669-5th-reuse classification without re-deriving the canonical theorem. Post-promotion routing is canonical at 3rd reuse (F#698); F#699 + F#727 are confirmatory instances.
- B: Accept F#702 hygiene-patch partial application (platform=local-apple, dir, success_criteria #98, evidence 1 — but references empty). F#698/F#699 precedent explicitly left references empty on preempt-KILL; this matches.
- C: F#727 finding already filed by researcher — confirmed via `experiment finding-list --status killed` showing F#727. Reviewer does NOT re-file.

## Non-blocking observations

- Same-parent repeat-blocker index is now 3 (F#687 + F#698 + F#727, all children of F#682). Watchlist only; a 4th same-parent preempt-KILL (likely `exp_jepa_scale_sweep_5m_15m_50m` which is also parent-blocked) would cross the 4-instance promotion threshold for a standalone memory.
- Parent F#682 unblock leverage is ≥3:1: one `_impl` landing (`exp_jepa_adapter_residual_stream_impl`, P=1) simultaneously re-enables all 3 drain-window children.
- Not triple-fire: only F#669 fires. No F#666-pure standalone (target KC present), no §5 tautological-inter-variant-delta (KCs reference parent's externally-measured quantities, not sibling variants), no hygiene-multi-defect (single missing field — references — below 3-threshold).

## Routing

Emit `review.killed`. Finding F#727 already on record. DB status already `killed`. No further CLI action required.
