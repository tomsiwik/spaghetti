# REVIEW-adversarial — `exp_jepa_adapter_residual_stream_impl`

**Verdict.** PROVISIONAL (novel-mechanism design-only sub-case per F#682 precedent).

**Date.** 2026-04-25 (reviewer iter ~42, drain-window).

**Reviewer.** Reviewer hat, doom-loop self-check exit=0 (different mechanism than prior 4 schema-repair iters; this is the verdict-path companion to claim-and-run #1).

---

## Adversarial checklist (every item)

**Consistency:**
- (a) `results.json["verdict"]` = `"PROVISIONAL"` ↔ DB `status` = `provisional` — **CONSISTENT** ✓
- (b) `all_pass=false` ↔ PROVISIONAL claim — **CONSISTENT** ✓
- (c) PAPER.md verdict line "PROVISIONAL" ↔ DB `provisional` — **CONSISTENT** ✓
- (d) `is_smoke=true` ↔ PROVISIONAL (correct downgrade, not silent upgrade) ✓

**KC integrity:**
- (e) KC IDs #1817-#1820 inherited verbatim from parent K#1766-#1769; no addition/modification post-claim. Verified via `experiment get` ✓
- (f) Tautology sniff: K#1817 (Epps-Pulley rejection rate against N(0,1)), K#1818 (loss ratio between training steps), K#1819 (GSM8K-Hard accuracy delta vs measured baseline), K#1820 (λ=0 ablation gap) — none algebraic identities; KC text unchanged from parent ✓
- (g) K-IDs in `run_experiment.py:243-247` results dict label match MATH.md §5 + DB ✓

**Code ↔ math:**
- (h) No `sum(lora_A)` / `add_weighted_adapter` / safetensors composition bug — single-adapter experiment ✓
- (i) `LORA_SCALE = 6.0` (line 51) ≤ 8 per F#328/F#330 ✓
- (j) No routing logic ✓
- (k) No `shutil.copy(...)` of sibling adapter ✓
- (l) No hardcoded `{"pass": True}` — all KCs read `"untested"` ✓
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` (line 41) matches MATH.md §0 / parent §1 ✓
- (m2) **/mlx-dev attestation present in MATH.md §0** (lines 5-13) — knowledge cited (`mx.eval` boundaries, `mx.clear_cache()`, `nn.value_and_grad`, `mlx.optimizers.AdamW`, lazy graph, slice-copy semantics, bfloat16 preferred). `/fast-mlx` deferred with explicit rationale (Phase A is subprocess-only, `mlx_lm.lora` CLI handles training). Reviewer (m2) gate **SATISFIED** ✓

**Eval integrity:**
- (n) Phase D baseline accuracy = 40% (4/10 GSM8K-Hard) with `max_tokens=1024` (F#1629 honored) — not 0% / not thinking-suppression ✓
- (o) Phase D n=10 < 15 — but `is_smoke=true` so this is permissible smoke routing; PROVISIONAL absorbs this caveat ✓
- (p) No synthetic padding ✓
- (q) Baseline measured **in-run**, not cited externally ✓

**Deliverables:**
- (r) PAPER.md prediction-vs-measurement table present (4 rows, all "untested" or "partial — baseline only" for K#1819) ✓
- (s) Math chain (LeJEPA Thm 1 / Cramér-Wold / Epps-Pulley statistic) cited consistently ✓

**Kill-gating (informational, not blocking PROVISIONAL):**
- (t) Target-gated KILL (F#666) — N/A (verdict is PROVISIONAL, not KILL)
- (u) Scope-changing fix — Phase B `train_jepa_adapter()` honestly raises `NotImplementedError` with structured marker; no silent fallback to standard LoRA. Researcher correctly preserved scope ✓

## PROVISIONAL routing rationale (per reviewer hat clause)

JEPA is a **novel training mechanism** not executable via `mlx_lm.lora` CLI (requires custom training loop with layer-21 hook + 2-layer MLP prediction head + SIGReg Epps-Pulley regularizer over M=1024 random projections). The researcher filed the canonical design-only artifact pattern:
1. ✓ MATH.md §0 cites `/mlx-dev` (invoked) + `/fast-mlx` (deferred with rationale).
2. ✓ `run_experiment.py:main()` never raises; always writes `results.json` with `verdict="PROVISIONAL"` and KCs `"untested"`.
3. Need: `_impl_v2` follow-up filed at P3 inheriting MATH.md verbatim (Phase B implementation iteration).
4. ✓ PAPER.md prediction-vs-measurement table with all rows "not measured" + scope rationale.

The marginal contribution this iteration over the parent's PROVISIONAL state (F#682):
- Plumbing-verified Phase A token-space LoRA r=16 baseline (parent deferred this) — val loss 1.840→0.581 in 50 steps, 100s wall, 6.71 GB peak.
- Phase D baseline anchor measured: GSM8K-Hard n=10 = 40.0% (smoke; full n=200 needed for K#1819 to bind).
- /mlx-dev knowledge attested.

These are **incremental but real** drain-progress: claim-and-run executed without scope-violation, baseline anchor exists for future K#1819 binding.

## Assumptions (per reviewer autonomy)

- 40% n=10 baseline has wide CI (~±30pp); accepted as smoke-anchor only, not as a binding measurement. Future Phase B run must beat or match the **full n=200** rerun of Phase A, not the smoke value.
- `_impl_v2` follow-up at P3 maintains drain criterion 1 (no open P≤2). Keeping at P3 is the canonical pattern (F#682/F#683/F#684 precedents).

## Routing

PROVISIONAL → emit `review.proceed` with `PROVISIONAL:` prefix. Two-step DB workaround already executed by researcher (status=provisional + evidence). Reviewer adds: `experiment finding-add --status provisional` + verify via `experiment finding-list --status provisional` + file `_impl_v2` follow-up.

No blocking fixes. No revise cycle.
