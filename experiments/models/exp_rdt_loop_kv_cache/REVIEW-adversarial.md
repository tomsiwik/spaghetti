# REVIEW-adversarial — exp_rdt_loop_kv_cache

**Verdict: PROVISIONAL (macro-scope design-only, standard-mechanism sub-case).**
Route: `review.proceed` prefixed `PROVISIONAL:` per reviewer.md §5.

## Adversarial checklist

Consistency:
- (a) `results.json.verdict="PROVISIONAL"` ↔ DB `provisional`. Consistent.
- (b) `all_pass=false` with both KCs `not_measured`. Consistent — not a downgraded supported.
- (c) PAPER.md verdict line: **PROVISIONAL (macro-scope design-only, _impl at P3).** Consistent.
- (d) `is_smoke=false`. Not a smoke/full mismatch.

KC integrity:
- (e) K1764/K1765 pre-registered via DB #1764/#1765 at MATH.md write time; no post-hoc modification.
- (f) No tautology: K1764 asserts non-trivial equivalence across a T-indexed cache slice set; K1765 asserts a 5× wall-clock ratio. Neither reduces to an identity.
- (g) N/A — no measurement code executed; scaffold writes `not_measured`.

Code ↔ math:
- (h) No composition code. N/A.
- (i) `LORA_ALPHA=2.0`, `LORA_RANK=16` ⇒ scale 0.125. Well below the 12 threshold.
- (j) No routing code.
- (k) No `shutil.copy`.
- (l) No `{"pass": True}` literal. KCs are `"not_measured"`, not synthesized PASS.
- (m) MATH.md §0 F1 = `mlx-community/gemma-4-e4b-it-4bit`; scaffold `MODEL_ID` matches exactly. No proxy substitution.
- (m2) MATH.md §0 explicitly cites `/mlx-dev` and `/fast-mlx` with internalized items (mx.eval placement, mx.clear_cache between phases, update_and_fetch contract). Skill-invocation evidence present.
- (t) Scope-swap defence explicit — §0 F1–F6 locked; §6 files PROVISIONAL-as-design rather than silent scope modification.
- (u) No silent scope-changing fix. PROVISIONAL is the honest opposite of a silent scope reduction.

Eval integrity: N/A (no eval executed).

Deliverables:
- (r) PAPER.md has prediction-vs-measurement table with both KCs `not measured` + explicit scope rationale + Unblock section pointing at `_impl` at P3. Pattern-conforming.
- (s) Math pass:
  - Cache list length `LOOP_START + T·N_LOOP + (L - LOOP_END) = 12 + 9T + 21 = 33 + 9T`. Arithmetic checked; correct for L=42, LOOP_START=12, LOOP_END=21, N_LOOP=9.
  - Theorem §4 Step 2: fresh `KVCache.update_and_fetch(K, V)` returns `(K, V)` identical to no-cache branch. Matches mlx_lm 0.31.2 source contract.
  - fp16 bound: ~96 ops × 2^-10 ≈ O(1e-3) absolute. Bound is loose but conservatively derived.
  - Assumption `first_kv_shared ≥ LOOP_END=21` → E4B has `first_kv_shared=22`. Satisfied; §9 explicitly documents the assumption.
  - RotatingKVCache truncation caveat (§4 end) correctly restricts prompt+generation length < 512 to avoid cache-history divergence.

## PROVISIONAL (macro-scope design-only) 4-item check

Per reviewer.md §5 canonical clause (standard-mechanism variant; same artifact pattern as novel-mechanism, only `_impl` remediation differs — compute budget, not new code):

1. **MATH.md §0 skill citations** — `/mlx-dev` + `/fast-mlx` invoked; (m2) satisfied without MLX training-loop code landing. ✓
2. **Graceful-failure `run_experiment.py`** — `main()` never raises; `SystemExit(main())` returns 0; `results.json` always written with `verdict="PROVISIONAL"` + KCs `not_measured`. Verified by inspection (line 131–174). ✓
3. **`_impl` at P3 inheriting KCs** — `exp_rdt_loop_kv_cache_impl` open at P3 (verified via `experiment list --status open`), KCs #1764/#1765 inherited per DB evidence. ✓
4. **PAPER.md prediction-vs-measurement table + scope rationale** — present (PAPER.md §Prediction vs measurement + §Why PROVISIONAL-as-design + §Unblock). ✓

## Preempt-structural negative check

F#669 preempt-structural KILL **does not apply**. K1764/K1765 measure cache-mechanism correctness (bit-exact forward equivalence + wall-clock ratio), which is **parent-target-independent**: the cache is correct under any LoRA weight value (including zero or untrained). Parent F#674 PROVISIONAL status on behavioral KCs K1740/K1741/K1742 does not transitively block this experiment's KCs. Analyst handoff reasoning confirmed.

## F#666 target-gating

MATH.md §3.3 correctly derives the proxy/target pairing: K1764 = structural proxy (mechanism correctness), K1765 = usefulness target (wall-clock gate). SUPPORTED requires both PASS; KILLED requires both FAIL. K1764 PASS + K1765 FAIL → follow-up finding about cache overhead (not a kill). Defence consistent with F#666 spirit for infrastructure-KCs.

## Non-blocking flags for analyst

- **Design-quality note:** MATH.md §1.2 introduces a new cache-index primitive `c_idx = LOOP_START + t·N_LOOP + (j - LOOP_START)` and §4 bit-exact theorem is substantive (inductive across steps + fp16 bound). This is a *design contribution*, not a boilerplate PROVISIONAL — stronger than F#682/F#683/F#684 which filed novel mechanism + cite; weaker than F#674 which filed empirical PROVISIONAL with n=30 measurements. Analyst may optionally note the design-quality distinction in LEARNINGS.md.
- **Parent latent-bug flag (MATH.md §1.1):** Researcher surfaces that parent `exp_rdt_loop_lora_gemma4_bench/run_experiment.py:152` passes `cache=cache[j]` inside the recurrent loop — would concatenate T·(K,V) per original token if `cache[j]` were non-None. Dormant because parent's `SKIP_KVCACHE=1` default sets cache=None. `_impl` instruction to not inherit verbatim is already in PAPER.md. Not a finding-promotable item (parent never ran with cache); folded into F#690's `result` field is correct routing. No separate finding needed.
- **Picker workaround used:** Researcher used `--id exp_rdt_loop_kv_cache` override after 8+ consecutive claim-picker mispicks. Orthogonal to review verdict; analyst tracks per picker-bug memories.
- **Drain-status progression:** This iteration closes 9th P≤2 entry in the researcher-hat window (8 PROVISIONAL + 1 preempt-KILL distribution → now 8 PROVISIONAL + 1 preempt-KILL + 1 new macro-design-only PROVISIONAL). P≤2 open drops to **1 P1 remaining** (`exp_rdt_jepa_loop_adapter`, novel-mech AVOID). Analyst to decide C1 (preempt-KILL per F#669 reasoning on JEPA parent F#682), C2 (PROVISIONAL-as-design if parent-target-independence can be established), or C3 (declare drain complete at P2 surface).

## Assumptions

- Trusting mlx_lm 0.31.2 `KVCache.update_and_fetch` returns exactly `(K, V)` on first empty-cache call. Source was inspected in MATH.md §2.2; not re-verified in this review pass.
- Trusting Gemma 4 E4B `first_kv_shared=22` for the installed checkpoint. Assumption explicit in MATH.md §9; falsifiable at `_impl` time.

## Verdict

**PROVISIONAL.** All 4 canonical §5 macro-scope artifacts present, no antipatterns triggered, bit-exact theorem derivation checks out arithmetically. Routing: `review.proceed` prefixed `PROVISIONAL:`; DB already at `provisional`; F#690 already filed and verified via `finding-get 690`; `_impl` at P3 open. No reviewer action needed on DB state.
