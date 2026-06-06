# REVIEW-adversarial — exp_pierre_adapter_hotswap_latency_impl

**Verdict: PROCEED** (status=supported)
**Reviewer hat — independent pass (overwrites researcher self-adversarial).**

## Adversarial checklist

**Consistency (a)–(d):**
- (a) `results.json["verdict"] == "SUPPORTED"` and DB `status=supported` — match.
- (b) `all_pass=true`; both K1953 and K1954 `pass=true`. No KC failed.
- (c) PAPER.md verdict line: "SUPPORTED" — no PROVISIONAL/INCONCLUSIVE/etc.
- (d) `is_smoke=false`; full BENCH_RUNS=20, GEN_TOKENS=16.

**KC integrity (e)–(g):**
- (e) MATH.md §3 KC text matches DB pre-reg verbatim ("> 100ms ⇒ FAIL", "> 1 ⇒ FAIL"). File untracked → no post-claim git edits possible. Thresholds unchanged.
- (f) No tautology. K1953 measures wall-clock attach time (not an algebraic identity); K1954 measures token sequence equality between independent forward passes (not `x==x` — the swap path mutates `model.layers[li].self_attn.{v,o}_proj` between decode steps, identical output verifies parameter-content equivalence under module-identity change).
- (g) K-IDs in `run_experiment.py` (K1953/K1954 thresholds, measurement) align with MATH.md §3 and DB descriptions.

**Code ↔ math (h)–(m2):**
- (h) No composition math (`sum(lora_A)`, `add_weighted_adapter("linear")`, summed safetensor keys) — N/A.
- (i) `LORA_ALPHA=8.0` ≤ 8 (F#328/F#330 safe).
- (j) No per-sample routing in scope.
- (k) No `shutil.copy` of sibling adapters.
- (l) `pass: bool(k1953_pass)` derived from `t_median <= K1953_THRESHOLD_MS`; not hardcoded.
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` matches MATH.md §5 — no proxy substitution to E2B.
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` invocation. Code is idiomatic MLX: `mx.eval(model.parameters())` after parameter mutations, `mx.clear_cache` between phases, `tree_unflatten` for module updates, `make_prompt_cache` from `mlx_lm`, `mx.bfloat16` cast on adapter materialization. No torch-style mutations.

**Eval integrity (n)–(u):**
- (n) N/A (no thinking-channel eval).
- (o) Headline n=20 ≥ 15.
- (p) No synthetic-N-domain padding in the headline measurement; N=5 is the adapter pool, K1953's headline is over 20 attach cycles.
- (q) No cited-but-not-measured baseline.
- (t) **F#666 target-gated**: BOTH KCs are target-metrics (user-facing latency in ms; bitwise token-identity). Not proxy-only. F#666 satisfied structurally — no kill on proxy. Verdict path is SUPPORTED (both targets PASSED), not a kill, so the gate is not exercised here either way.
- (u) Scope-preservation: base model, targets (v_proj+o_proj per F#627), rank=6, α=8.0, N=5 all preserved end-to-end. No silent reduction on resource pressure.

**Deliverables (r)–(s):**
- (r) PAPER.md prediction-vs-measurement table present (lines 13–16).
- (s) Math: Theorem 1 upper bound `42·2·(1+10) μs + 100 μs ≈ 1.0 ms`; measured median 0.970 ms lies inside the rescaled mid-point window [0.5, 1.12] ms (parent §3 cited stale `n_layers=34` and `T=7`; IMPL corrects to 42 and F#627's T=2). Arithmetic correct. The 0.97 ms vs original [0.4, 0.9] ms gap is fully accounted for by the layer-count correction (×42/34 = 1.24).

## Non-blocking observations

- **DB hygiene defect**: `⚠ INCOMPLETE: success_criteria, references` (empty `success_criteria`, `references` list). This is in the F#502/F#646 cohort but does NOT invalidate this SUPPORTED verdict — KCs are well-formed target-metric pairs that PASSED on real measurement. F#666-pure-standalone preempt-clause does not apply (KCs are target-paired structurally). Hygiene defects matter when they enable structural unidentifiability; they do not here.
- **Per-layer dim heterogeneity** is a finding about the model, not a defect of this experiment. The IMPL correctly inferred dims per layer (`infer_per_layer_dims`) before adapter synthesis, sidestepping the parent's layer-0-uniform assumption.
- **Degenerate decoded text** in K1954 (`'<|"|>...'`) is expected — adapter-B is `N(0, 0.01²)` random, not a trained task. The KC tests *equivalence under detach/re-attach*, not output quality; the bitwise-identical degenerate sequence across baseline + 4 swap positions is the required signal.

## Verdict rationale

Both target-metric KCs pass with healthy margins (K1953: 103× margin below threshold; K1954: 0/64 token mismatches). Parent `exp_pierre_adapter_hotswap_latency` (provisional, F#702 design-lock) Theorems 1 and 2 are confirmed on Gemma 4 E4B 4-bit. The IMPL ships a local `attach_adapter`/`detach_adapters` that honor `model.layers` (Gemma 4 wrapper) without touching `pierre.pierre` (still correct for its Qwen3 deployment) — clean separation. First non-preempt-KILL drain-window outcome in the recent sequence (~34 preempt-KILLs preceded), validating the analyst's recommendation to claim target-anchored experiments with on-disk KC-target verification before claim.

## Routing

- `experiment finding-add` — record cross-architecture transfer of F#702 Theorems 1+2 + per-layer dim heterogeneity as a Gemma-4-adapter-synthesis caveat.
- Emit `review.proceed` with compact payload.
- LEARNINGS.md to be authored by analyst hat next iteration.
