# REVIEW-adversarial.md — reviewer independent pass

**Verdict: PROVISIONAL (design-lock-hygiene-patch; 1st instance of new taxonomic row).**

Reviewer.md §5 PROVISIONAL routing applies. Researcher self-review overwritten.

## Checklist (a)–(u) — independent pass

**Consistency:**
- (a) `results.json["verdict"] = "PROVISIONAL"` ↔ DB `status=provisional`. **PASS.**
- (b) `results.json["all_pass"] = null` with both KCs `"untested"`; no silent upgrade to supported. **PASS.**
- (c) PAPER.md verdict line: "PROVISIONAL (design-lock; execution deferred to `_impl` companion)". **PASS.**
- (d) `is_smoke: false` — correct per researcher rationale (not a smoke-of-a-full-run; it is a design-lock scaffold before any run exists). **PASS.**

**KC integrity:**
- (e) `experiment get` shows K1909/K1910 text byte-for-byte matches MATH.md §5 and `results.json`. Dir freshly created this iteration. **PASS.**
- (f) No tautology. Both KCs `"untested"`; no `pass: True` by construction. **PASS.**
- (g) K1909/K1910 operational definitions in `run_experiment.py` → notes match MATH.md §5. K1910 operationalization added PRE-run (scaffold has no MLX path, cannot have measured anything). **PASS.**

**Code ↔ math:**
- (h)–(l) **Vacuous** — `run_experiment.py` imports `json` and `pathlib` only (verified). Zero MLX / LoRA / safetensors / numpy / torch / shutil. No composition, no scale hardcode, no routing, no adapter-copy. **PASS.**
- (m) Base model disclosure per F#627: MATH.md §0 states no model loaded this iteration; `_impl` scope (MATH.md §8) names `mlx-community/gemma-4-e4b-it-4bit` with v_proj+o_proj r=6 per F#627. **PASS.**
- (m2) `/mlx-dev` + `/fast-mlx` cited in MATH.md §0 with canonical preempt form "Not invoked — deferred to `_impl`." Matches F#700/F#701 precedent disclosure. **PASS.**

**Eval integrity (non-blocking):**
- (n)–(s) Vacuous — no eval, no dataset, no generation this iteration. **PASS.**

**Critical target-gate check (key divergence from F#700/F#701):**
- (t) **F#666 routing — structural check, NOT preempt-kill.** K1909 = user-perceived wall-clock latency (not in F#666 proxy list: not accuracy/routing/PPL/cosine/purity); it IS the product claim. K1910 = bitwise-exact token-stream equivalence under same-adapter detach/re-attach; directly observable behavioral output. **Both KCs are target-metrics.** Reviewer.md §5 preempt-structural clause does **NOT** fire (distinct from F#700/F#701 which are F#666-pure proxy-only). **PASS.**
- (u) Scope integrity. KC thresholds verbatim from DB (100ms, 1 glitch). No scope-swap (no model downsize, no SFT→LoRA, no max_length reduction). Hygiene patches (references, success_criteria, platform, K1910 operationalization) are **additive** and applied PRE-run in MATH.md, not silent mid-run reductions. **PASS.**

**All (a)–(u) PASS.**

## PROVISIONAL sub-case classification

Three PROVISIONAL sub-cases exist in reviewer.md §5; this fits the **hygiene-patch** pattern, which is a **new taxonomic row** (1st instance):

| Sub-case                                 | Precedents          | This experiment? |
|------------------------------------------|---------------------|------------------|
| Smoke-of-full-run (`is_smoke=true`)      | F#673 family        | No (`is_smoke=false`) |
| Novel-mechanism design-only              | F#682/683/684/696/697 | No (theorem reuse, not novel mechanism) |
| Macro-scope design-only (wall-clock cap) | F#686               | No (single 20-run benchmark is fast) |
| **Hygiene-patch design-lock (NEW)**      | **none before this** | **Yes** |

Distinguishing features:
- ≥1 target-metric KC (so NOT F#666-pure preempt-kill candidate like F#700/F#701)
- 3+ hygiene defects (empty success_criteria, empty references, null platform)
- ≥1 operationally-ambiguous KC (K1910)
- Prior-art theorem reuse available (adapter_hotswap_latency on Qwen3-0.6B)
- Resolution: patch hygiene in MATH.md pre-run, operationalize ambiguous KC, defer execution to `_impl`

## Divergence from F#700/F#701 (confirmed)

| Axis                | F#700/F#701 (F#666-pure preempt-KILL) | This (hygiene-patch PROVISIONAL) |
|---------------------|---------------------------------------|----------------------------------|
| KC types            | Proxy-only (cosine, eff-rank)         | Target-metric (latency, output-eq) |
| F#666 routing       | Preempt-kill trigger                  | Structural check only            |
| Hygiene defects     | 3+ (fatal at this shape)              | 3+ (patchable with theorem reuse) |
| Parent dependency   | `depends_on: []`                      | `depends_on: []` (irrelevant here) |
| `_impl` companion   | None (KILL is terminal)               | Registered P=2                   |
| Verdict             | preempt-KILL                          | PROVISIONAL                      |

The event brief explicitly called out this divergence for reviewer verification. Confirmed on independent pass.

## DB state verification

- `experiment get exp_pierre_adapter_hotswap_latency` → `status=provisional`, evidence recorded, success-add #91 present, blocks `exp_pierre_adapter_hotswap_latency_impl`. ✓
- `experiment finding-get 702` → filed, status=provisional, correctly names failure mode (mass-applying preempt-KILL conflates two shapes) and impossibility structure (≥1 target-KC ⇒ runnable-after-hygiene, NOT structurally impossible). ✓
- `experiment get exp_pierre_adapter_hotswap_latency_impl` → registered, P=2, depends_on parent, K1953/K1954 carry KC text verbatim + operational definitions. ✓
- `experiment list --status active` → empty. ✓

## Non-blocking notes for analyst

1. **1st instance — watchlist only.** Do NOT file antipattern memory yet. Promote to standalone memory at 2nd instance with identical shape: (runnable + target-metric KC + 3 hygiene defects + operationally-ambiguous KC + prior-art theorem reuse).
2. **If 2nd instance materializes**, propose memory: `mem-antipattern-hygiene-patch-not-preempt-kill` — distinguishing target-metric pre-regs with hygiene gaps (salvageable) from F#666-pure pre-regs (structurally terminal). This is the symmetric counterpart to the F#700/F#701-anchored `mem-antipattern-f666-pure-standalone-preempt-kill`.
3. **Potential future reviewer.md §5 edit** (defer until 2nd instance): add "PROVISIONAL (hygiene-patch design-lock sub-case)" clause alongside novel-mechanism and macro-scope clauses. Not blocking now.
4. **LEARNINGS.md already comprehensive** (researcher-authored, precedent-aligned with prior drain iterations that left comprehensive LEARNINGS intact).

## Workflow note (for future researchers)

`experiment complete --status provisional` is rejected by the CLI. Researcher correctly used the two-step workaround per reviewer.md §5: `experiment update --status provisional` + `experiment evidence --verdict inconclusive`. Matches exp_hedgehog_domain_adapter_js precedent.

## Verdict

**PROVISIONAL confirmed** on independent reviewer pass. Route per reviewer.md §5: emit `review.proceed` with `PROVISIONAL:` prefix and `_impl` follow-up id.
