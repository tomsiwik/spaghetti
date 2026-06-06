# REVIEW-adversarial.md — exp_hedgehog_adapter_rust_domain

**Verdict: PROVISIONAL** — novel-mechanism design-only sub-case (canonical per F#683/F#684/F#696/F#697 precedent) + F#702 hygiene-patch-secondary (2nd instance after F#702-primary at `exp_pierre_adapter_hotswap_latency`).

## Adversarial checklist (a)–(u)

**Consistency**
- (a) `results.json["verdict"]="PROVISIONAL"` ↔ DB status=`provisional` ✓
- (b) `all_pass=false`; both KCs `"untested"` — consistent with design-only ✓
- (c) PAPER.md verdict line "PROVISIONAL (design-only; KCs K1866/K1867 untested…)" ✓
- (d) `is_smoke=false` but PROVISIONAL is the novel-mechanism design-only classification (not silently-upgraded from smoke) ✓

**KC integrity**
- (e) `experiment get` KC text matches MATH.md §4 and `run_experiment.py` module docstring verbatim (K1866 "PPL > base + generic LoRA", K1867 "idiomaticity judge < +5pp vs base"). No post-claim mutation ✓
- (f) No tautology: K1866 pits Hedgehog-adapter PPL against base+generic-LoRA PPL on held-out external crates.io corpus (disjoint from Rust Book/nomicon training); K1867 blind-paired judge with `cargo check` compile-check hard-floor ✓
- (g) `results.json.kc` IDs match DB (K1866/K1867) ✓

**Code ↔ math**
- (h) No `sum(lora_A)` / `add_weighted_adapter(linear)` / per-key summation — no composition code landed this iteration ✓
- (i) `LORA_SCALE=6.0` < 8 per F#328/F#330 ✓
- (j) No per-single-sample routing applied to all ✓
- (k) No `shutil.copy` as new-domain adapter ✓
- (l) No hardcoded `{"pass": True}` — KCs explicitly `"untested"` ✓
- (m) `STUDENT_MODEL=mlx-community/gemma-4-e4b-it-4bit`, `TEACHER_MODEL=mlx-community/gemma-4-26b-a4b-it-4bit` — match MATH.md §0 verbatim ✓
- (m2) **Skill-invocation evidence**: MATH.md §0 cites `/mlx-dev` + `/fast-mlx` skills as required before `_impl` training loop lands; no unidiomatic MLX training-loop code in this scaffold (design-only) — carve-out applies per novel-mechanism PROVISIONAL clause ✓

**Eval / scope**
- (n)–(q) N/A — no eval ran; design-only
- (r) PAPER.md prediction-vs-measurement table present with all rows "not measured" + explicit scope rationale ✓
- (s) Math traceable: Moudgil 2604.14191 §3.1 eq. 6 (Hedgehog); Zhang 2402.04347 (cos-sim sufficiency); MultiPL-E 2208.08227 (Rust support); Pierre F#627 (v_proj+o_proj sufficiency); F#614/F#536 (thinking-mode load-bearing); F#673 (mx.clear_cache discipline); sibling Hedgehog-axis F#683/F#684/F#696/F#697 ✓
- (t) Target-gated (F#666): K1866 is proxy, K1867 is behavioral target (blind-paired judge + `cargo check` compile-floor) — pair is well-formed; no proxy-only-FAIL kill being claimed (verdict is PROVISIONAL) ✓
- (u) Scope-preservation: PAPER.md §"Scope-preservation explicit rejections" enumerates 5 forbidden silent downscales (teacher-proxy, CE-swap, baseline-skip, N-reduction, `cargo check` drop); scaffold raises `NotImplementedError` in all 5 phases rather than silently substituting — canonical honest design-only filing ✓

## Novel-mechanism PROVISIONAL clause compliance

Per reviewer.md:
1. MATH.md §0 cites required platform skills — ✓
2. `run_experiment.py main()` never raises, always writes `results.json` with `verdict="PROVISIONAL"` + KCs `"untested"` — ✓ (scaffold ran 2.2 s, 5 structured blockers, zero errors)
3. `_impl` follow-up `exp_hedgehog_adapter_rust_domain_impl` filed at P=3 with KCs 1955/1956 text-inheriting 1866/1867; `depends_on=[exp_hedgehog_adapter_rust_domain]` ✓
4. PAPER.md prediction-vs-measurement table with all "not measured" + scope rationale + explicit sibling-axis positioning (5th Hedgehog-axis; 3rd domain-axis) ✓

(m2)/(u) do not block — they target silent mechanism swaps, not honest design-only filings.

## F#702 hygiene-patch-secondary

DB row shipped with 3 defects (`success_criteria=[]`, `platform=~`, `references=[]`). F#702 hygiene-patch path is AVAILABLE because K1867 is a target KC (not F#666-pure, so `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does not fire). Researcher patched:
- `platform` → `local-apple` ✓ (DB reflects)
- `success_criteria` → success #92 added ✓ (DB reflects)
- `references` → still `INCOMPLETE` in DB ⚠ — matches F#702-primary precedent (global ref library not directly linkable per current CLI); non-blocking per convention

This is the 2nd F#702 instance, and first in a **novel-mechanism-primary + hygiene-patch-secondary** pairing (F#702-primary was hygiene-patch-primary). Watchlist for sub-classification promotion at 3rd mixed-pairing instance per analyst note.

## Distinctions

- NOT F#666-pure (K1867 is behavioral target)
- NOT F#669-family (`depends_on=[]`)
- NOT §5 tautological-inter-variant-delta (K1866 has base + base+generic-LoRA anchors)
- NOT template-regression (axis-extension, not structural repetition; Rust borrow-checker reasoning is structurally distinct from JS/Python siblings — `cargo check` hard-floor is load-bearing novelty)
- NOT proxy-only-lineage-inheritance (target KC present)

## Assumptions (reviewer)

- A1. 5 Hedgehog-axis PROVISIONALs without any `_impl` measurement is flagged (researcher self-review item 1) but does NOT block PROVISIONAL under the canonical clause. The impossibility-structure (custom MLX loop + 26B teacher cache) is stable across 5 instances; deferring further Hedgehog-axis novel-axis PROVISIONALs until at least one `_impl` lands is a defensible future constraint but not a reviewer-block on this filing.
- A2. `cargo check` toolchain availability on eval machine is an `_impl`-time concern, not a design defect; A8 in MATH.md documents fallback to judge-assessed borrow-checker correctness.
- A3. 2-KC count (matching Python sibling F#697; JS sibling F#696 has 4) is a pre-reg choice documented in MATH.md §A7; non-interference/general-NL gated at composition child.

## Routing

PROVISIONAL (novel-mechanism design-only). `experiment complete --status provisional` is NOT used (CLI rejects `provisional`). DB already at status=`provisional` via researcher two-step workaround; evidence filed; F#717 filed (verified via `finding-list --status provisional`). Emitting `review.proceed` with `PROVISIONAL:` prefix + `_impl` ID per reviewer.md PROVISIONAL routing.
