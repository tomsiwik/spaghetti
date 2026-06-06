# REVIEW-adversarial.md — exp_hedgehog_adapter_sql_domain

**Verdict: PROVISIONAL** (novel-mechanism design-only sub-case per reviewer.md clause +
F#702 hygiene-patch-secondary pairing; 6th Hedgehog-axis, 4th/closing domain-axis).

## Routing rationale

Applies the **PROVISIONAL (novel-mechanism design-only sub-case)** clause. All 4
required-artifact-pattern items present:

1. **MATH.md §0** cites `/mlx-dev` + `/fast-mlx` with the exact responsibilities they
   satisfy (`mx.eval` discipline, `mx.clear_cache` between phases,
   `nn.value_and_grad` functional gradients, compile/lazy/bandwidth kernels). Satisfies
   (m2) carve-out without MLX training-loop code landing this iteration.
2. **run_experiment.py** `main()` never raises: 1.6 s runtime, 5 structured blockers
   captured under `NotImplementedError` catch per phase, `verdict="PROVISIONAL"`,
   `all_pass=false`, both KCs `"untested"`. Writes `results.json` even when every
   phase fails.
3. **_impl follow-up filed**: `exp_hedgehog_adapter_sql_domain_impl` exists at P=3,
   KC-IDs #1957 / #1958 text-inherited verbatim from parent #1868 / #1869
   (F#666-compliant pair — PPL proxy + query-correctness target).
4. **PAPER.md prediction-vs-measurement table** present (lines 42–49); both rows "not
   measured"; explicit scope rationale ("Scope (this iteration)" + "Measurement
   blockers" sections); 5 explicit scope-preservation rejections (teacher proxy,
   CE swap, baseline skip, silent N_STEPS reduction, EXPLAIN hard-floor drop).

## Adversarial checklist

**Consistency (a–d):** `results.json["verdict"]="PROVISIONAL"` matches DB `status=provisional` and PAPER.md line 3. `all_pass=false` consistent with both KCs `"untested"`. `is_smoke=false` but no full-run claim — design-only.

**KC integrity (e–g):** K1868/K1869 match DB canonical text verbatim (verified via `experiment get`). No post-hoc KC mutation. No tautology: K1868 is head-to-head `PPL(Hedgehog) vs PPL(base+generic LoRA)` (different configs, non-trivial); K1869 is judge `Δ(Hedgehog − base)` with PostgreSQL `EXPLAIN` parse-and-plan hard-floor. Both `result_flags` map cleanly onto measured quantities.

**Code↔math (h–m2):** No composition bugs. `LORA_SCALE=6.0` ≤ 8 per F#328/F#330. No routing. No `shutil.copy` of sibling adapter. No hardcoded `pass` dicts. Student `mlx-community/gemma-4-e4b-it-4bit` in `run_experiment.py` matches MATH.md §0. Teacher `mlx-community/gemma-4-26b-a4b-it-4bit` is a larger Gemma 4 variant, not a proxy substitution. (m2) satisfied by §0 skill citations.

**Eval integrity (n–u):** No base-accuracy / thinking-suppression risk (nothing measured). No statistical claim (N/A). No synthetic padding. (t) Target-gated kill N/A — both KCs untested; K1869 is the behavioral target if ever measured. (u) Scope-preservation affirmed by explicit rejections of teacher proxy / CE swap / baseline skip / silent downscale / `EXPLAIN` hard-floor drop.

**Deliverables (r–s):** Prediction-vs-measurement table present. Math derivation chain (MATH.md §3) coherent; K1869 prediction of Δ∈[+4,+9] pp mean +6 pp is consistent with Rust sibling (F#717) at the same rank and structural difficulty; rationale for equal-Python-distinct-Rust/SQL is load-bearing, not noise.

## F#702 hygiene-patch classification (3rd instance)

- Applicable (target KC K1869 present — `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does NOT fire).
- DB patch applied pre-`experiment complete`: `platform=local-apple`, `success_criteria` #93 added. `references` remains INCOMPLETE matching prior F#702 precedents (CLI global-ref-library linkage not experiment-row scoped).
- **Pairing classification**: `novel-mechanism-primary + hygiene-patch-secondary` (same as F#717 Rust). This is the **2nd instance of that same-pairing** (1st = F#717). Mixed-pairing watchlist: confirmed-recurrent pending 3rd same-pairing instance for sub-classification promotion.

## Distinctions confirmed clean

- **NOT F#666-pure**: target KC K1869 present.
- **NOT F#669-family**: `depends_on=[]`.
- **NOT §5 tautological-inter-variant-delta**: K1868 includes base + generic LoRA anchor; not inter-variant-only.
- **NOT template-regression**: axis-content (declarative plan-cost reasoning + dual syntactic+semantic `EXPLAIN` ground truth) structurally distinct from imperative siblings; not repetition.
- **NOT proxy-only-lineage-inheritance**: target KC present.

## Novel observations (non-blocking)

- **Dual syntactic+semantic ground-truth** (PostgreSQL `EXPLAIN` parse-and-plan) is stricter than Rust sibling's single `cargo check`. First dual-ground-truth judge hard-floor in Hedgehog sub-family.
- **Axis-content novelty** is genuine: declarative SQL is structurally distinct from imperative JS/Python/Rust, closing the domain-axis sub-family at 4 structurally-distinct instances.
- **Analyst guidance in scratchpad "6-axis saturation"**: researcher's A10 note and PAPER.md "Sibling-axis position" acknowledge the forward-applicable deferral; no further Hedgehog-axis design-locks should be claimed until at least one `_impl` lands.

## Assumptions (judgment calls)

- Accepted `references` INCOMPLETE flag non-blocking (matches Pierre F#702 / Rust F#717 precedent — CLI limitation, not researcher hygiene fault).
- Accepted equal Rust/SQL +6 pp K1869 prediction as load-bearing reasoning (both require non-surface abstract-structure inference) rather than analytical noise.

## Verdict

**PROVISIONAL (novel-mechanism design-only + F#702 hygiene-patch-secondary, 3rd F#702 instance, 2nd same-pairing as F#717).** Route via two-step workaround (already executed by researcher — DB `status=provisional`, evidence filed, F#718 filed). Emit `review.proceed` with `PROVISIONAL:` prefix + `_impl` follow-up ID.
