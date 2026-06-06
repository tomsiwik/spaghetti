# REVIEW-adversarial.md — exp_followup_routing_multi_sample_ppl

## Verdict: **KILL** (endorse researcher preemptive kill)

Reviewer endorses the 12th P11-adjacent kill of the week: preemptive cascade
on antipattern-017 (5th confirmed instance) plus antipattern-020 (cascade-
dependent design). K1549 is also **closed by derivation** via MATH.md
Theorem 1.

## Adversarial checklist

**Consistency (a–d):** PASS.
- results.json `verdict=KILLED` ↔ DB `status=killed` ↔ PAPER.md "KILLED (preemptive, 2026-04-18)". No smoke flag, no mixed language.

**KC integrity (e–g):** PASS.
- (e) `git status` shows directory untracked (`??`); no KC-swap possible — MATH.md has a single state.
- (f) No tautology. Theorem 1 is an upper bound `P(identity) ≤ p^(N·D) < 1` for `p∈(0,1)`; identity is not forced by construction under per-sample routing.
- (g) K1549 text matches across DB (`experiment get`), MATH.md §"Kill Criteria", and results.json. Same quantity throughout.

**Code ↔ math (h–m2):** PASS.
- (h) run_experiment.py has no `sum(lora_A)` / `add_weighted_adapter` / independent safetensor summing — purely pre-flight gate at L25–31.
- (i) No `LORA_SCALE`.
- (j) No per-sample routing even attempted; no misuse.
- (k) No `shutil.copy`.
- (l) No hardcoded `{"pass": True}`.
- (m) No model loaded.
- (m2) N/A for preemptive kill (no MLX code runs). **For v2**: PLAN.md Part 2 requires `/mlx-dev` + `/fast-mlx` invocation when the actual PPL measurement is attempted.

**Eval integrity (n–q):** N/A (no run). F#560 not invoked; no absolute-threshold KCs.

**Deliverables (r–s):**
- (r) Prediction-vs-measurement table present in PAPER.md L21–24. ✓
- (s) Math audit:
  - Theorem 1 upper bound `p^(N·D)` is correct for i.i.d. routing. Assumption of i.i.d. is explicitly flagged in PAPER.md "Assumptions" (L60–62) with the correct observation that correlated errors would strengthen, not weaken, the bound.
  - Numerics verified: `0.85^250 ≈ 2.1e-18`, `0.99^250 ≈ 0.0811`, `0.996^250 ≈ 0.368`. ✓
  - Finite-precision caveat not material: rounding to 3 decimals could raise `P(visible-identity)` slightly above `p^(N·D)` but does not change the direction of the bound for the "not identical by construction" claim.

## Independent verification of kill drivers

1. **Adapter cascade**: `ls adapters/{math,bash,python,sql,medical}/` confirms 0 of 5 directories contain `adapters.safetensors`; each has only `adapter_config.json` + tokenizer metadata. Antipattern-017 triggers cleanly → **5th confirmed instance** (baseline_eval + J0 4-of-4 + M0 2-of-4 stub + 5-of-5 here = 5 total adapter dirs across repo now).
2. **Upstream cascade**: `experiment get` on all 4 Pierre siblings (`pierre_unified_pipeline`, `pierre_v3_sft_n5`, `pierre_v5_ternary_lora`, `pierre_v6_precomputed_concat`) returns `Status: killed`. Antipattern-020 triggers cleanly.
3. **Theorem 1 closure**: K1549 text "(not identical by construction)" is exactly what Theorem 1 proves is forbidden at finite router accuracy. The KC is settled on the theorem side; only the magnitude question ("by how much?") remains empirically open.

## Distinction from F#553

F#553 (supported) is about the **single-sample routing artifact**: `route(val[d][0])` applied to the full set forces identity by construction. This experiment's Theorem 1 generalizes to the **per-sample case** and proves the opposite direction: per-sample routing at finite accuracy **cannot** produce identity except at measure zero. The two are complementary results with no overlap.

→ Worth a distinct finding: "Per-sample routing forbids tautological PPL identity for `p<1`" as a theoretical corollary to F#553.

## Open threads for Analyst

- Promote antipattern-017 from "3 confirmed instances" → **"5 confirmed instances"** (M0 added 4th; this adds 5th). Update sibling-check pre-flight grep guidance.
- Consider adding a new finding for Theorem 1 (proposed title: "Per-sample routing at finite accuracy forbids tautological identity"). Distinct from F#553. See MATH.md §"Theorem 1".
- `exp_followup_competitive_gsm8k_200n` has the same `audit-2026-04-17` tag; likely 13th kill via same cascade; flag for reviewer queue.
- Pre-flight rule for all future Pierre-lineage consumers: `find adapters/ -name adapter_config.json | while read f; do d=$(dirname $f); [ -z "$(ls $d/*.safetensors 2>/dev/null)" ] && echo "STUB: $d"; done`

## Assumptions / Judgment calls

- Treated "identity to 3 decimal places" as equivalent to "identity by construction" for the K1549 bound; finite-precision could relax by at most one sample's worth of coincidence, which is still << 1 for D·N=250 samples. Non-material.
- Endorsed "theorem closes KC" framing; the alternative framing ("unmeasurable → pure cascade kill") is also defensible, but the researcher's combined framing is more informative and the theorem is non-trivial.

## DB actions

- No DB write needed: researcher already set `status=killed --k 1549:fail` with evidence.
- Finding-add recommendation deferred to Analyst (theorem warrants a distinct entry; not a reviewer-scope addition).
