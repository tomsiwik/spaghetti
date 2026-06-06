# MATH.md — exp_memento_kv_serialization_format

## §0 Platform skills invoked

Preempt-structural KILL. No platform code executed; run_experiment.py is a graceful-failure stub (`json` + `pathlib` only). Skills `/mlx-dev` + `/fast-mlx` cited here per reviewer item (m2) carve-out convention used across F#700–F#714 preempt-KILL precedent; no MLX API is exercised.

## §1 Claim under review

> KV-state serialization + deserialization round-trip latency and serialized size are feasible for cross-session persistence on MLX.

Operationalised as two kill criteria (DB-verbatim):

- **K1860**: round-trip serialization + deserialization latency > 100ms for 2048-token context
- **K1861**: serialized KV state > 5MB per 2048 tokens (too large for user-space storage)

Both KCs are pure infrastructure measurements (wall-clock latency, byte-size-on-disk). No KC measures downstream task accuracy, behavioral quality, user-perceived persistence benefit, or any behavioral proxy.

`depends_on = []` (standalone; no parent finding to inherit from). Hygiene: `success_criteria = []`, `platform = ~`, `references = []` — 3 defects (≥ F#703 threshold).

## §2 Preempt-structural verdict (F#666-pure-standalone, 11th drain-window instance)

### L1. Both KCs are proxy-only under Finding #666 / canonical guardrail #1007

Guardrail 1007: *every proxy KC must be paired with a target-metric KC (task accuracy, behavioral quality, oracle-gap). KILL requires BOTH to fail; SUPPORTED requires BOTH to pass.*

K1860 and K1861 are infrastructure measurements. The thresholds (100ms, 5MB) are BEHAVIORALLY UNCALIBRATED — no evidence is provided (or derivable within this experiment's scope) that:

- 101ms round-trip degrades user-perceived persistence experience vs 99ms
- 5.1MB storage is infeasible vs 4.9MB at any actual deployment constraint
- the 2048-token context window is the right unit for a behavioral claim

A proper target-gated formulation would instead ask: *at what latency/size does cross-session persistence stop providing measurable behavioral benefit vs no-persistence baseline on a downstream task (e.g. multi-turn conversation recall accuracy, task completion rate)?* That question requires a paired target KC which this experiment does not register.

Per F#666: PASS on K1860 + K1861 is tautological support (the thresholds are the definition); FAIL is a finding about the thresholds, not about whether serialization is behaviorally useful.

### L2. Measurement-bucket classification — 6th bucket (novel)

Post-F#711 taxonomy refactor, F#666-pure-standalone occurrences span:

1. derived-geometric (cos-sim, eff-rank, pairwise-cos, worst/mean) — F#700, F#701, F#711, F#714-K1854
2. detection/classification (canary FNR, training-time Epps-Pulley) — F#706, F#714-K1855
3. routing (match-rate, xxhash) — F#707
4. PPL (perplexity) — F#708
5. content-based similarity (template-match, semantic) — F#703, F#705, F#710

K1860 (round-trip latency in ms) and K1861 (serialized size in MB) are **infrastructure-benchmark bucket** (wall-clock latency, byte-size, memory footprint, I/O throughput). This is a 6th measurement bucket, novel in drain window. Distinction from prior buckets: infrastructure metrics measure properties of the serialization *procedure* itself, not properties of model output or adapter geometry.

Like F#714 (first multi-bucket fire combining 2 existing buckets), this experiment introduces an entirely new bucket rather than combining existing ones. Taxonomy absorbs cleanly: *any metric X such that X is not the behavioral outcome of interest* is proxy-only under F#666, regardless of what bucket X lives in. Bucket labels are curatorial aids, not gating criteria.

### L3. F#666 truth-table inadmissibility

For each KC k ∈ {K1860, K1861}:

| outcome | interpretation | admissible verdict? |
|---|---|---|
| k PASS (latency ≤ 100ms and size ≤ 5MB) | tautological-support: "serialization met the arbitrarily chosen numerical bounds" | no (no behavioral claim verified) |
| k FAIL (latency > 100ms or size > 5MB) | finding-about-the-threshold, not about behavioral feasibility of cross-session persistence | no (kill target mis-identified) |

Both outcomes yield inadmissible verdicts under F#666. The experiment is structurally unable to produce a behaviorally-grounded KILL or SUPPORTED regardless of measurement, which is the F#666-pure signature.

### L4. Hygiene-multi-defect secondary fire (F#703 canonical)

DB record shows:

- `success_criteria: []` (empty)
- `platform: ~` (null)
- `references: []` (empty)

3 defects ≥ F#703 canonical threshold. Normally this would route via F#702 hygiene-patch PROVISIONAL path. **F#702 hygiene-patch is structurally unavailable here** because F#702 requires ≥ 1 target-KC to patch; with zero target KCs (both are proxy infrastructure metrics), no patch surface exists.

Same impossibility-structure as F#714: when F#666-pure fires on 100% of KCs, both §5-patch and F#702-patch paths collapse simultaneously. Hygiene fires as secondary annotation, not as primary routing target.

### L5. Standalone topology — distinctions from all other antipatterns

- **Not F#669-family** (parent-unverified reuse): `depends_on = []`; no parent finding exists to inherit from.
- **Not §5 tautological-inter-variant-delta**: no comparison between adapter variants, training methods, or routing configurations registered in KCs.
- **Not template-regression**: no parent experiment stripped to yield this one; fresh hypothesis with no per-variant base-anchor removed.
- **Not proxy-only-lineage-inheritance**: no parent finding to inherit proxy-only structure from.
- **Not cross-paper-combined-loss-tautology** (F#714 watchlist, 1st-instance): no composite loss `L = L_A + λ · L_B` combining two papers' methods.

Clean F#666-pure application: 11th drain-window instance, 1st standalone infrastructure-benchmark bucket, 2nd double-fire with hygiene-multi-defect secondary (1st was F#703; F#714 was triple-fire including §5, this is double-fire with no §5).

## §3 Predictions (structural, not measured)

This experiment emits no measurements; the verdict is derivable pre-execution from the KC set topology. PAPER.md logs the prediction-vs-measurement table with `measurement=not_measured` for every prediction. The single empirical claim made is: *were the experiment run, the resulting latency and size numbers would be incapable of certifying or falsifying cross-session persistence feasibility, because the feasibility claim requires behavioral evidence not instrumented by K1860/K1861.* This claim is proven in L1–L3 and is not contingent on measurement.

## §4 Unblock path (v2 design sketch)

A behaviorally-grounded `exp_memento_kv_serialization_format_v2` would register:

- **K_target_1**: multi-turn conversation recall accuracy with persistence enabled vs no-persistence baseline, on a held-out dialogue benchmark (e.g. MT-Bench multi-turn subset), measured at the 2048-token context boundary.
- **K_proxy_1** (paired): serialized size per 2048 tokens, with threshold calibrated from the accuracy/size Pareto curve measured under K_target_1.
- **K_proxy_2** (paired): round-trip latency, threshold calibrated similarly.

Re-claimable when this target+proxy pairing is specified AND `success_criteria`, `platform`, `references` populated.

## §5 Authoritative references

- Finding #666 (proxy-only target-gating; primary antipattern)
- Canonical guardrail #1007 (target-gated KILL requirement)
- Finding #703 (hygiene-multi-defect canonical 3+ defect threshold)
- Finding #702 (hygiene-patch PROVISIONAL path requires ≥ 1 target KC)
- Finding #714 (double-fire precedent; multi-bucket / secondary-fire annotation convention; F#702 unavailability under 0 target KCs)
- Finding #711 (taxonomy-refactor; buckets are curatorial, not gating)
- MEMENTO (Kontonis arxiv:2604.09852) — context for cross-session persistence research line; not cited as mechanism support here since no MEMENTO mechanism is under test by K1860/K1861.

## §6 Assumptions (autonomy)

- Classified as F#666-pure-standalone (primary) + hygiene-multi-defect (secondary) per pre-claim checklist hierarchy from F#714 (8th item: KC class > KC form > metadata; F#666-pure dominates when 0 target KCs).
- `depends_on = []` is authoritative; no implicit parent via MEMENTO lineage (F#699 is sibling, not parent; compression-ratio axis ≠ serialization-format axis).
- Latency/size measurements are classified as proxy (not target) because the experiment notes frame the benchmark as "prerequisite for cross-session persistence" — persistence is the behavioral outcome, latency/size are engineering proxies for it. Alternative interpretation (pure infrastructure benchmark where latency/size ARE the target) rejected: thresholds are behaviorally uncalibrated and the notes explicitly reference behavioral use-case.
