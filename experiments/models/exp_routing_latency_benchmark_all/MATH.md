# MATH.md — exp_routing_latency_benchmark_all (PREEMPT-KILL, F#666-pure standalone, ≥23rd drain-window instance)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — multi-bucket: routing-accuracy 3rd sub-flavor + infrastructure-benchmark 2nd sub-flavor)

This experiment is preempt-killed before any code runs. The kill is **structural**: the pre-registered kill-criterion set K = {K1929, K1930} consists of two proxy metrics (latency threshold + routing classification accuracy) with no paired target-metric KC. Under F#666 (guardrail 1007 — target-gated KILL discipline) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is a continuation of the F#666-pure standalone canonical pattern (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#722, F#728, F#729, F#730, F#731, F#732, F#734 — at least 22 prior). Specifically:
- **3rd routing-accuracy sub-flavor**: K1930 is "best routing method accuracy < 80% at N=25" — the canonical F#666-forbidden routing-match-rate proxy (1st: F#703 K1569 TF-IDF acc; 2nd: F#710 K1591 Gumbel acc).
- **2nd infrastructure-benchmark sub-flavor**: K1929 is "any routing method > 10ms per query" — pure engineering-latency threshold with no behavioral coupling (1st: F#734 K-component).

## §0 Platform / skills / model pins

Included for reviewer checklist (m2) completeness. No platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — routing-method benchmark over a fixed N=25 adapter set; classifier comparison, no LoRA injection in this run.
- Parent dependency: **none** (`depends_on: []`). NOT an F#669 preempt.

## §1 Preempt-KILL theorem (F#666-pure, multi-bucket)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_routing_latency_benchmark_all` with kill-criterion set K = {K1929, K1930}:
- K1929 := "Any routing method > 10ms per query (too slow for real-time)"
- K1930 := "Best routing method accuracy < 80% at N=25"

**Classification of K.**
- K1929 is a **proxy metric** — pure latency threshold with no link to whether the chosen routing produces a behaviorally correct adapter selection. F#710 and F#740 precedents establish that "engineering-target-only" KCs are proxies under F#666 because PASS-condition (low latency) is achievable trivially by random/constant routing that ignores the input.
- K1930 is explicitly a **proxy metric** — F#666 guardrail 1007 enumerates *classification accuracy* and *routing match rate* by name as forbidden-solo proxies. "Best routing method accuracy" is precisely a routing match rate over the N=25 adapter set. F#666 canonical: routing accuracy 40.2% with 0.0% target gap demonstrates routing-match-rate is decoupled from downstream utility.

Neither KC measures task accuracy, behavioral quality, oracle-gap, or any downstream-behavioral outcome. K is a 2-proxy, 0-target set.

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological. Per F#714 precedent (multi-proxy F#666-pure), the analysis is per-KC then composed:

| K1929 | K1930 | V(K) under F#666                                                                              |
| ----- | ----- | --------------------------------------------------------------------------------------------- |
| PASS  | PASS  | Tautological SUPPORT — both proxies, no target. Routing could be fast and accurate per match-rate yet still select adapters whose output is no better than base (F#666 canonical 40.2% + 0.0% gap). |
| PASS  | FAIL  | Mixed proxy outcome; F#666 rule "Proxy-FAIL + target-absent = finding about the proxy, not a kill" — produces a finding about routing-accuracy threshold, not a behavioral kill. |
| FAIL  | PASS  | Mixed proxy outcome; same finding-not-kill rule applies on the latency proxy. |
| FAIL  | FAIL  | Both-fail proxy-only — under F#666 still "finding about proxies, not kill" because no target was measured. |

**No cell yields a valid F#666-compliant verdict.** K is unidentifiable at the F#666 layer. **QED.**

### §1.1 Latency-only is a proxy (sub-flavor justification)

Why is K1929 (latency) not a target? F#702 (the canonical latency-as-target precedent) paired latency with **bitwise-exact token equivalence** — i.e. the latency claim was anchored to a behavioral correctness invariant (output identical to merged-weights baseline). F#702 was runnable because the KC set was {latency, output-equivalence} = {proxy, target}. K1929 alone has no behavioral anchor. Past instances of standalone-latency KCs (F#734 K1903 component, F#740 Pierre-serving children) are F#666-pure preempts.

A routing method can pass K1929 (≤10ms) by routing to a fixed adapter regardless of input — fast but useless. Without a behavioral target metric, K1929 cannot distinguish utility from speed.

### §1.2 Routing-accuracy is the canonical F#666 forbidden proxy (3rd sub-flavor instance)

K1930 ("Best routing method accuracy < 80% at N=25") is a direct match for guardrail 1007's enumeration ("classification accuracy, routing match rate"). F#703 (1st routing-acc instance) and F#710 (2nd, confirmed-recurrent) have established this as a canonical F#666-violation pattern. This experiment is the 3rd routing-acc sub-flavor instance.

F#666 canonical counter-example: softmax router achieved gamma_top1 = gamma_oracle (0.0% target gap) at 40.2% per-sample classification accuracy. Match-rate is decoupled from utility via semantic-cluster routing. Therefore "best method ≥80% acc" can be PASS without behavioral lift, or FAIL with full behavioral lift; both outcomes are uninformative without a paired target.

## §2 Prior art (preempt-KILL precedents)

- **F#666** (2026-04-19, conclusive): target-gated KILL discipline; guardrail 1007 enumerates classification accuracy & routing match rate as forbidden-solo proxies; canonical 40.2% proxy + 0.0% target gap shows decoupling.
- **F#700** (2026-04-24): 1st F#666-pure standalone preempt-KILL — `exp_g4_per_layer_cos_baseline`, K1856 cos-sim variance.
- **F#701** (2026-04-24): 2nd F#666-pure standalone — `exp_adapter_orthogonality_audit`. Promotion trigger reached.
- **F#703** (2026-04-24): 3rd F#666-pure, **1st routing-accuracy sub-flavor** — `exp_followup_tfidf_medical_unaliased`, K1569 TF-IDF routing weighted accuracy.
- **F#705** (2026-04-24): 4th F#666-pure, PPL-only sub-flavor.
- **F#706** (2026-04-24): 5th F#666-pure, FNR-as-proxy canonical.
- **F#710** (2026-04-24): 8th F#666-pure, **2nd routing-accuracy sub-flavor** (confirmed-recurrent) — `exp_g4_gumbel_top2_n50`, K1591 routing acc ≥85%. Established cross-architecture transfer claim does not rescue F#666.
- **F#714** (2026-04-24): 10th F#666-pure, **first multi-bucket fire** (derived-geometric + detection); first triple-fire precedent.
- **F#722, F#728, F#729, F#730** (2026-04-24/25): triple-fire preempt-KILLs (F#666-pure + §5 + others).
- **F#734** (2026-04-25): 22nd F#666-pure, **quadruple-fire**, infrastructure-benchmark sub-flavor first identified.
- **`mem-antipattern-f666-pure-standalone-preempt-kill`** (filed 2026-04-24, escalated multiple times): claim-time detection rule; preempt-scaffold response.
- **Guardrail 1007** (PLAN.md): every proxy KC must be paired with a target-metric KC.
- **F#702**: latency-with-bitwise-equivalence-pair example of a runnable latency KC (illustrates the missing pair here).

## §3 Predictions (registered, not measured)

| KC    | Claim                                                          | Kind  | Sub-flavor               | Measurement status                |
| ----- | -------------------------------------------------------------- | ----- | ------------------------ | --------------------------------- |
| K1929 | Any routing method > 10ms per query                            | proxy | infrastructure-benchmark | untested (preempt-blocked)        |
| K1930 | Best routing method accuracy < 80% at N=25                     | proxy | routing-match-rate (3rd) | untested (preempt-blocked)        |

No target-metric KC exists. K is structurally malformed per F#666.

KC text preserved verbatim from `experiment get` output. No post-claim KC mutation (antipattern-u check: PASS).

## §4 Hygiene defects (noted, not load-bearing for kill)

Per `experiment get exp_routing_latency_benchmark_all`:

1. **`success_criteria: []`** (MISSING per DB output) — empty; no SUPPORTED-condition declared.
2. **`platform: ~`** (null) — guardrail/hygiene defect.
3. **`references: []`** — guardrail 1002 violation (every new experiment MUST cite an arxiv paper or prior finding).

Three hygiene defects total. Crosses the AP-prereg-hygiene-multi-defect threshold (≥3 defects). However, F#666-pure structural defect alone is sufficient for kill independent of hygiene count (per F#703 invariant).

Notes field reads "TF-IDF, xxHash, Gumbel-top2, semantic, signature — all head-to-head on same N=25 adapter set. Winner takes the default." This describes a benchmark methodology but specifies no behavioral outcome metric (only "winner" by routing match-rate proxy).

## §5 Unblock condition (re-claim requires KC-augmentation pre-registration)

Re-registration as a new experiment id (`exp_routing_latency_benchmark_all_behavioral` recommended) with the following fixes:

1. **Add a target-metric KC** pairing routing accuracy to a behavioral outcome. Candidate formulations:
   - **End-to-end task accuracy gap**: For each routing method, measure MMLU-Pro subject-domain accuracy with method-selected-adapter vs oracle-selected-adapter. Target KC: best method's task-accuracy gap ≤ 3pp at N=25. This couples routing acc to downstream utility.
   - **Confidence × quality correlation**: Spearman |r| ≥ 0.4 between per-sample routing confidence and generation-quality-delta vs base. Ties proxy to behavioral anchor.
   - **Pareto-quality**: Among methods with latency ≤ 10ms (K1929 PASS), best method's downstream task accuracy ≥ baseline + 5pp. Latency becomes a *constraint*, not the verdict.
2. **Add references**: F#666 (the violated guardrail), F#703 (1st routing-acc), F#710 (2nd routing-acc), F#147 (xxHash32 for SOLE routing), F#171 (routing mechanisms survey), arxiv:1611.01144 (Gumbel-softmax). Address guardrail 1002.
3. **Set `platform=local-apple`** (currently null; DB hygiene fix).
4. **Populate `success_criteria`** mirroring the new target-metric PASS condition.
5. **Tighten notes**: state the behavioral outcome being optimized, not just "winner takes the default" (which assumes the proxy IS the verdict — directly violates F#666).

Post-claim KC mutation is antipattern-u; edits must happen **before** re-claim. Recommendation: **close this pre-reg as structurally-malformed**; re-register `exp_routing_latency_benchmark_all_behavioral` with the target-pair KC structure.

### §5.1 Pre-existing partial coverage

Note that prior findings already partially answer the underlying research question:
- **F#108**: Hash ring routing latency negligible (sub-microsecond at all N).
- **F#147**: xxHash32 best hash for SOLE routing.
- **F#144** (killed): Inference routing strategies — all <5us latency, but quality capture bounded at 41.5%; pre-merge confirmed as sole default.
- **F#145** (killed): Routing latency at N=1000 — solved (6 strategies under 100ms).
- **F#171**: Routing mechanisms survey — 5 recommendations for N>25 scaling.
- **F#251 / F#257**: TF-IDF logistic routing on real NLP — 96.6% N=5, 86.1% N=25.
- **F#431**: TF-IDF routing scales to production NLP at N=25 weighted acc 86.1%.

The latency question (K1929) is **already answered** by F#108/F#144/F#145: hash-ring and TF-IDF are well under 10ms. The accuracy question (K1930) is **already answered** in pre-F#666 regime by F#251/F#431: TF-IDF achieves ~86% at N=25. So even setting aside F#666, the experiment is largely **redundant** — and a behavioral re-frame is needed to add new value.

## §6 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714 precedent + reviewer.md §5). Unblock is pre-registration-external (edit the DB entry to add target-pair KC + references + platform), not implementation-external.

`mem-antipattern-impl-follow-up-delegation` does not apply: that antipattern targets novel-mechanism PROVISIONAL, not preempt-structural KILL.
