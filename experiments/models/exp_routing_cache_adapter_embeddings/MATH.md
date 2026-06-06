# MATH.md — exp_routing_cache_adapter_embeddings (PREEMPT-KILL, F#666-pure standalone, ≥24th drain-window instance)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure standalone — multi-bucket: routing-accuracy 4th sub-flavor + infrastructure-benchmark 3rd sub-flavor)

This experiment is preempt-killed before any code runs. The kill is **structural**: the pre-registered kill-criterion set K = {K1931, K1932} consists of two proxy metrics (routing-accuracy delta vs live + cache-staleness frequency) with no paired target-metric KC. Under F#666 (guardrail 1007 — target-gated KILL discipline) neither KILL nor SUPPORTED is derivable regardless of empirical outcome.

This is a continuation of the F#666-pure standalone canonical pattern (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#753 — at least 23 prior). Specifically:
- **4th routing-accuracy sub-flavor**: K1931 is a delta on "routing accuracy" — match-rate proxy explicitly forbidden by guardrail 1007 (1st: F#703 K1569 TF-IDF acc; 2nd: F#710 K1591 Gumbel acc; 3rd: F#753 K1930 best-method-at-N=25).
- **3rd infrastructure-benchmark sub-flavor**: K1932 is "cache invalidation frequency > 10% per session" — pure ops/serving metric with no behavioral coupling (1st: F#734 K-component latency; 2nd: F#753 K1929 routing-method latency).

## §0 Platform / skills / model pins

Included for reviewer checklist (m2) completeness. No platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — caching benchmark over a fixed adapter set; ANN-index scaffolding, no LoRA injection or training in this run.
- Parent dependency: **none** (`depends_on: []`). NOT an F#669 preempt.

## §1 Preempt-KILL theorem (F#666-pure, multi-bucket)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_routing_cache_adapter_embeddings` with kill-criterion set K = {K1931, K1932}:
- K1931 := "Cached routing accuracy < live routing accuracy by > 5pp"
- K1932 := "Cache invalidation frequency > 10% per session (too stale)"

**Classification of K.**
- K1931 is a **proxy metric** — routing-accuracy delta. Delta form does not change kind: F#666 guardrail 1007 enumerates *classification accuracy* and *routing match rate* by name; a delta of two routing accuracies is still a routing-match-rate measurement (Δ-of-proxies = proxy). No coupling to whether the chosen adapter produces behaviorally correct output. F#666 canonical (40.2% routing-acc + 0.0% target gap) shows match-rate decoupled from utility — therefore Δ-match-rate inherits decoupling.
- K1932 is a **proxy metric** — cache-invalidation frequency is a serving/ops engineering metric with no behavioral anchor. PASS-condition (low invalidation) is achievable trivially by never invalidating (always-stale cache, fast but useless), or by always-invalidating (always-fresh cache, slow but degenerate to live routing). Neither condition tests whether cache outputs match live outputs in *behavior*.

Neither KC measures task accuracy, behavioral quality, oracle-gap, or any downstream-behavioral outcome. K is a 2-proxy, 0-target set.

**F#666 gating (guardrail 1007).** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. A verdict derived from a proxy-only KC set is tautological. Per F#714 / F#753 multi-proxy precedents, the analysis is per-KC then composed:

| K1931 | K1932 | V(K) under F#666                                                                              |
| ----- | ----- | --------------------------------------------------------------------------------------------- |
| PASS  | PASS  | Tautological SUPPORT — both proxies, no target. Cache could match live's routing-acc and rarely invalidate yet still cache *the wrong adapter selection*: cache fidelity to live ≠ live fidelity to oracle. |
| PASS  | FAIL  | Mixed proxy outcome; F#666 rule "Proxy-FAIL + target-absent = finding about the proxy, not a kill" — produces a finding about cache-staleness threshold, not a behavioral kill. |
| FAIL  | PASS  | Mixed proxy outcome; same finding-not-kill rule applies on the routing-acc-delta proxy. |
| FAIL  | FAIL  | Both-fail proxy-only — under F#666 still "finding about proxies, not kill" because no target was measured. |

**No cell yields a valid F#666-compliant verdict.** K is unidentifiable at the F#666 layer. **QED.**

### §1.1 Routing-accuracy-delta is still a routing-accuracy proxy (4th sub-flavor instance)

K1931 measures *Δ(cached_acc, live_acc)*. Even if `live_acc` were a behavioral target, K1931 is calibrated against the *classifier's match-rate*, not the downstream output. Two pathological cases illustrate decoupling:

1. **Δ ≤ 5pp PASS, behavior degraded**: cache matches live's mistakes (e.g., both route medical→legal at the same wrong rate). Cache accurately reproduces a tautologically-routing live system; routing-acc-delta is small but every selection is wrong.
2. **Δ > 5pp FAIL, behavior preserved**: cache routes differently for ambiguous samples (intra-cluster dispersion) but lands in the same semantic cluster — F#666 canonical shows cluster-routing achieves oracle behavior at 40.2% per-sample acc; cache and live could disagree per-sample yet produce identical generations.

Both cases require a target-metric KC (downstream task accuracy or oracle-gap) to disambiguate.

### §1.2 Cache-invalidation-frequency is serving-engineering, not behavior (3rd infra-bench sub-flavor instance)

K1932 measures cache hit/miss rate. Why is this not a target?
- A cache that never invalidates passes K1932 trivially but defeats the purpose (always-stale).
- A cache that invalidates on every request passes K1932 only if "invalidation" is defined exclusively (e.g., adapter weight change), but then the metric measures *adapter version churn*, not cache utility.
- Behavioral question: does cached routing produce identical adapter selections to live routing for prompts that should route the same? Or, more strictly: does cache-routed generation match live-routed generation under task-accuracy?

Neither variant of K1932 answers either behavioral question. F#702 precedent (latency + bitwise-equivalence) shows the canonical infra-metric+target pair; K1932 has no such pair. F#734 K-component (1st infra-bench sub-flavor) and F#753 K1929 routing-method-latency (2nd) established this as a recurrent F#666-pure subclass.

## §2 Prior art (preempt-KILL precedents)

- **F#666** (2026-04-19, conclusive): target-gated KILL discipline; guardrail 1007 enumerates classification accuracy & routing match rate as forbidden-solo proxies; canonical 40.2% proxy + 0.0% target gap shows decoupling.
- **F#700** (2026-04-24): 1st F#666-pure standalone preempt-KILL.
- **F#703** (2026-04-24): **1st routing-accuracy sub-flavor** — `exp_followup_tfidf_medical_unaliased`, K1569 TF-IDF routing weighted accuracy.
- **F#710** (2026-04-24): **2nd routing-accuracy sub-flavor** (confirmed-recurrent) — `exp_g4_gumbel_top2_n50`, K1591 routing acc ≥85%.
- **F#714** (2026-04-24): 10th F#666-pure, **first multi-bucket fire** (derived-geometric + detection); first triple-fire precedent.
- **F#734** (2026-04-25): 22nd F#666-pure, **infrastructure-benchmark sub-flavor 1st identified** (K-component latency).
- **F#753** (2026-04-25): **3rd routing-accuracy + 2nd infrastructure-benchmark sub-flavor** — `exp_routing_latency_benchmark_all`, K1929 latency + K1930 best-at-N=25 acc; first cross-pollination of routing-acc and latency-only buckets.
- **`mem-antipattern-f666-pure-standalone-preempt-kill`** (filed 2026-04-24, escalated multiple times): claim-time detection rule; preempt-scaffold response.
- **Guardrail 1007** (PLAN.md): every proxy KC must be paired with a target-metric KC.
- **F#702**: latency-with-bitwise-equivalence-pair example of a runnable infra-metric KC (illustrates the missing pair here).

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                       | Kind  | Sub-flavor                        | Measurement status                |
| ----- | --------------------------------------------------------------------------- | ----- | --------------------------------- | --------------------------------- |
| K1931 | Cached routing accuracy < live routing accuracy by > 5pp                    | proxy | routing-match-rate-delta (4th)    | untested (preempt-blocked)        |
| K1932 | Cache invalidation frequency > 10% per session                              | proxy | infrastructure-benchmark (3rd)    | untested (preempt-blocked)        |

No target-metric KC exists. K is structurally malformed per F#666.

KC text preserved verbatim from `experiment get exp_routing_cache_adapter_embeddings` output. No post-claim KC mutation (antipattern-u check: PASS).

## §4 Hygiene defects (noted, not load-bearing for kill)

Per `experiment get exp_routing_cache_adapter_embeddings`:

1. **`success_criteria: []`** (MISSING per DB output) — empty; no SUPPORTED-condition declared.
2. **`platform: ~`** (null) — guardrail/hygiene defect.
3. **`references: []`** — guardrail 1002 violation (every new experiment MUST cite an arxiv paper or prior finding).

Three hygiene defects total. Crosses the AP-prereg-hygiene-multi-defect threshold (≥3 defects). However, F#666-pure structural defect alone is sufficient for kill independent of hygiene count (per F#703 invariant; same shape as F#753).

Notes field reads: "Precompute adapter embeddings, index with ANN. Tests whether offline routing matches online." This describes a benchmark methodology against the *live router as ground truth* — confirming the F#666 violation: live routing is a proxy (already in F#703/F#710/F#753 forbidden set), and matching to a proxy is a Δ-proxy.

## §5 Unblock condition (re-claim requires KC-augmentation pre-registration)

Re-registration as a new experiment id (`exp_routing_cache_adapter_embeddings_behavioral` recommended) with the following fixes:

1. **Add a target-metric KC** pairing cache-vs-live routing to a behavioral outcome. Candidate formulations:
   - **End-to-end task-accuracy preservation**: For prompts where cache and live select different adapters, measure MMLU-Pro subject-domain accuracy under each selection. Target KC: per-prompt task-accuracy gap (cache vs live) ≤ 2pp at N=25. Couples cache fidelity to downstream utility.
   - **Bitwise-output equivalence on cache-hits** (F#702 pattern): for cache-hit prompts (cache and live agree), generated tokens must match bit-for-bit at temperature=0; cache-miss prompts must close gap to oracle within 3pp. Two-tier target.
   - **Pareto-staleness**: among caches that PASS K1932 (≤10% invalidation), best cache's downstream task-accuracy ≥ live − 2pp. Engineering metric becomes a *constraint*, behavioral metric is the *verdict*.
2. **Add references**: F#666 (guardrail), F#703/F#710/F#753 (routing-acc preempt precedents), arxiv:2401.04658 (RAG-style adapter retrieval), arxiv:2310.18362 (LoRA Hub), F#171 (routing mechanisms survey), F#108/F#147 (hash routing prior art). Address guardrail 1002.
3. **Set `platform=local-apple`** (currently null; DB hygiene fix).
4. **Populate `success_criteria`** mirroring the new target-metric PASS condition (e.g., "behavioral target KC PASS ∧ ≥1 cache method PASSES K1932").
5. **Tighten notes**: state the *behavioral* outcome the cache must preserve, not just "whether offline routing matches online" (which assumes live routing IS the verdict — directly violates F#666 in the same way as F#753's "winner takes the default").

Post-claim KC mutation is antipattern-u; edits must happen **before** re-claim. Recommendation: **close this pre-reg as structurally-malformed**; re-register `exp_routing_cache_adapter_embeddings_behavioral` with the target-pair KC structure.

### §5.1 Pre-existing partial coverage

Note that prior findings already partially answer the underlying research question:
- **F#108**: Hash ring routing already sub-microsecond — a "cache" of precomputed hash buckets is *already* the live router for hash-routing. Caching ANN-embedding routing layers on top of existing fast routers has marginal latency gain.
- **F#171**: Routing mechanisms survey — 5 recommendations for N>25 scaling.
- **F#251 / F#431**: TF-IDF logistic routing on real NLP — 96.6% N=5, 86.1% N=25. TF-IDF is a precomputed-IDF index; it is *already* a cached routing scheme.
- **F#394**: Adapter hot-swap overhead 0.26ms inject (free), TTFT dominates total latency. Implies routing-classifier latency is *not the bottleneck*; cache-vs-no-cache should produce sub-millisecond differences swamped by TTFT.

The cache-fidelity question (K1931) and staleness question (K1932) are *engineering questions about a non-bottleneck* per F#394, and the underlying routing question is already answered by F#251/F#431. So even setting aside F#666, the experiment is largely **redundant** — and a behavioral re-frame is needed to add new value.

## §6 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699/F#700/F#701/F#703/F#705/F#710/F#714/F#753 precedent + reviewer.md §5). Unblock is pre-registration-external (edit the DB entry to add target-pair KC + references + platform), not implementation-external.

`mem-antipattern-impl-follow-up-delegation` does not apply: that antipattern targets novel-mechanism PROVISIONAL, not preempt-structural KILL.
