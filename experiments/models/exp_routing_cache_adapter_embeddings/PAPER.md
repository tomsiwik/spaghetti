# PAPER — exp_routing_cache_adapter_embeddings (KILLED — preempt-structural, F#666-pure standalone, multi-bucket)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, ≥24th drain-window instance, multi-bucket)

This experiment is preempt-killed on structural grounds before any code executes. The verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1931, K1932} contains two proxy metrics with no paired target-metric KC:

- **K1931** ("Cached routing accuracy < live routing accuracy by > 5pp") — routing-accuracy delta. Delta-of-proxies is still a proxy: F#666 guardrail 1007 enumerates classification accuracy / routing match rate by name as forbidden-solo; 4th routing-accuracy sub-flavor instance (F#703 1st = TF-IDF, F#710 2nd = Gumbel, F#753 K1930 3rd = best-at-N=25).
- **K1932** ("Cache invalidation frequency > 10% per session") — cache-staleness ops metric with no behavioral coupling. 3rd infrastructure-benchmark sub-flavor instance (F#734 K-component 1st, F#753 K1929 routing-method-latency 2nd).

Under F#666, a proxy-only KC set has no valid verdict regardless of empirical outcome:
- Both PASS → tautological SUPPORT (cache faithful to live, but live itself is a proxy; F#666 canonical: 40.2% routing-acc + 0.0% target gap demonstrates match-rate decoupled from utility).
- Any FAIL → "finding about the proxy, not a kill" per F#666 explicit rule.

This is a **multi-bucket** F#666-pure preempt: routing-accuracy-delta sub-flavor (4th instance) + infrastructure-benchmark sub-flavor (3rd instance). It is the **2nd cross-pollination** of routing-acc and infra-bench sub-flavors in a single pre-reg (F#753 was the 1st), promoting the multi-bucket pattern to confirmed-recurrent.

The pattern is well-established post the taxonomy refactor at F#714: ≥23 prior F#666-pure standalone instances exist (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#722, F#728, F#729, F#730, F#731, F#732, F#734, F#753, …).

## Prediction vs measurement

| KC    | Claim                                                                  | Kind  | Sub-flavor                       | Verdict                                 |
| ----- | ---------------------------------------------------------------------- | ----- | -------------------------------- | --------------------------------------- |
| K1931 | Cached routing accuracy < live routing accuracy by > 5pp               | proxy | routing-match-rate-delta (4th)   | UNTESTED (preempt-blocked, F#666-pure)  |
| K1932 | Cache invalidation frequency > 10% per session                         | proxy | infrastructure-benchmark (3rd)   | UNTESTED (preempt-blocked, F#666-pure)  |

No measurement was taken. No ANN index was built; no adapter embeddings were precomputed; no cache invalidation simulation was run; no live-vs-cache routing accuracy was evaluated. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ tautological-for-all-outcomes (4-cell truth table in MATH.md §1).

## Why this is not runnable as-is

Even if the cache benchmark were executed, every cell of the {K1931, K1932} × {PASS, FAIL} outcome space maps to an inadmissible verdict under F#666:

| K1931  | K1932  | Verdict                                                                                                                          |
| ------ | ------ | -------------------------------------------------------------------------------------------------------------------------------- |
| PASS   | PASS   | Tautological SUPPORT — cache faithful to live, but live itself is a proxy (F#666 canonical: routing-acc decoupled from utility) |
| PASS   | FAIL   | Finding about cache-staleness threshold, not a behavioral kill                                                                   |
| FAIL   | PASS   | Finding about routing-acc-delta threshold, not a behavioral kill                                                                 |
| FAIL   | FAIL   | Both proxies fail; still "finding about proxies, not kill" — no target was measured                                              |

The F#666 rule operates on KC *kind*, not measurement value. Proxy-only structure is the disease; measurement outcome is the symptom.

### Pathological-case illustrations of decoupling

1. **K1931 PASS + behavior degraded**: cache faithfully reproduces live's mistakes (e.g., both route medical→legal at the same wrong rate). Δ-routing-acc is small but every selection is wrong.
2. **K1931 FAIL + behavior preserved**: cache disagrees per-sample for ambiguous prompts but lands in the same semantic cluster. F#666 canonical (cluster routing achieves oracle behavior at 40.2% per-sample acc) shows this is benign.
3. **K1932 PASS via never-invalidating**: cache is always-stale, fast cache hit but adapter weights drift; target metric (downstream task accuracy) would catch the silent staleness, but K1932 alone misses it.
4. **K1932 PASS via always-invalidating-on-trigger**: technically zero "stale" hits but the cache degenerates to live routing — engineering goal defeated.

Each pathology requires a behavioral target KC to disambiguate.

## Hygiene defects

| Defect                  | Status                                                                          |
| ----------------------- | ------------------------------------------------------------------------------- |
| F#666 violation         | Present (K1931 + K1932 are both proxies; routing-match-rate forbidden by name)  |
| `success_criteria: []`  | Present (DB explicitly flags `# ⚠ INCOMPLETE: missing success_criteria`)        |
| `platform: ~`           | Present (DB shows `platform: ~` null; `# ⚠ INCOMPLETE: missing ... platform`)   |
| `references: []`        | Present (guardrail 1002 violation; no arxiv or finding citation)                |
| Notes coherence         | "Tests whether offline routing matches online" treats live router as ground truth — F#666 violation in framing (live router is itself a proxy per F#703/F#710/F#753) |

Hygiene-defect count = 3 (SC + platform + refs). Crosses AP-prereg-hygiene-multi-defect (≥3) threshold. F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#703 (1st routing-acc)             | F#710 (2nd routing-acc)             | F#753 (3rd routing-acc + 2nd infra-bench) | This (4th routing-acc + 3rd infra-bench) | F#702 (runnable infra)                  |
| ---------------------- | ----------------------------------- | ----------------------------------- | ----------------------------------------- | ---------------------------------------- | --------------------------------------- |
| Parent dep             | none                                | none                                | none                                      | none                                     | none                                    |
| KC count               | 1                                   | 1                                   | 2                                         | 2                                        | 2                                       |
| KC kinds               | proxy-only                          | proxy-only                          | **proxy-only (×2)**                       | **proxy-only (×2)**                      | **proxy + target** (latency + bitwise-eq) |
| F#666 violation        | yes                                 | yes                                 | yes (multi-bucket)                        | yes (multi-bucket)                       | no (runnable)                           |
| Hygiene defects        | 2                                   | 2                                   | 3 (SC + platform + refs)                  | **3** (SC + platform + refs)             | 3                                       |
| Routing-acc form       | solo                                | solo                                | solo (best-at-N=25)                       | **delta** (cache vs live)                | n/a                                     |
| Sub-flavor             | routing-acc 1st                     | routing-acc 2nd                     | routing-acc 3rd + infra-bench 2nd         | routing-acc **4th** + infra-bench **3rd**| latency-WITH-pair                       |
| Multi-bucket pair      | no                                  | no                                  | 1st routing-acc+infra-bench               | **2nd** routing-acc+infra-bench (confirmed-recurrent) | n/a                          |
| Verdict                | KILLED (preempt-structural)         | KILLED (preempt-structural)         | KILLED (preempt-structural)               | **KILLED (preempt-structural)**          | PROVISIONAL                             |
| `_impl` follow-up      | none                                | none                                | none                                      | none                                     | yes                                     |

The invariant: `depends_on: []` + proxy-only KC set ⇒ preempt-KILL, independent of KC count, hygiene count, sub-flavor, or proxy form (solo vs delta).

## Caveats

- All four proxy-form variants of routing-accuracy KC have now been observed (TF-IDF, Gumbel, best-at-N, delta-vs-live). The space of "obvious routing-acc KC formulations" is closing; researchers are likely to next attempt second-derivative variants (cache-vs-cache delta, multi-classifier ensemble agreement) — these too will be F#666-pure unless paired with a behavioral target.
- Both proxy KCs already have prior-art partial answers in pre-F#666 regime (F#108/F#147 hash routing already cache-friendly; F#251/F#431 TF-IDF is itself a precomputed-IDF cache; F#394 routing latency is non-bottleneck since TTFT dominates). Even setting aside F#666, the experiment is largely **redundant**.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters injected; no MLX code executed; no ANN index built.
- "Tests whether offline routing matches online" framing in pre-reg notes is itself an F#666 violation: it operationalizes the live proxy as the verdict (matching to a proxy is a Δ-proxy).

## Follow-up (recommended)

If a cache-fidelity experiment on Gemma 4 E4B at N=25 is still a research question of interest, register `exp_routing_cache_adapter_embeddings_behavioral` with target-gated KCs:

```yaml
kill_criteria:
  - K_proxy_acc_delta  : Cached routing accuracy < live routing accuracy by > 5pp
  - K_proxy_staleness  : Cache invalidation frequency > 10% per session
  - K_target_quality   : Per-prompt MMLU-Pro task-accuracy gap (cache-routed vs live-routed) > 2pp at N=25
  - K_target_pareto    : No cache method achieves Pareto-frontier (top-quartile staleness AND top-quartile task-accuracy preservation)
references:
  - F#666 (target-gated KILL discipline)
  - F#703, F#710, F#753 (routing-acc preempt precedents)
  - F#147, F#108 (hash routing prior art)
  - F#251, F#431 (TF-IDF cache-like routing baseline)
  - F#394 (routing-latency non-bottleneck)
  - arxiv:2401.04658 (LoRA Hub adapter retrieval)
  - arxiv:2310.18362 (RAG-style adapter routing)
  - arxiv:2305.14314 (ANN index + LoRA, e.g. Faiss/HNSW)
platform: local-apple
success_criteria: [K_target_quality PASS AND K_target_pareto PASS AND >=1 method PASSES K_proxy_staleness]
notes: "Pareto-quality cache-vs-live routing benchmark on Gemma 4 E4B 4-bit at N=25. Cache fidelity to live is the proxy; downstream task-accuracy preservation is the verdict. Eliminates F#666 violation."
```

This closes the F#666 gap (target-pair present), addresses the redundancy concern (Pareto-quality framing is novel), and preserves the engineering goal (still picks a winner among caches that both faithfully approximate live AND don't degrade behavior).

## Unblock condition (no rerun of this pre-reg)

See MATH.md §5. Pre-reg must be edited before any re-claim to add a target-metric KC, references, populate success_criteria, set platform=local-apple, and remove the "matches online" framing (which treats live as ground truth). Post-claim KC mutation is antipattern-u; recommendation is to **close this pre-reg as structurally-malformed** and use the follow-up template above instead.
