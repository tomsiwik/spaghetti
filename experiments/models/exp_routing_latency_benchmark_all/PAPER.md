# PAPER — exp_routing_latency_benchmark_all (KILLED — preempt-structural, F#666-pure standalone, multi-bucket)

## Verdict: KILLED (KC-structural preempt, F#666-pure standalone, ≥23rd drain-window instance, multi-bucket)

This experiment is preempt-killed on structural grounds before any code executes. The verdict is deterministic from the KC-set shape, not from measurement.

## Summary

Pre-registered kill-criterion set K = {K1929, K1930} contains two proxy metrics with no paired target-metric KC:

- **K1929** ("Any routing method > 10ms per query") — pure latency threshold with no behavioral anchor. Per F#666, latency-only KCs are proxies (cf. F#702 which paired latency with *bitwise-exact token equivalence* and was therefore runnable).
- **K1930** ("Best routing method accuracy < 80% at N=25") — direct match for F#666 guardrail 1007 enumeration ("classification accuracy, routing match rate"). 3rd routing-accuracy sub-flavor instance (F#703 1st = TF-IDF, F#710 2nd = Gumbel-top2).

Under F#666, a proxy-only KC set has no valid verdict regardless of empirical outcome:
- Both PASS → tautological SUPPORT (F#666 canonical: 40.2% routing-acc + 0.0% target gap shows match-rate decoupled from utility via semantic-cluster routing).
- Any FAIL → "finding about the proxy, not a kill" per F#666 explicit rule.

This is a **multi-bucket** F#666-pure preempt: routing-accuracy sub-flavor (3rd instance) + infrastructure-benchmark-latency sub-flavor (2nd instance, after F#734 K-component). The pattern is well-established post the taxonomy refactor at F#714: ≥22 prior F#666-pure standalone instances exist (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, F#722, F#728, F#729, F#730, F#731, F#732, F#734, …).

## Prediction vs measurement

| KC    | Claim                                                        | Kind  | Sub-flavor                | Verdict                                  |
| ----- | ------------------------------------------------------------ | ----- | ------------------------- | ---------------------------------------- |
| K1929 | Any routing method > 10ms per query                          | proxy | infrastructure-benchmark  | UNTESTED (preempt-blocked, F#666-pure)   |
| K1930 | Best routing method accuracy < 80% at N=25                   | proxy | routing-match-rate (3rd)  | UNTESTED (preempt-blocked, F#666-pure)   |

No measurement was taken. No routing classifier (TF-IDF / xxHash / Gumbel-top2 / semantic / signature) was trained or benchmarked. No N=25 adapter set was loaded. The verdict derives from `F#666 proxy-only KC set` + `no target-metric pair` ⇒ tautological-for-all-outcomes (4-cell truth table in MATH.md §1).

## Why this is not runnable as-is

Even if the head-to-head benchmark were executed, every cell of the {K1929, K1930} × {PASS, FAIL} outcome space maps to an inadmissible verdict under F#666:

| K1929   | K1930   | Verdict                                                                                                  |
| ------- | ------- | -------------------------------------------------------------------------------------------------------- |
| PASS    | PASS    | Tautological SUPPORT (canonical decoupling: 40.2% acc + 0.0% target gap)                                 |
| PASS    | FAIL    | Finding about routing-accuracy threshold, not a behavioral kill                                          |
| FAIL    | PASS    | Finding about latency threshold, not a behavioral kill                                                   |
| FAIL    | FAIL    | Both proxies fail; still "finding about proxies, not kill" — no target was measured                      |

The F#666 rule operates on KC *kind*, not measurement value. Proxy-only structure is the disease; measurement outcome is the symptom.

## Hygiene defects

| Defect                | Status                                                                          |
| --------------------- | ------------------------------------------------------------------------------- |
| F#666 violation       | Present (K1929 + K1930 are both proxies; routing-match-rate forbidden by name)  |
| `success_criteria: []`| Present (DB explicitly flags `# ⚠ INCOMPLETE: missing success_criteria`)        |
| `platform: ~`         | Present (DB shows `platform: ~` null; `# ⚠ INCOMPLETE: missing ... platform`)   |
| `references: []`      | Present (guardrail 1002 violation; no arxiv or finding citation)                |
| Notes coherence       | "Winner takes the default" assumes the proxy IS the verdict — F#666 violation in framing |

Hygiene-defect count = 3 (F#666 + SC + ref + platform = 4 if F#666 counted, 3 hygiene proper). Crosses AP-prereg-hygiene-multi-defect (≥3) threshold. F#666-pure structural defect alone is sufficient for kill independent of hygiene count.

## Taxonomic comparison with drain-window precedents

| Dimension              | F#703 (1st routing-acc)             | F#710 (2nd routing-acc)             | This (3rd routing-acc)                | F#702 (runnable latency)                |
| ---------------------- | ----------------------------------- | ----------------------------------- | ------------------------------------- | --------------------------------------- |
| Parent dep             | none                                | none                                | none                                  | none                                    |
| KC count               | 1                                   | 1                                   | 2                                     | 2                                       |
| KC kinds               | proxy-only                          | proxy-only                          | **proxy-only (×2)**                   | **proxy + target** (latency + bitwise-eq) |
| F#666 violation        | yes                                 | yes                                 | yes (multi-bucket)                    | no (runnable)                           |
| Hygiene defects        | 2                                   | 2                                   | **3** (SC + platform + refs)          | 3                                       |
| Sub-flavor             | routing-acc 1st                     | routing-acc 2nd                     | routing-acc **3rd** + infra-bench 2nd | latency-WITH-pair                       |
| Verdict                | KILLED (preempt-structural)         | KILLED (preempt-structural)         | **KILLED (preempt-structural)**       | PROVISIONAL                             |
| `_impl` follow-up      | none                                | none                                | none                                  | yes                                     |

The invariant: `depends_on: []` + proxy-only KC set ⇒ preempt-KILL, independent of KC count, hygiene count, or sub-flavor.

## Caveats

- Both proxy KCs already have prior-art partial answers in pre-F#666 regime (F#108/F#144/F#145 for latency; F#251/F#431/F#171 for routing accuracy at N=25). Even setting aside F#666, the experiment is largely **redundant**.
- Base model `mlx-community/gemma-4-e4b-it-4bit` not loaded; no adapters injected; no MLX code executed; no routing classifier trained.
- "Winner takes the default" framing in pre-reg notes is itself an F#666 violation: it operationalizes the proxy as the verdict.

## Follow-up (recommended)

If a head-to-head routing-method benchmark on Gemma 4 E4B at N=25 is still a research question of interest, register `exp_routing_latency_benchmark_all_behavioral` with target-gated KCs:

```yaml
kill_criteria:
  - K_proxy_latency  : Any routing method > 10ms per query at N=25
  - K_proxy_accuracy : Best routing method routing-match-rate < 80% at N=25
  - K_target_quality : Best method's MMLU-Pro subject-domain task accuracy gap > 3pp vs oracle-adapter baseline at N=25
  - K_target_pareto  : No method achieves Pareto-frontier (both top-quartile latency AND top-quartile task-accuracy)
references:
  - F#666 (target-gated KILL discipline)
  - F#703 (routing-acc preempt 1st)
  - F#710 (routing-acc preempt 2nd)
  - F#147 (xxHash32 best for SOLE)
  - F#171 (routing mechanisms survey)
  - F#251, F#431 (TF-IDF prior art)
  - arxiv:1611.01144 (Gumbel-softmax)
  - arxiv:2310.14840 (arrow routing)
platform: local-apple
success_criteria: [K_target_quality PASS AND K_target_pareto PASS AND ≥1 method PASSES K_proxy_latency]
notes: "Pareto-quality routing benchmark on Gemma 4 E4B 4-bit at N=25. Latency is a constraint; behavioral utility is the verdict. Eliminates F#666 violation."
```

This closes the F#666 gap (target-pair present), addresses the redundancy concern (Pareto-quality is a novel framing), and preserves the engineering goal (still picks a winner among methods that both route well AND route fast).

## Unblock condition (no rerun of this pre-reg)

See MATH.md §5. Pre-reg must be edited before any re-claim to add a target-metric KC, references, populate success_criteria, set platform=local-apple, and remove the "winner takes the default" framing. Post-claim KC mutation is antipattern-u; recommendation is to **close this pre-reg as structurally-malformed** and use the follow-up template above instead.
