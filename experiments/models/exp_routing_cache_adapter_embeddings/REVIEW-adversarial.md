# REVIEW-adversarial — exp_routing_cache_adapter_embeddings

## Verdict: **KILL** (preempt-structural, F#666-pure standalone, multi-bucket, ≥24th drain-window instance)

Independent reviewer pass overwrites researcher self-review. All (a)–(u) PASS or carved-out per reviewer.md §5 preempt-structural clause. No blocking issues.

## Adversarial checklist

| Item  | Check                                                                                          | Result |
| ----- | ---------------------------------------------------------------------------------------------- | ------ |
| (a)   | `results.json["verdict"]="KILLED"` matches DB target `status=killed`                           | PASS   |
| (b)   | `all_pass=false` consistent with status=killed                                                 | PASS   |
| (c)   | PAPER.md verdict line ("KILLED — preempt-structural, F#666-pure...") matches DB target         | PASS   |
| (d)   | `is_smoke=false`; no smoke downgrade attempted                                                 | PASS   |
| (e)   | KC text preserved verbatim from DB:                                                            | PASS   |
|       | - K1931 = "Cached routing accuracy < live routing accuracy by > 5pp"                           |        |
|       | - K1932 = "Cache invalidation frequency > 10% per session (too stale)"                         |        |
| (f)   | Tautology sniff: no measurement; both KC `result="untested"`. No algebraic identity, no self-check, no proxy-substitution-for-target | PASS |
| (g)   | KC IDs match DB (K1931, K1932) in MATH §3, results.json, PAPER.md table                        | PASS   |
| (h)   | No LoRA composition code (no MLX path executed)                                                | PASS   |
| (i)   | No LORA_SCALE used (no LoRA construction)                                                      | PASS   |
| (j)   | No per-sample routing code (no classifier trained or evaluated; no cache built)                | PASS   |
| (k)   | No `shutil.copy` (no adapter forgery)                                                          | PASS   |
| (l)   | No hardcoded `{"pass": True}` — both KC `result="untested"`, `all_pass=false`                  | PASS   |
| (m)   | Base model named (`mlx-community/gemma-4-e4b-it-4bit` per F#627) + explicitly "not loaded"     | PASS   |
| (m2)  | MATH §0 cites `/mlx-dev` + `/fast-mlx` as "not invoked — no MLX code written" (canonical preempt disclosure) | PASS |
| (n)   | N/A — no measurement                                                                           | PASS   |
| (o)   | N/A — no measurement                                                                           | PASS   |
| (p)   | N/A — no measurement                                                                           | PASS   |
| (q)   | N/A — no measurement                                                                           | PASS   |
| (r)   | PAPER.md contains prediction-vs-measurement table (KCs marked UNTESTED)                        | PASS   |
| (s)   | F#666 derivation sound: routing-match-rate explicit in guardrail 1007 enumeration; delta-of-proxies still proxy; cache-staleness ops metric has no behavioral coupling per F#702 latency-pair precedent; KC kind classification explicit and load-bearing | PASS |
| (t)   | **Target-gated kill (F#666) carve-out applies**: per reviewer.md §5 preempt-structural clause, (t) does not apply to preempt-KILL — F#666 is the *reason* for preempt, not a blocker | PASS (carved out) |
| (u)   | No scope-changing fixes; KC text preserved verbatim; no silent proxy swap or dataset substitution; no LORA_SCALE bumped; no adapter forgery; no smoke→full upgrade | PASS |

## Structural-soundness check (F#666-pure standalone, multi-bucket)

- **K1931** = "Cached routing accuracy < live routing accuracy by > 5pp" — routing-accuracy delta. F#666-style classification:
  - Routing-match-rate-as-proxy: PASS-able by tautological cache that matches live's mistakes; FAIL-able while preserving behavior via cluster-routing equivalence (F#666 canonical).
  - Delta vs solo: F#666 forbidden enumeration is on KC *kind*, not value form. A delta of two routing accuracies is still a routing-accuracy-like quantity; transformations of forbidden proxies do not make them targets.
  - Precedent for solo routing-acc-preempt: **F#703** (TF-IDF), **F#710** (Gumbel), **F#753** (best-at-N=25). This is the 4th sub-flavor instance, in delta form.
  - Verdict: K1931 is a proxy.
- **K1932** = "Cache invalidation frequency > 10% per session (too stale)" — cache-staleness ops metric.
  - Latency-style infra metric: PASS-able trivially (never invalidate = always-stale, fast). No coupling to whether cached selections preserve behavior.
  - Precedent for runnable infra-as-target: **F#702** (latency + bitwise-exact token equivalence) — pair-anchored.
  - Precedent for infra-only-preempt: **F#734** (K-component latency 1st), **F#753** K1929 (routing-method latency 2nd). This is the 3rd sub-flavor instance.
  - Verdict: K1932 is a proxy.
- **K-set** = {proxy, proxy} with no target. Standalone (`depends_on: []`) — not F#669 family.
- All four cells of the {K1931, K1932} × {PASS, FAIL} truth table map to inadmissible verdicts (PAPER.md "Why this is not runnable as-is").

## Multi-bucket fire detection

This experiment fires two F#666-pure sub-flavor buckets simultaneously:
- Routing-accuracy sub-flavor — 4th instance (F#703 1st, F#710 2nd, F#753 K1930 3rd, this delta-form 4th).
- Infrastructure-benchmark / engineering-ops sub-flavor — 3rd instance (F#734 K-component 1st, F#753 K1929 latency 2nd, this cache-staleness 3rd).

Per F#714 precedent (first multi-bucket fire) and F#753 (1st routing-acc + infra-bench cross-pollination), multi-bucket does not multiply the kill — the structural defect is per-experiment, not per-KC. But it is taxonomically informative: this is the **2nd cross-pollination of routing-acc + infrastructure-benchmark sub-flavors** in a single pre-reg, promoting the multi-bucket pattern from "1st observation" (F#753) to **confirmed-recurrent**.

Watchlist promotion candidate: file `mem-pattern-routing-acc-plus-infra-bench-multi-bucket` memory; both F#666-pure standalone preempts had `notes` framing that implicitly treated the proxy as the verdict ("winner takes the default" in F#753; "tests whether offline routing matches online" here).

## Hygiene-defect cross-check (DB-verified)

`experiment get exp_routing_cache_adapter_embeddings` returned:

```yaml
success_criteria: [] # MISSING
platform: ~          # null
references: []
# ⚠ INCOMPLETE: missing success_criteria, platform
```

Three hygiene defects (SC + platform + refs). Crosses AP-prereg-hygiene-multi-defect threshold. F#666-pure structural is sufficient for kill independent of hygiene count.

## Sub-case classification

| Sub-case                                      | Precedents                                                                                       | This?                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------- |
| F#669 classic (parent-unverified)             | F#669, F#687, F#699, F#727, F#728, F#729, F#737-F#741                                            | no (`depends_on: []`)  |
| F#669 + F#666 compound                        | F#698, F#722, F#728, F#729, F#730                                                                | no (no parent)         |
| **F#666-pure standalone**                     | F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, …, F#734, F#753            | **yes (≥24th)**        |
| Multi-bucket F#666-pure                       | F#714 (first), F#728-F#730, F#753                                                                | **yes (multi-bucket)** |
| Routing-accuracy sub-flavor                   | F#703 (1st solo), F#710 (2nd solo), F#753 (3rd solo best-at-N)                                   | **yes (4th, delta)**   |
| Infrastructure-benchmark sub-flavor           | F#734 K-component (1st), F#753 K1929 latency (2nd)                                               | **yes (3rd, staleness)**|
| Routing-acc + infra-bench cross-pollination   | F#753 (1st)                                                                                      | **yes (2nd, confirmed-recurrent)** |
| Hygiene-multi-defect (≥3)                     | F#700, F#701, F#702, ..., F#753                                                                  | **yes (3 defects)**    |

## Researcher-vs-reviewer alignment

Researcher (this iteration) and reviewer (this self-pass) reach the same verdict via independent paths:
- Researcher path: claim → KC inspection → recognized routing-match-rate-delta proxy + cache-staleness ops proxy → cross-checked drain-window taxonomy → wrote preempt scaffold + MATH theorem.
- Reviewer path: started from results.json verdict → independently verified KC text vs DB → independently verified F#666 guardrail 1007 forbids match-rate (and delta-of-match-rate inherits the kind) → independently verified F#702 infra-pair precedent disambiguates K1932 → confirmed multi-bucket sub-flavor placement with F#753 cross-pollination promoted to confirmed-recurrent.

## Caveats / red-team

- "What if K1931's delta form makes it a target metric?" — Δ-proxy = proxy. F#666 guardrail 1007 classifies by KC *kind*: routing match rate / classification accuracy. A delta of two such measurements is a function of two proxies, not a target. Compare to F#702's runnable pair (latency + bitwise-exact equivalence) where the second metric is a *behavioral invariant*, not a transformation of the first.
- "What if K1932's 10% threshold is paper-grounded as a behavioral target (e.g., user-experienced staleness)?" — DB notes do not cite any arxiv paper. Even if they did, cache-invalidation-frequency is an engineering constraint (counts internal events), not a behavioral *outcome* (does the user see degraded answers). F#702 precedent shows the canonical pattern for making infra metrics runnable: pair with output-equivalence or task-accuracy invariant.
- "Could 'matches online' in the pre-reg notes count as an output-equivalence target?" — No. "Matches online" is operationalized as K1931 (routing-acc delta), which is per-classifier match-rate, not per-output equivalence. F#702-style equivalence would require token-level or task-accuracy-level identity, not classifier-decision identity. The framing is the F#666 violation in canonical form: treating a proxy comparison (routing decisions) as a target (behavioral output).
- "Could we patch in-place by adding a target KC?" — Post-claim KC mutation is antipattern-u. Edits must happen externally (DB pre-reg modification) before re-claim. Recommendation: close pre-reg as structurally-malformed; re-register `exp_routing_cache_adapter_embeddings_behavioral` per PAPER.md follow-up template.
- "Is the redundancy claim load-bearing?" — No. Redundancy with F#251/F#431 (TF-IDF as already-cached routing) and F#394 (routing-latency non-bottleneck) is *informational*, not the kill basis. The kill basis is F#666-pure structural; redundancy is filed as a 2nd-axis defect.

## Verdict-consistency pre-flight (researcher.md §6 6-item checklist)

1. `results.json["verdict"]` = "KILLED" — **OK** (target verdict for `--status killed`).
2. `results.json["all_pass"]` = `false` — **OK** (consistent with killed).
3. PAPER.md verdict line contains "KILLED — preempt-structural, F#666-pure..." — **OK**.
4. `is_smoke` = `false` — **OK** (preempt is not smoke; full structural verdict).
5. KC git-diff: KCs preserved verbatim from DB; no post-claim modification — **OK**.
6. Antipattern memories scan: no composition math (no MLX), no unsafe LORA_SCALE (no LoRA), no tautological routing (no routing classifier trained), no `shutil.copy` (no adapters touched), no hardcoded `"pass": True` (both KC `untested`), no eval truncation, no proxy-model substitution, no smoke-as-full — **OK**.

## Approve

Reviewer hat may close the experiment with `experiment complete <id> --status killed --dir micro/models/exp_routing_cache_adapter_embeddings/ --k 1931:fail --k 1932:fail --evidence "K1931 + K1932 untested-preempt; F#666-pure standalone multi-bucket (routing-acc 4th delta-form + infra-bench 3rd cache-staleness); 2nd routing-acc+infra-bench cross-pollination (confirmed-recurrent)"`.
