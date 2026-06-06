# REVIEW-adversarial — exp_routing_latency_benchmark_all

## Verdict: **KILL** (preempt-structural, F#666-pure standalone, multi-bucket, ≥23rd drain-window instance)

Independent reviewer pass overwrites researcher self-review. All (a)–(u) PASS or carved-out per reviewer.md §5 preempt-structural clause. No blocking issues.

## Adversarial checklist

| Item  | Check                                                                                          | Result |
| ----- | ---------------------------------------------------------------------------------------------- | ------ |
| (a)   | `results.json["verdict"]="KILLED"` matches DB target `status=killed`                           | PASS   |
| (b)   | `all_pass=false` consistent with status=killed                                                 | PASS   |
| (c)   | PAPER.md verdict line ("KILLED — preempt-structural, F#666-pure...") matches DB target         | PASS   |
| (d)   | `is_smoke=false`; no smoke downgrade attempted                                                 | PASS   |
| (e)   | KC text preserved verbatim from DB:                                                            | PASS   |
|       | - K1929 = "Any routing method > 10ms per query (too slow for real-time)"                       |        |
|       | - K1930 = "Best routing method accuracy < 80% at N=25"                                         |        |
| (f)   | Tautology sniff: no measurement; both KC `result="untested"`. No algebraic identity, no self-check, no proxy-substitution-for-target | PASS |
| (g)   | KC IDs match DB (K1929, K1930) in MATH §3, results.json, PAPER.md table                        | PASS   |
| (h)   | No LoRA composition code (no MLX path executed)                                                | PASS   |
| (i)   | No LORA_SCALE used (no LoRA construction)                                                      | PASS   |
| (j)   | No per-sample routing code (no classifier trained or evaluated)                                | PASS   |
| (k)   | No `shutil.copy` (no adapter forgery)                                                          | PASS   |
| (l)   | No hardcoded `{"pass": True}` — both KC `result="untested"`, `all_pass=false`                  | PASS   |
| (m)   | Base model named (`mlx-community/gemma-4-e4b-it-4bit` per F#627) + explicitly "not loaded"     | PASS   |
| (m2)  | MATH §0 cites `/mlx-dev` + `/fast-mlx` as "not invoked — no MLX code written" (canonical preempt disclosure) | PASS |
| (n)   | N/A — no measurement                                                                           | PASS   |
| (o)   | N/A — no measurement                                                                           | PASS   |
| (p)   | N/A — no measurement                                                                           | PASS   |
| (q)   | N/A — no measurement                                                                           | PASS   |
| (r)   | PAPER.md contains prediction-vs-measurement table (KCs marked UNTESTED)                        | PASS   |
| (s)   | F#666 derivation sound: routing-match-rate explicit in guardrail 1007 enumeration; latency-only-as-proxy precedent F#702 (runnable WITH bitwise-pair) vs F#734/F#740 (preempt WITHOUT pair); KC kind classification explicit and load-bearing | PASS |
| (t)   | **Target-gated kill (F#666) carve-out applies**: per reviewer.md §5 preempt-structural clause, (t) does not apply to preempt-KILL — F#666 is the *reason* for preempt, not a blocker. Alternative reading: K1930 IS the proxy half of the missing pair; preempt blocks until target half is added | PASS (carved out) |
| (u)   | No scope-changing fixes; KC text preserved verbatim; no silent proxy swap or dataset substitution; no LORA_SCALE bumped; no adapter forgery; no smoke→full upgrade | PASS |

## Structural-soundness check (F#666-pure standalone, multi-bucket)

- **K1929** = "any routing method > 10ms per query" — pure latency threshold. F#666-style classification:
  - Latency-only-as-proxy: PASS-able by random/constant routing that ignores input. No coupling to whether the routing produces correct adapter selection.
  - Precedent for runnable latency-as-target: **F#702** paired latency with *bitwise-exact token equivalence* — that pair was target-anchored. K1929 has no such pair.
  - Precedent for latency-only-preempt: **F#734** K-component (infrastructure-benchmark sub-flavor 1st instance), **F#740** Pierre-serving children.
  - Verdict: K1929 is a proxy.
- **K1930** = "best routing method accuracy < 80% at N=25" — direct routing match rate.
  - F#666 guardrail 1007 explicit enumeration: "classification accuracy, routing match rate".
  - F#666 canonical: 40.2% acc + 0.0% target gap demonstrates decoupling via semantic-cluster routing.
  - 3rd sub-flavor instance (F#703 1st, F#710 2nd, this 3rd).
  - Verdict: K1930 is a proxy.
- **K-set** = {proxy, proxy} with no target. Standalone (`depends_on: []`) — not F#669 family.
- All four cells of the {K1929, K1930} × {PASS, FAIL} truth table map to inadmissible verdicts (PAPER.md "Why this is not runnable as-is").

## Multi-bucket fire detection

This experiment fires two F#666-pure sub-flavor buckets simultaneously:
- Routing-accuracy sub-flavor — 3rd instance (F#703 1st, F#710 2nd).
- Infrastructure-benchmark / latency-only sub-flavor — 2nd instance (F#734 K-component 1st).

Per F#714 precedent (first multi-bucket fire, 2 buckets simultaneously), multi-bucket does not multiply the kill — the structural defect is per-experiment, not per-KC. But it is taxonomically informative: this is the 2nd cross-pollination of routing-acc and infrastructure-benchmark sub-flavors in a single pre-reg.

## Hygiene-defect cross-check (DB-verified)

`experiment get exp_routing_latency_benchmark_all` returned:

```yaml
success_criteria: [] # MISSING
platform: ~          # null
references: []
# ⚠ INCOMPLETE: missing success_criteria, platform
```

Three hygiene defects (SC + platform + refs). Crosses AP-prereg-hygiene-multi-defect threshold. F#666-pure structural is sufficient for kill independent of hygiene count.

## Sub-case classification

| Sub-case                            | Precedents                                                                                   | This?                  |
| ----------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- |
| F#669 classic (parent-unverified)   | F#669, F#687, F#699, F#727, F#728, F#729, F#737-F#741                                        | no (`depends_on: []`)  |
| F#669 + F#666 compound              | F#698, F#722, F#728, F#729, F#730                                                            | no (no parent)         |
| **F#666-pure standalone**           | F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, F#714, …, F#734               | **yes (≥23rd)**        |
| Multi-bucket F#666-pure             | F#714 (first), F#728-F#730                                                                   | **yes (multi-bucket)** |
| Routing-accuracy sub-flavor         | F#703 (1st), F#710 (2nd)                                                                     | **yes (3rd)**          |
| Infrastructure-benchmark sub-flavor | F#734 K-component (1st)                                                                      | **yes (2nd)**          |
| Hygiene-multi-defect (≥3)           | F#700, F#701, F#702, ...                                                                     | **yes (3 defects)**    |

## Researcher-vs-reviewer alignment

Researcher (this iteration) and reviewer (this self-pass) reach the same verdict via independent paths:
- Researcher path: claim → KC inspection → recognized routing-match-rate proxy + latency-only proxy → cross-checked drain-window taxonomy → wrote preempt scaffold + MATH theorem.
- Reviewer path: started from results.json verdict → independently verified KC text vs DB → independently verified F#666 guardrail 1007 enumeration includes K1930's metric kind → independently verified F#702 latency-pair precedent disambiguates K1929 → confirmed multi-bucket sub-flavor placement.

## Caveats / red-team

- "What if K1929's 10ms threshold is actually paper-grounded as a behavioral target (e.g., real-time UX SLA)?" — The DB notes field does not cite any arxiv paper or product spec. Even if it did, real-time-UX-latency is an engineering constraint, not a behavioral *outcome*; F#702 precedent is the canonical pattern for making latency runnable (pair with output-equivalence or task-accuracy invariant).
- "What if K1930's 80% threshold IS the target — a Pareto-quality benchmark winner?" — The framing "winner takes the default" treats the proxy as the verdict. F#666 explicitly forbids this: F#666 canonical case showed 40.2% routing-acc passes the behavioral test and 0.0% target gap is achievable through semantic-cluster routing rather than per-sample correctness. Match-rate is decoupled from utility; thresholding on it is unsafe.
- "Could we patch in-place by adding a target KC?" — Post-claim KC mutation is antipattern-u. Edits must happen externally (DB pre-reg modification) before re-claim. Recommendation: close pre-reg as structurally-malformed; re-register `exp_routing_latency_benchmark_all_behavioral` per PAPER.md follow-up template.

## Verdict-consistency pre-flight (researcher.md §6 6-item checklist)

1. `results.json["verdict"]` = "KILLED" — **OK** (target verdict for `--status killed`).
2. `results.json["all_pass"]` = `false` — **OK** (consistent with killed).
3. PAPER.md verdict line contains "KILLED — preempt-structural, F#666-pure..." — **OK**.
4. `is_smoke` = `false` — **OK** (preempt is not smoke; full structural verdict).
5. KC git-diff: KCs preserved verbatim from DB; no post-claim modification — **OK**.
6. Antipattern memories scan: no composition math (no MLX), no unsafe LORA_SCALE (no LoRA), no tautological routing (no routing classifier trained), no `shutil.copy` (no adapters touched), no hardcoded `"pass": True` (both KC `untested`), no eval truncation, no proxy-model substitution, no smoke-as-full — **OK**.

## Approve

Reviewer hat may close the experiment with `experiment complete <id> --status killed --dir micro/models/exp_routing_latency_benchmark_all/ --k 1929:fail --k 1930:fail --evidence "K1929 + K1930 untested-preempt; F#666-pure standalone multi-bucket (routing-acc 3rd + infra-bench 2nd)"`.
