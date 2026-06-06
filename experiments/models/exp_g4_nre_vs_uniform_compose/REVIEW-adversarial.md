# REVIEW-adversarial.md — exp_g4_nre_vs_uniform_compose

## Verdict
**KILL** — pre-registered precondition probe, 3/3 preconditions FAIL, K1579 unmeasurable.

## Summary
Researcher ported Finding #275 (NRE Karcher mean on Qwen3-4B) to Gemma 4 E4B N=5
GSM8K as a precondition probe. All three pre-registered preconditions (P1 five
adapter safetensors, P2 upstream training not KILLED, P3 base GSM8K ≥ 20% at
max_tokens ≥ 512) fail on disk. K1579 routes to UNMEASURABLE per MATH.md
§Kill-criterion. No heavy training attempted — 8th precondition-probe KILL in
the audit-2026-04-17 cohort, consistent with standing rule.

## Adversarial checklist

| Check | Result |
|---|---|
| (a) results.json verdict vs DB status | KILLED ↔ killed — **consistent** |
| (b) all_pass=false vs KILLED | **consistent** |
| (c) PAPER.md verdict line | "KILLED (precondition-probe …)" — **consistent** |
| (d) is_smoke false | **consistent** (probe, not smoke) |
| (e) MATH.md post-reg diff | files untracked; no commits → no post-hoc KC relaxation possible. KC text in MATH.md matches results.json K1579 text — **consistent** |
| (f) Tautology sniff | K1579 FAIL with explicit "unmeasurable" reason; three independent disk/upstream/accuracy probes, no identity PASS |
| (g) K-ID text vs DB | `acc(NRE) − acc(1/N) ≥ 3pp on GSM8K N=5` — **matches DB #1579** |
| (h) sum(lora_A) / linear add_weighted / summed safetensors | absent (probe-only, no composition executed) |
| (i) LORA_SCALE ≥ 12 hardcoded | absent; MATH.md registers LORA_SCALE=5 per Finding #586 |
| (j) single-sample routing | N/A (no routing) |
| (k) shutil.copy sibling adapter | absent |
| (l) hardcoded `{"pass": True}` | absent; probe statuses computed from `Path.exists()`, upstream JSON parse, numeric threshold |
| (m) target model ≠ loaded model | target = Gemma 4 E4B; no model loaded (probe-only), paths check Gemma 4 dirs — **consistent** |
| (m2) skill invocation evidence | N/A (no MLX forward/backward executed; probe is pure disk I/O) |
| (n) base-acc=0 w/ thinking=0 | P3 explicitly surfaces `base_gsm8k_pct=0.0` as FAIL (format artifact), not as a measured "gain" |
| (o) headline n<15 | N/A (no eval) |
| (p) synthetic padding | explicitly rejected — finance/legal adapters are missing, researcher did not substitute random-init placeholders |
| (q) cited baseline drift | Finding #275 cited for mechanism, not as headline number |
| (r) prediction-vs-measurement table | present in PAPER.md |
| (s) Math / unsupported claims | Findings #275, #320, #330, #586 correctly cited; NRE-vs-1/N √N attenuation argument matches Finding #275 derivation |

## Why KILL and not REVISE
The KILL is a *measured outcome* of a pre-registered routing, not a skip. The
antipattern here would be: synthesize the five adapters (random init or
`shutil.copy` from a sibling), force P1/P2/P3 to PASS, and report a fabricated
acc(NRE)−acc(1/N) delta. Researcher invoked the cohort standing rule and
routed honestly to UNMEASURABLE. This is the correct researcher behavior under
PLAN.md §1000 (proof-first) and Guardrail #1009 (verdict consistency).

## Unblock path (for analyst/LEARNINGS.md)
Single upstream fix unblocks the cohort: rerun
`exp_p1_t2_single_domain_training` at LORA_SCALE=5 with disjoint
math/code/medical corpora at max_tokens ≥ 512, then train finance+legal
adapters. Re-running this probe on the resulting safetensors flips P1/P2/P3
to PASS and unlocks the K1579 measurement branch.

## Assumptions
- DB already shows `status=killed`; no additional `experiment complete` call
  needed from the reviewer.
- Files are untracked in git; reviewer cannot diff MATH.md against a prior
  commit. KC text is cross-checked against results.json and DB instead.
- The finance/legal domains are pre-registered as the two additional disjoint
  domains; a v2 experiment may substitute other disjoint domains without
  invalidating this probe.
