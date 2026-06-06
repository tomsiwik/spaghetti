# REVIEW-adversarial.md — exp_g4_per_token_top2_routing

## Verdict
**KILL** — pre-registered precondition probe, 3/3 preconditions FAIL, K1578 unmeasurable.

## Summary
Researcher ported Finding #58 (BitNet per-token top-2) to Gemma 4 E4B N=25 as a
precondition probe. All three pre-registered preconditions (P1 adapter weights,
P2 per-token router, P3 exclusive_PPL baseline) fail on disk. K1578 routes to
FAIL-unmeasurable per MATH.md §Preconditions. No heavy training attempted — 7th
precondition-probe KILL in audit-2026-04-17 cohort, consistent with standing rule.

## Adversarial checklist

| Check | Result |
|---|---|
| (a) results.json verdict vs DB status | KILLED ↔ killed — **consistent** |
| (b) all_pass=false vs KILLED | **consistent** |
| (c) PAPER.md verdict line | "KILLED" — **consistent** |
| (d) is_smoke false, ran false | **consistent** (probe, not smoke) |
| (e) MATH.md post-reg diff | **single commit** (3693362) — no post-hoc edits |
| (f) Tautology sniff | K1578 FAIL with explicit "unmeasurable" reason; no identity PASS |
| (g) K-ID text vs DB | `routed_PPL < 0.95 * exclusive_PPL on 5 domains` — **matches DB** |
| (h) sum(lora_A) / linear add_weighted / summed safetensors | absent (probe-only) |
| (i) LORA_SCALE ≥ 12 hardcoded | absent |
| (j) single-sample routing | N/A (probe didn't route) |
| (k) shutil.copy sibling adapter | absent |
| (l) hardcoded `{"pass": True}` | absent; probe statuses computed from disk |
| (m) target model ≠ loaded model | no model loaded (probe-only) |
| (m2) skill invocation evidence | N/A (no MLX code path executed) |
| (n) base-acc=0 w/ thinking=0 | N/A |
| (o) headline n<15 | N/A (no eval) |
| (p) synthetic padding | explicitly rejected in MATH.md — key anti-pattern avoided |
| (r) prediction-vs-measurement table | present in PAPER.md |
| (s) Math/unsupported claims | Theorems A–D correctly cited; Theorem C (shared-KV null) used to reject TF-IDF substitution fallback, which is the principled call |

## Why KILL and not REVISE
The KILL is a *measured outcome* of a pre-registered routing, not a skip. The
researcher could have synthesized P1/P2/P3 (random-init adapters, TF-IDF fallback
as "per-token") to force a PASS, which would be the antipattern MATH.md §Assumptions
explicitly forbids (Finding #305 Theorem C). Instead they invoked the cohort standing
rule and routed honestly to FAIL-unmeasurable. This is the correct researcher behavior.

## Unblock path (for analyst/LEARNINGS.md)
Single upstream fix unblocks the cohort: rerun `exp_p1_t2_single_domain_training`
at LORA_SCALE=5 (Finding #586) to regenerate math/code/medical adapters, then train
finance+legal and the Gemma 4 per-token ridge router.

## Assumptions
- DB already shows `status=killed` for this experiment — `experiment complete` was
  invoked before this review; no additional completion call needed.
- Git-log check (single commit on MATH.md) confirms no post-hoc KC relaxation.
