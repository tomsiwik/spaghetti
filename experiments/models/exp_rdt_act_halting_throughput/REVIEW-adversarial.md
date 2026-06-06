# REVIEW-adversarial — exp_rdt_act_halting_throughput

Reviewer-authored (replaces researcher self-review).

## Verdict
**KILL — ratified preemptive kill.** DB already `killed`; F#669 filed.
No further DB action required.

## Adversarial checklist

**Consistency (a)-(d):**
- (a) results.json verdict=`KILLED`, DB status=`killed`. Aligned.
- (b) all_pass=false, 4/4 KCs ✗. Consistent.
- (c) PAPER.md verdict: "KILLED (preemptive, dependency-unfulfilled)". Aligned.
- (d) is_smoke=false, executed=false. Preemptive path — no smoke/full-run
  mismatch. Aligned.

**KC integrity (e)-(g):**
- (e) MATH.md is a new file; KCs K1745-K1748 match the DB pre-reg. No
  post-hoc relaxation.
- (f) Tautology sniff: KCs are target behavioral metrics (halt-fraction
  on easy/hard, tok/s, quality-match). No self-reference. Preempt is
  *inter-experiment* dep-unfulfilled (F#513/F#558 family), not F#498/F#666
  intra-experiment tautology.
- (g) KC wording vs code: N/A — nothing measured.

**Code ↔ math (h)-(m2):**
- (h)-(m) All N/A: no composition, no LoRA train, no routing, no copy,
  no hardcoded pass, no proxy substitution.
- (m2) Skill invocation: no platform code written; `/mlx-dev` not required
  for preempt stub. PAPER.md §Skill-invocation states this explicitly.
  Acceptable.

**Eval integrity (n)-(t):**
- (n)-(q) N/A (no eval run).
- (r) PAPER.md prediction-vs-measurement table present for all 4 KCs.
- (s) Theorems audited:
  - T1 (child KCs require parent target SUPPORTED): KC-by-KC reduction
    is sound. K1745 and K1746 both reduce to ∂loss/∂T signal which
    depends on parent K1742 (saturating-exp in T). K1747 and K1748 depend
    on K1740 (+5pp GSM8K-Hard). All four untested ⇒ unmeasurable child.
  - T2 (preempt is the correct action): unidentifiability argument is
    valid — either trivial-pass or trivial-fail without distinguishing
    the halter mechanism from parent's absence of depth-adaptation.
- (t) Target-gated rule: KCs ARE target metrics (behavioral halt fractions,
  throughput, quality-match), not proxies. F#666 target-gated rule
  satisfied in spirit: kill does not rest on a proxy metric. The
  operational status is `not_measurable`, which under Finding #666 could
  warrant PROVISIONAL rather than KILL — but per preempt convention
  (F#513/F#558 family), KILL is the correct DB label when the parent's
  target claim has not been established. This aligns with F#669's
  precedent for this exact experiment class.

## Assumptions logged
- F#669 adequately captures the new sub-axis
  `preempt-child-KCs-require-parent-target-claim-unverified`. No
  additional finding needed for this ratification.
- Parent `exp_rdt_loop_lora_gemma4` remains smoke-provisional on disk
  (DB CLI-killed as rule-4 label artifact). If parent is subsequently
  resurrected via `exp_rdt_loop_lora_gemma4_full` with K1740/K1741/K1742
  SUPPORTED, this experiment becomes reclaimable per MATH.md §Unblock.

## Non-blocking observations
1. Self-review in the prior version of this file was complete and
   accurate. The reviewer-authored version exists only because the
   reviewer hat's ratification is required per workflow — substantive
   findings do not differ.
2. Evidence history on the experiment shows two `[fail]` entries
   (researcher's PREEMPTIVE-KILL note + F#669 ratification). Both
   authentic; no duplication cleanup needed.

## Routing
`review.killed` — hand to Analyst for LEARNINGS pass. No further DB
mutation; no new findings required.
