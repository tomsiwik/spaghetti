# LEARNINGS — exp_followup_grassmannian_native_macro

## Core Finding

16th consecutive audit-2026-04-17 cohort precondition-probe KILL. K1550
UNMEASURABLE: P2=FAIL (0 Grassmannian-AP init markers) and P3=FAIL (upstream
`exp_p1_t2_single_domain_training` status=killed; K1030 metric-swap, K1028
format-artifact) independently force `all_pass=false` → killed. P1 loose-glob
false-positive (15 non-target pre-Gemma-4 hits) does not flip the verdict and
is documented openly in PAPER §Probe-bias. Wall 1.054s. No MLX model load.

## Why

The over-packed Grassmannian orthogonality claim from `killed_19.md` was
measured on a d≤2048 proxy. Extending to Gemma 4 E4B (d=3072) / Qwen3-4B
(d=2048) requires real trained target-model adapters, which require the
upstream T2.1 rerun. Until that upstream is `supported`, the orthogonality
claim is measurement-gated — not wrong, just not testable on the target
platform. Same blocker as Findings #605/#606/#608/#610/#611/#612/#613/#615/
#616/#617/#618/#619/#620/#621/#622/#624 (15 prior cohort siblings + this one).

## Implications for Next Experiment

1. **Upstream-reclaim obstacle is real and new.** Researcher iter-9 confirmed
   `experiment claim exp_p1_t2_single_domain_training` refuses with "Cannot
   claim — status is killed, not open". This means the highest-leverage
   single action (rerun upstream → unblock 15+ cohort siblings + 2 stale
   followups) is **mechanically blocked for researcher hats**. Orchestrator
   must either (a) design `exp_p1_t2_single_domain_training_v2` with
   preserved KCs + pre-registered code-bug fixes for K1028/K1030, OR (b)
   permit `experiment update --status open` on killed upstream when the
   fix is code-scoped (guardrail 1009 must still forbid silent KC changes).

2. **Cohort claim-queue filter remains unapplied after 9 analyst escalations.**
   Eight prior escalations + this one = 9 total. Until a
   `tag=audit-2026-04-17` filter is wired into the claim queue OR the
   upstream is resurrected, researcher hats will keep probe-KILLing cohort
   members (bounded work, but burns iterations).

3. **Probe-precision vs. probe-speed tradeoff validated at N=16.** Across
   16 cohort members, P3 (upstream killed) independently determined every
   verdict. Loose P1 globs that produce false-positives are acceptable in
   this regime — they do not affect the outcome and cost <1 ms. Next
   researcher should continue the established tripwire pattern; tightening
   P1 is deferred until the upstream is resurrected.

4. **Next researcher action priority (ranked):**
   - (a) Attempt `experiment update --status open exp_p1_t2_single_domain_training`
     if CLI permits; if it succeeds without silent KC change, reclaim and
     fix. If CLI refuses, proceed to (b).
   - (b) Design v2 experiment with new ID + preserved KCs.
   - (c) If (a)/(b) are out of scope, probe-KILL the 17th cohort member
     in <10s with the established pattern.

5. **No antipattern to add.** REVIEW 17/17 PASS/N-A. `ap-017`
   (cohort-saturation-upstream-blocker) already covers all 16 instances.
