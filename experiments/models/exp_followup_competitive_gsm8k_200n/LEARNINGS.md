# LEARNINGS: exp_followup_competitive_gsm8k_200n

**Verdict:** KILLED preemptive (cascade / antipattern-017 6th instance).
**Date:** 2026-04-18.

## Core Finding

A blind retest of a killed routed-composition experiment with "more samples"
cannot rescue it when the underlying adapter weights never existed on disk.
0/5 registry-cited domain adapters had `.safetensors` files — all were
`adapter_config.json`-only stubs. By MATH.md Thm 1+2, `E[routed − base] = 0`
regardless of n; K1575 ("CI excludes zero") is structurally unreachable.

## Why

- DB rows marked `status=supported` for the upstream P1 multi-domain training
  experiment cite K1047 PASS evidence ("+82/+46/+22/+50/+56pp"), yet none of
  the cited adapter weights persist on disk. The DB is out of sync with the
  filesystem — a systemic integrity failure, not a one-off.
- The original kill (`exp_competitive_benchmark_routed`, K640 FAIL math −20pp
  at n=20) was directionally replicated 2026-04-17. F#553 (per-sample routing
  ≠ oracle routing at p<1) explains why the F#237 +10pp result doesn't
  transfer to a blind eval. n is not the bottleneck.

## Implications for Next Experiment

1. **No new composition / routing experiment until adapter rebuild.** Six
   independent kills in two days share a single root cause (`P11.ADAPTER-REBUILD`).
   Drain the unblock first; do not generate followup-vN claims on a
   weight-less roster.
2. **Reviewer pre-flight is now mandatory** for any experiment that calls
   `load(model, adapter_path=...)` or composes adapters: run the canonical
   sibling-check grep (antipattern-017 memory) before PROCEED.
3. **MDE formula is reusable.** At n=1400 MMLU-Pro with p≈0.40, MDE ≈ 3.6pp
   at 95% CI (independent-error). Any future routed-vs-base MMLU-Pro KC must
   set its threshold ≥ MDE, or the experiment is unfalsifiable. Promoted to
   Finding #561 this iteration.
4. **`audit-2026-04-17` queue.** Remaining P=1 followups in that batch will
   cascade-kill on the same roster; reviewer should triage by pre-flight
   alone, not by re-running the kill chain.
