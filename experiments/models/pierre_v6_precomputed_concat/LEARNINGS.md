# LEARNINGS — Pierre v6: Precomputed Concatenated Deltas

**Experiment:** `exp_pierre_v6_precomputed_concat` | **Verdict:** KILLED | **Date:** 2026-04-17

## Core Finding

Dispatch count is the **wrong optimisation axis** for BitNet-2B side-paths at d=2560.
Reducing Metal dispatches 420 → 60 (86% fewer) raised tok/s from 77 (v5) to 86.8
(v6) — only +13%, still 38% slower than the 140.8 tok/s native ceiling and short
of the 100 tok/s gate (K742 fail). Behavioral overall collapsed to 0.315 vs
v3's 0.41 (K743 fail), falsifying the proof's "bit-exact equivalence to v3"
prediction. Memory passed comfortably (K744). Two disclosed antipatterns
(tautological routing :155/:172, LORA_SCALE=20 :44) are real but the kill is
robust to both — speed is routing-independent and the router agreed with
ground truth on all 5 domains. (Finding #559)

## Why

Per-dispatch FLOP cost rose **faster** than dispatch count fell. The full-rank
ΔW (d×d = 2560²) replacing two rank-16 multiplies (d×r + r×d) is ~7× more FLOPs;
Metal's launch overhead is not the binding constraint on a side-path of this
size. The "bit-exact" framing also fails empirically because bf16 precompute +
LORA_SCALE=20 + 60-vs-420 dispatch rounding accumulate differently —
algebraic associativity holds, numerical equivalence does not.

## Implications for Next Experiment

1. **Pierre v7 must target FLOP count, not dispatch count.** Candidate:
   keep rank-r factors `A`, `B` separate; fuse only within QKV (~120 dispatches,
   ~4× FLOP reduction). Do not chase dispatch reduction further.
2. **Treat the v3–v6 family as repo-wide tautology-bug carrier** (Finding #553).
   Any v7+ experiment must implement per-sample routing and assert
   `ppl.composed != ppl.single` byte-inequality as a sanity check before
   measuring quality.
3. **Drop LORA_SCALE to 1.0** (or cite paper) before any speed/quality claim.
   v5/v6 inherited scale=20 by copy-paste (mem-antipattern-003).
4. **Rebuild missing infrastructure as a separate upstream experiment.**
   `pierre.v6` module, SFT adapter bank, grassmannian skeleton are absent;
   bundling recovery with a new speed claim (audit lesson) masks bugs.
5. **Pierre v7 is now unblocked** by this kill. Forward path per PLAN.md Part 2.
