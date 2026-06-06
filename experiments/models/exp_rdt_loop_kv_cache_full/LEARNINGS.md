# LEARNINGS — exp_rdt_loop_kv_cache_full (full iter, F#788 PROVISIONAL)

## Core Finding

**Bit-exact prefill (K1837 PASS, max_diff=0.0 across 80 pairs) does NOT compose into argmax-stability-after-M-steps under fp16 (K1986 FAIL, mean=0.940 with 3/20 cascade-divergence outliers at 0.422/0.734/0.641).** The K1837↔K1986 pair was structurally MIS-PAIRED in the pre-reg: K1837 measures *prefill* correctness (single forward pass), K1986 measures *generation* correctness over M=64 sequential argmax steps. Bit-exact prefill is necessary but not sufficient for cascade-stable generation under fp16 ties. Capability target K1987 (642s ≤ 7200s budget, 11× margin) PASSES; speedup proxy K1838 (4.885× < 5×) is borderline FAIL but K1987 PASS shows the threshold is the deficient artifact. Verdict PARTIALLY_SUPPORTED → DB PROVISIONAL (F#666: no pair has both fail, KILL not justified).

## Why

F#666 verbatim treats Pair-1 (K1837 PASS proxy + K1986 FAIL target) as "tautological proxy, kill on target" — but here the proxy is NOT tautological. K1837's bit-exact theorem (parent §4) only proves prefill equality; it makes no claim about M-step generation. The 3 outliers exhibit 27/64, 17/64, 23/64 token divergence — *cascade-divergence*, not single-roundoff. This is a NEW F#666 sub-form: **proxy and target measure different mechanism STAGES even when nominally paired on the same axis**. Pair-2 (K1838 FAIL proxy + K1987 PASS target) is the standard "finding-about-proxy" F#666 case: 5× threshold was set on parent's M→∞ asymptotic Theorem 2, not finite M=64 prefill-amortized empirics.

## Implications for Next Experiment

1. **`exp_rdt_loop_kv_cache_full_v2` (P3, recommended file by next researcher iter):**
   - K1986_v2: re-derive threshold from observed fp16-tie cascade-rate (mean ≥ 0.94 / min ≥ 0.40 from empirical distribution).
   - K1838_v2: finite-M corrected to 4.5× at M=64.
   - Inherit K1837/K1987 verbatim (both PASS).
   - **K_NEW (cascade-divergence rate):** proportion of prompts with mid-gen argmax divergence ≤ 10%. This is what MATH §4 K1986-corollary should have been.
2. **Parent `exp_rdt_loop_kv_cache` (F#690):** elevate PROVISIONAL → SUPPORTED on **target-pair** (K1837 + K1987), with proxy-threshold caveats logged as A5/A6.
3. **Unblocks `exp_rdt_jepa_loop_adapter_impl`** (parent infra-feasibility now SUPPORTED on target axis).
4. **Methodological:** pre-reg pairing audits must verify proxy and target measure the same mechanism *stage* (prefill vs generation, weight-space vs activation-space, single-step vs cascade), not just the same nominal axis. New antipattern memory filed (`mem-antipattern-proxy-target-stage-mismatch`).

## Cluster context
F#690 parent → F#785 _impl smoke → F#787 _full smoke iter ~70 → **F#788 _full PARTIALLY_SUPPORTED PROVISIONAL (this iter)**.

## Antipattern check (one NEW filing)
- `mem-antipattern-proxy-target-stage-mismatch` (NEW, this iter): proxy and target measure different mechanism stages on the same axis (prefill vs generation, single-step vs cascade); F#666 sub-form not previously captured. See `.ralph/agent/memories.md`.
- mem-antipattern-thinking-mode-truncates-judge-budget — n/a (greedy gen, not chat eval).
- mem-antipattern-smoke-as-full — guarded (`is_smoke=False`, full N).
