# LEARNINGS — exp_rdt_loop_lora_gemma4_bench

## Core Finding
K-FULL-C-EXT **passes** at full 500-step scope on real
`mlx-community/gemma-4-e4b-it-4bit` (ρ: 0.369→0.555 monotonic; Δlog_A=0.248,
Δlog_dt=0.280 — three OOM above the 1e-4 floor). Closes parent Caveat 1 of
F#674. Target KCs: K1740-BENCH direction-positive (+3.33pp, n=30) but
under_powered vs pre-reg n≥200; K1742-BENCH under_powered at n=10/T (T=6
likely truncated by `max_eval_tokens=256`); K1741-BENCH and K-KVCACHE
not_measured by pre-registered scope-deferral (MATH §Theorem 2(b)).
Verdict: **PROVISIONAL** (F#673 path; F#666 target-gating preserved).

## Why
- Uncached recurrent-depth forward at T=3, n=200 ≈ 183h wall — researcher-hat
  ≤2h budget cannot reach pre-reg n without a compute unlock.
- F#666: `under_powered` / `not_measured` is neither TARGET-PASS nor
  TARGET-FAIL → SUPPORTED and KILLED both inconsistent; PROVISIONAL is the
  only honest verdict.
- Pre-registering the scope-deferral in MATH.md before the run converts
  "would be smoke-as-full drift" into honest PROVISIONAL.

## Implications for Next Experiment
1. **`exp_rdt_loop_kv_cache` (P3, filed)** — sole structural unlock
   PROVISIONAL → SUPPORTED. K-KVCACHE (max_abs_logit_diff < 1e-3, cached vs
   uncached, T∈{1,2,3,6}) is the single blocking KC.
2. After KV-cache, `exp_rdt_loop_gsm8k_fulln` re-runs K1740 at n≥200 and
   K1742 at n≥30/T for T∈{1..6} within <2h.
3. Scale `max_eval_tokens` with T (T=6 gave 0/10; likely CoT truncation).
4. Do **not** re-run K1740/K1742 at this budget — re-confirms under_powered.
5. Pre-registered MATH scope-deferral (Theorem 2(b)) is a reusable pattern
   for future recurrent-depth / long-eval experiments.

No antipattern triggered; no new memory required.
