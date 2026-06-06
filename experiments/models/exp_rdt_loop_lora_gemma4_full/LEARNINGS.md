# LEARNINGS — exp_rdt_loop_lora_gemma4_full

## Core Finding

First end-to-end validation of recurrent-depth (Bae 2024) loop-indexed
LoRA on the *real* product target (`mlx-community/gemma-4-e4b-it-4bit`,
layers 12..20, T=3). Structural + dynamical KCs PASS (K-FULL-A isinstance
on 18 modules; K-FULL-B grad magnitudes 2.41e-2 / 6.92e-2; K-FULL-C max
ρ=0.439 with Δlog_A=0.101, Δlog_dt=0.094 over 50 real-CE steps). Target
behavioural KCs K1740/K1741/K1742 `not_measured` — verdict **PROVISIONAL**
per rule (t) / F#666 / F#673.

## Why

- **Architecture wires on real quantised weights.** Class-level monkey
  patch of `Gemma4TextModel.__call__` is required (instance patch is
  silently ignored because `type(obj).__call__` is how Python resolves).
  Per-layer v/o dim mismatch (L17 full-attention 1024/4096 vs others
  sliding 512/2048) must be sized explicitly — smoke hardcoding would
  have silently broken here.
- **Gradient-underflow hypothesis from smoke is refuted.** Under real
  CE loss on GSM8K, log_A / log_dt drift 3 orders of magnitude above
  the 1e-4 threshold; ρ drifts 0.369→0.439. The smoke artefact (static
  LTI params) was an MSE-proxy pathology, not a structural dead end.
- **K1743 partition-QR orthogonality extends to RDT.** max|cos|=3.75e-8
  across 270 loop-pair comparisons — Pierre F#562 primitive holds at
  rank-16 loop-indexed composition on Gemma 4 native dims.
- **Target KCs deferred for real reasons, not skipped.** Uncached greedy
  gen with T-looped forward is ~T× per-token cost; 200 × 512 × 6 greatly
  exceeds a researcher-hat cycle. MATH §Theorem 4 pre-registered this.

## Implications for Next Experiment

1. **exp_rdt_loop_lora_gemma4_bench (macro P1, already filed) is the
   binding test of the thesis** — K1740/K1741/K1742 remain the behavioural
   gate. Infrastructure from `_full` is reusable as-is.
2. **KV cache for recurrent-depth forward is the critical-path blocker.**
   Without it, benchmark wall-clock is infeasible. Strategy: cache the
   block-entry hidden state per loop iteration; reuse standard KV for
   layers 0..11 and 21..41. File `exp_rdt_loop_kv_cache_strategy`
   (micro P2) as prerequisite if bench hits wall-clock ceiling.
3. **Extend training to ≥10k GSM8K samples, ≥500 steps** to close the
   K-FULL-C 500-step clause in the same run as bench eval.
4. **Do NOT kill or re-litigate K-FULL-A/B/C.** They PASS on the real
   target; scope reduction (n=50) is transparent, monotone, and 3 orders
   above threshold. Follow-up extends, does not replace.
5. **F#666 target-gating continues to protect structural experiments
   from premature kills** — proxy PASS + target not_measured = PROVISIONAL,
   not KILL. Pattern should keep being reused for architecture-wiring
   experiments whose behavioural payoff lives downstream.
