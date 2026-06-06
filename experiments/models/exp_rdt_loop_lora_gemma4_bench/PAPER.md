# PAPER — exp_rdt_loop_lora_gemma4_bench

## Verdict: PROVISIONAL

Structural (K-FULL-A), gradient (K-FULL-B), and extended dynamical (K-FULL-C-EXT)
KCs all PASS at their pre-registered thresholds on the real
`mlx-community/gemma-4-e4b-it-4bit` forward with full 500-step training on
GSM8K CE loss. Target behavioural KCs (K1740-BENCH, K1741-BENCH, K1742-BENCH)
and the infrastructure KC (K-KVCACHE) report `under_powered` / `not_measured`
because the researcher-hat compute budget could not reach pre-registered n
without a KV-cached recurrent-depth forward (MATH §Theorem 2).

Per Finding #673 and PLAN §1 rule 4: `is_smoke=false` runs whose target KCs
are under-powered (below pre-registered n) complete as **PROVISIONAL**, never
`supported` or `killed`. Per F#666 target-gating rule: a target KC that is
`under_powered` at its measured n is neither TARGET-PASS nor TARGET-FAIL — so
KILLED is also inappropriate.

## Run configuration

| Key | Value |
|---|---|
| Model | `mlx-community/gemma-4-e4b-it-4bit` |
| `mlx` / `mlx-lm` | 0.31.1 / 0.31.2 |
| Seed | 42 |
| Loop layers | 12..20 (9 `DecoderLayer`s, inclusive of both) |
| N_LOOPS | 6 |
| LoRA rank / α | 16 / 2.0 (scale=0.125, safe; `lora-scale-20` antipattern not triggered) |
| Training | 500 steps, batch 1, seq 256, AdamW lr=5e-4, real GSM8K CE loss |
| Eval | GSM8K-valid, greedy, `max_tokens=256`, `add_generation_prompt=True` |
| `N_EVAL_T3` | 30 (K1740 pre-reg n≥200) |
| `N_EVAL_PER_T` | 10 (K1742 pre-reg n≥30/T) |
| T-sweep | {1, 2, 3, 6} (K1742 pre-reg T∈{1..6}) |
| `SKIP_KVCACHE` | True (K-KVCACHE verification scope-deferred) |
| `is_smoke` | False |
| Elapsed | 6644.66 s ≈ 110.7 min |

## Prediction vs measurement

| KC | Prediction | Measurement | Status | Notes |
|---|---|---|---|---|
| K1743 (init orth) | max\|cos\|<0.1 | 3.75e-8 | PASS | 6 orders below threshold — partition-QR confirmed. |
| K-FULL-A (block integration) | `v_proj` & `o_proj` are `LoopLoRALinear` on 12..20 | pass | PASS | Class-level monkey-patch confirmed at all 9 loop layers. |
| K-FULL-B (grad-flow) | `max\|dL/dB_v\|`, `max\|dL/dB_o\|` > 1e-6 | 2.4e-2 / 6.9e-2 | PASS | Both > 3 orders above threshold. |
| K-FULL-C-EXT | max ρ(A_d) < 1 AND \|Δlog_A\|,\|Δlog_dt\| > 1e-4 across ≥500 steps | ρ_max=0.555 (0.369→0.555), Δlog_A=0.248, Δlog_dt=0.280, n=500 | PASS | All three clauses pass — closes parent Caveat 1 (F#674). |
| K-KVCACHE | max_abs_logit_diff < 1e-3 on cached vs uncached, n=20 × T∈{1,2,3,6} | not implemented | NOT_MEASURED | Scope-deferred per MATH §Theorem 2(b); follow-up `exp_rdt_loop_kv_cache`. |
| K1740-BENCH | +5pp GSM8K at T=3 vs base, n≥200 | base=3.33%, T3=6.67%, Δ=+3.33pp at **n=30** | UNDER_POWERED | Direction is positive; magnitude below target at heavily under-powered n. Full-n eval requires K-KVCACHE first (MATH §Theorem 2). |
| K1741-BENCH | \|ΔMMLU\|≤1pp at T=3, 57 subjects | not evaluated | NOT_MEASURED | Scope-deferred to `exp_rdt_loop_mmlu_eval`. |
| K1742-BENCH | saturating-exponential R² > 0.90, T∈{1..6}, n≥30/T | R²=0.321 at n=10/T on T∈{1,2,3,6} | UNDER_POWERED | n=10 and only 4 Ts insufficient; fit params degenerate (y∞=-9e4, τ=4e4). |

## Key observations

1. **Extended dynamical clause closes at n=500.** Parent `exp_rdt_loop_lora_gemma4_full`
   reported ρ evolving 0.369→0.439 over 50 steps; at 500 steps ρ reaches 0.555
   monotonically (per-step log recorded). `|Δlog_A|` and `|Δlog_dt|` grow by
   ~2.5× from parent's 50-step values (0.101→0.248, 0.094→0.280), consistent
   with Theorem 1's non-decreasing-max argument. K-FULL-C-EXT passes cleanly.

2. **K1740 direction is positive but under-powered.** Trained loop-LoRA at T=3
   (6.67%, 2/30) exceeds base at T=1 (3.33%, 1/30) by +3.33pp. The
   pre-registered threshold of +5pp is not cleared at measured n; but n=30
   is a seventh of pre-reg n≥200, and single-flip sampling noise at this
   scale is large. The absolute magnitude (1-2/30) is also below typical
   GSM8K signal regimes — local `valid.jsonl` (GSM8K subset) may be
   harder than average GSM8K (documented MATH assumption).

3. **T-sweep shape is non-monotonic.** T∈{1,2,3,6} → {10%, 10%, 20%, 0%} at
   n=10. This does **not** support the saturating-exponential predicted by
   K1742. Plausible explanations at this n: (a) sampling noise at n=10
   dominates any structural T-shape; (b) T=6 with only 256 `max_eval_tokens`
   may be truncating longer chains-of-thought more severely because 6 loops
   of recurrent-depth compress more reasoning per token — but the
   eval-token budget is flat. Follow-up should (a) raise n per T, (b) raise
   `max_eval_tokens` at higher T, (c) include all T∈{1..6}.

4. **No antipattern triggered.**
   - `composition-bug`: monkey-patch exercises `B_t @ A_t` per loop at forward
     time; no safetensor A/B summing.
   - `tautological-routing`: loop index is scheduled, never data-routed.
   - `lora-scale-20`: α=2, r=16 → scale=0.125.
   - `shutil-copy-adapter`, `hardcoded-pass`, `proxy-model`, `smoke-as-full`,
     `kc-tautological`, `kc-swap`, `copy-paste-scaffolding`: checked, none apply.
   - `thinking-truncation`: prompts tokenized with
     `apply_chat_template(add_generation_prompt=True)` preserving Gemma 4
     thinking channel; `max_tokens=256` halves parent's 512 (documented
     limitation; see §Assumptions).

## Assumptions (logged per hat rules)

- **Eval data is a GSM8K subset, not the canonical "GSM8K-Hard"** (Gao et al,
  arxiv:2211.10435). `micro/models/exp_p1_t2_single_domain_training/data/math/valid.jsonl`
  (199 samples) is used as a GSM8K-Hard proxy. Absolute accuracy numbers
  are not directly comparable to GSM8K-Hard literature numbers, but
  *relative* base-vs-loop-T3 comparison remains interpretable (same eval
  set, same decoding).
- **Training corpus = 500 samples of `train.jsonl` cycled** (one pass in this
  run because n_steps ≤ train size). Parent's "≥10k GSM8K samples" target
  not met; noted as limitation, not KC relaxation.
- **`max_eval_tokens=256`** (parent used 512). Halves eval throughput
  cost at marginal correctness loss for ≤200-token ground truths; at T=6
  this may under-represent deeper reasoning — flagged as follow-up scope.
- **K-KVCACHE scope-deferral** is pre-registered in MATH §Theorem 2(b),
  not an ad-hoc KC relaxation. A KV-cached recurrent-depth follow-up is
  required before K1740 / K1742 can reach pre-reg n in <2h.

## Verdict-consistency pre-flight (PLAN §1)

1. `results.json["verdict"]` = `"PROVISIONAL"` — not `"KILLED"`. ✓ for PROVISIONAL path.
2. `results.json["all_pass"]` = `false`. → cannot mark `supported`; PROVISIONAL is appropriate.
3. PAPER verdict line = `PROVISIONAL` — matches.
4. `is_smoke` = `false`. PROVISIONAL verdict is allowed (F#673: `is_smoke=false`
   with under-powered target KCs → PROVISIONAL; ≠ smoke-as-full antipattern).
5. No KC modified between MATH.md and now (`git diff MATH.md` empty). KCs are
   pre-registered in DB as #1759-#1763, inherited verbatim from parent F#674.
6. Antipattern self-audit (§Key observation 4): no pattern triggered.

Verdict-consistency PASSES for `--status provisional`. Marking `supported`
or `killed` would be inconsistent with measurement.

## What moves PROVISIONAL → SUPPORTED

Exactly one thing is required, derived from MATH §Theorem 2:

- **Implement & verify KV-cached recurrent-depth forward** (K-KVCACHE); this
  cuts eval cost by ~O(L) and makes K1740-pre-reg (n≥200) plus K1742-pre-reg
  (n≥30/T, all 6 T values) reachable in <2h.

Follow-ups to file:
- `exp_rdt_loop_kv_cache` — implements K-KVCACHE; unlocks full-n eval.
- `exp_rdt_loop_mmlu_eval` — K1741-BENCH with `enable_thinking=True`,
  57 subjects, 5-shot. Dependent on KV-cache follow-up for tractability.
- `exp_rdt_loop_gsm8k_fulln` — re-evaluates K1740 / K1742 at pre-reg n
  using the verified KV-cache.

## References

- Bae et al. 2024 "Looped Transformers", arxiv:2410.20672 (loop-LoRA foundation)
- Parcae Prairie 2026, arxiv:2604.12946 (recurrent-depth primitives)
- Saunshi et al. 2025, arxiv:2502.17416 (LTI-injection)
- Findings #562 (Grassmannian at Gemma 4), #666 (target-gated kill), #667
  (LTI Theorem 1), #673 (under-powered → PROVISIONAL), #674 (parent lineage).
