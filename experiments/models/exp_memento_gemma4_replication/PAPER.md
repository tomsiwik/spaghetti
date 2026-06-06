# PAPER.md — exp_memento_gemma4_replication

## Verdict: PROVISIONAL (design-only filing)

This experiment pre-registers MATH.md for a novel-mechanism replication (MEMENTO 2-stage SFT + block-mask attention with dynamic KV eviction) on Gemma 4 E4B 4-bit MLX. The mechanism is not executable via the `mlx_lm.lora` CLI and the full pipeline exceeds the researcher-hat 30-min / 40-tool-call cap. Per `reviewer.md` §5 canonical PROVISIONAL-as-design clause (3-precedent threshold: F#682 JEPA, F#683 hedgehog_behavior, F#684 hedgehog_procedural) and `mem-antipattern-novel-mechanism-single-iteration-scope`, this filing is design-only: locked MATH.md + graceful-failure `run_experiment.py` + `results.json` with `verdict="PROVISIONAL"` and every KC `pass="untested"`. The load-bearing implementation is filed as `exp_memento_gemma4_replication_impl` (P3), inheriting MATH.md verbatim and all 4 KC IDs (#1799, #1800, #1801, #1802).

## Scope rationale

MEMENTO requires three capabilities that are not in the current MLX adapter pipeline:
1. **Tokenizer extension** (4 boundary tokens + embedding/lm_head resize) — feasible but non-standard in `mlx_lm.load()` path.
2. **Custom 2-stage full-parameter SFT** — stage 1 is standard next-token CE but must be full-parameter (not LoRA, to preserve what K2 measures — the memory burden the mechanism is meant to reduce); stage 2 requires the mask-surgery forward path from step 3.
3. **Block-mask attention with dynamic KV eviction** at inference — custom MLX generation loop with per-token `BlockMaskState` + `mx.fast.scaled_dot_product_attention(mask=...)` + selective KV-tensor eviction between blocks. Not in the `mlx-lm.generate` path.

Full pipeline estimate: 6-10h on M5 Pro 48GB (2-stage SFT 2×2000 steps + 4 KC evals + K3 ablation arm). Exceeds the 30-min researcher-hat cap.

## Prediction vs measurement

| KC  | KC id | Prediction (per MATH.md §5) | Measurement | Status |
|-----|-------|-----------------------------|-------------|--------|
| K1 (proxy, paired with K2)   | #1799 | E[KV(memento)] / E[KV(base)] ∈ [0.43, 0.50] on GSM8K-Hard n≥200 (≥2× reduction, lower than paper's 2.5× at 8B) | not measured — Phase C blocker | untested |
| K2 GSM8K drop (target)       | #1800 | GSM8K-Hard drop ∈ [2, 6]pp (may fail 5pp boundary — watch) | not measured — depends on Phase B + C | untested |
| K2 MMLU drop (target)        | #1800 | MMLU drop ∈ [1, 4]pp | not measured — depends on Phase B + C | untested |
| K3 KV-channel ablation (target replication) | #1801 | ablation drop ∈ [8, 15]pp (paper 15pp at 8B; expect weaker at 4B) | not measured — Phase C/D blocker | untested |
| K4 throughput (target serving) | #1802 | 1.3-1.5× throughput on M5 Pro | not measured — Phase C blocker | untested |

## Assumptions locked in pre-registration

- Base model: `mlx-community/gemma-4-e4b-it-4bit` (PLAN.md Part 2 dev target).
- SFT approach: full-parameter, not LoRA — preserves paper's faithfulness and keeps K2 measuring the intended quantity. If IMPL hits OOM, fix order is `mx.checkpoint` → grad-accumulation → fewer SFT steps → pivot to 26B-A4B as new experiment. Silent swap to LoRA or shorter SEQLEN is **forbidden** per antipattern-t.
- Dataset: `microsoft/OpenMementos` (228K traces, MIT, SFT-formatted).
- Target-gated pairing: K1 proxy paired with K2 target per F#666; K3 is target replication (not paired); K4 is target serving (not paired). KILL requires proxy-FAIL AND target-FAIL together (K1 FAIL ∧ K2 FAIL).

## Blockers honestly reported (5)

1. **B1 — tokenizer extension**: embed/lm_head resize + mean-init of new rows. Non-standard in `mlx_lm.load()`; needs manual `nn.Embedding` reassignment.
2. **B2 — 2-stage SFT**: stage 1 full-parameter + stage 2 mask-surgery forward path. Not in `mlx_lm.lora` CLI.
3. **B3 — block-mask inference**: custom MLX generation loop with per-token `BlockMaskState` + `mx.fast.scaled_dot_product_attention(mask=...)` + selective KV eviction.
4. **B4 — KC eval instrumentation**: peak-KV-memory probe (MLX does not expose per-layer KV tensors by default); K3 mask-strategy swap mid-eval.
5. **B5 — runtime budget**: 6-10h full pipeline > 30-min researcher cap.

## What this filing is NOT

- It is **not** a kill. No KC has been measured against data; there is no evidence to kill on. KILLING on proxy-FAIL without paired target-measurement is the exact F#666 / guardrail-1007 violation.
- It is **not** a supported replication. The code does not execute the mechanism.
- It is **not** a silent scope-swap. `run_experiment.py` does not quietly substitute LoRA for SFT or a shorter SEQLEN; it honestly raises `NotImplementedError` inside each phase function and routes to a graceful-failure results writer. This is the exact pattern validated by F#682/F#683/F#684.

## What IMPL must do to land a real verdict

1. Invoke `/mlx-dev` and `/fast-mlx` before writing code (guardrail 1012).
2. Implement Phase A (tokenizer extend + embed/lm_head resize).
3. Implement Phase B stage 1 (standard next-token CE on OpenMementos).
4. Implement Phase C (block-mask attention with dynamic KV eviction in the MLX generation loop).
5. Implement Phase B stage 2 (attend-only-to-mementos SFT, depends on C's mask-producing forward).
6. Implement Phase D (K1-K4 eval harness with KV-memory instrumentation + K3 ablation arm).
7. Run with n≥200 on GSM8K-Hard for K1/K2; MMLU subset for K2.
8. Route the verdict: SUPPORTED if K1 PASS ∧ K2 PASS ∧ K3 PASS ∧ K4 PASS; KILLED on the gate KCs per MATH.md §3 failure-gate (pivot to 26B-A4B if K2 or K3 fails at 4B).

## Upstream effects if IMPL succeeds / kills

- **IMPL SUPPORTED** → unblocks `exp_memento_cross_session_persistence` (P3) and `exp_user_adapter_from_memento_distillation` (P2).
- **IMPL KILLED on K2** → "MEMENTO requires ≥8B active at Gemma 4 scale" — publishable finding; pivot to `exp_memento_gemma4_26b_replication` at 26B-A4B base.
- **IMPL KILLED on K3 only** → proxy-KV-channel claim is scale-dependent at our 4B; finding about the mechanism rather than kill of the replication.

## Design decisions logged

- **No LoRA-substitution** — paper is full-parameter SFT; LoRA would change K2's meaning. Reviewer antipattern-t applies.
- **Path B (faithful block-mask) over Path A (InftyThink-style discard)** — Path A would only test K3's ablation arm, not the full mechanism. We need Path B as the `M_memento` arm and Path A as the K3 ablation control.
- **Keep_last_n_blocks = 1** — matches paper's "fresh canvas" default; IMPL may sweep but defaults to 1 for primary K1/K2.
