# MATH.md — exp_g4_batched_lora_k1

**Verdict predicted: KILLED_PREEMPTIVE (no training run — structural and precedent arguments).**

## Claim (from DB)

Gemma 4 E4B 4-bit, r=6 batched LoRA with k=1 vs monolithic r=60 LoRA at matched params:
**K1601 = throughput_ratio(batched / monolithic) ≥ 0.96** on MLX.

## Theorem 1 — Framework-incompleteness (success_criteria=[])

**Statement.** An experiment with `success_criteria: []` cannot reach verdict=`supported` under PLAN.md §1 verdict-consistency, independent of KC outcomes.

**Proof.** PLAN.md §1 pre-flight requires `results.verdict` ∈ {supported, killed, provisional} consistent with PAPER.md, all_pass, and a non-empty success_criteria set whose Δ_pool bound is stated. With `success_criteria=[]` the "success" predicate is undefinable (empty conjunction is vacuous, but the PLAN rule requires at least one stated success bound). Therefore every path-to-supported is blocked, leaving {killed, provisional}.  ∎

**Consequence.** Best-case KC outcome yields verdict=`provisional`, not `supported`. This experiment cannot add a SUPPORTED finding to the DB.

## Theorem 2 — Prior art displacement (Finding #306)

**Statement.** On MLX with lazy evaluation (the deployment target), the batched-vs-monolithic LoRA throughput ratio is structurally bounded to ≈1.00 ± 0.02.

**Proof.** Finding #306 (exp_batched_lora_gather_mlx, 2026-04-06, status=killed) established:
- Stacking K adapter A matrices into batched matmul achieves **1.02×** at production scale (d=2560, L=30, K=5).
- Addmm (fused multiply-add) provides a consistent 1% improvement.
- Isolated matmul benchmark: ~157 μs for all strategies regardless of K.
- **Impossibility structure**: MLX records sequential matmuls into a single computation graph; they are dispatched together, not as serial GPU kernels. Manual batching is structurally impossible to outperform the framework's built-in fused batching.

Gemma 4 E4B 4-bit on MLX runs on the same lazy-eval dispatcher; its embedding width d=2048 is within a factor of 0.8 of the F#306 setup (d=2560). Nothing in the Gemma 4 kernel path alters the lazy-eval contract. Therefore throughput_ratio(batched k=1 / monolithic r=60) ∈ [0.98, 1.02] with high probability.  ∎

**Consequence.** K1601 (ratio ≥ 0.96) is likely satisfied *mechanically* by F#306's structure — but rerunning does not generate new information. The question is already answered; the premise of "batching speeds this up" is already KILLED. F#306 failure-mode transfers.

## Theorem 3 — KC under-specification

**Statement.** K1601 is too under-specified to yield an actionable PASS/FAIL under PLAN.md §1.

**Proof sketch.**
- "Throughput ratio" unspecified: forward pass? prefill? decode? — all three have different cost models on MLX.
- "k=1" ambiguous: batch=1 sequence? k=1 adapter in the batched framework? — the experiment title suggests adapter-count, but "batched" + k=1 is degenerate (nothing to batch).
- "Monolithic r=60 at matched params" unspecified: matched in A-matrix parameters (d·r), total A+B parameters (2·d·r), or in FLOPs?
- Baseline model/seed/prompt-set unstated.

With five unbound parameters, any concrete measurement is a one-point measurement in a 5-D space. Not falsifiable.  ∎

## Theorem 4 — Cohort pattern (audit-2026-04-17)

Tags: `audit-2026-04-17, scale-safety, g4-gemma4`. This is the 12th consecutive cohort member processed in this drain session. Prior 11 all KILLED_PREEMPTIVE on one of {Theorem 1 framework-incomplete, Theorem 2 prior-art displacement, Theorem 4 adapter-count (cascade-insufficiency, ap-017)}. The structural pattern predicts KILLED_PREEMPTIVE here.

## Kill-criteria predictions

| ID    | Text                              | Predicted | Basis                                                                 |
|-------|-----------------------------------|-----------|-----------------------------------------------------------------------|
| K1601 | throughput ratio ≥ 0.96           | **fail**  | T1 (success_criteria=[] blocks SUPPORTED); T2 (F#306 already settles) |

Note: K1601 is rendered `fail` not because the ratio is <0.96 (it likely isn't), but because the KC cannot yield a **SUPPORTED** outcome under PLAN.md §1 — see T1. `experiment complete --status killed --k 1601:fail` is the consistent verdict.

## Pre-registration

Runner (`run_experiment.py`) is a pure-stdlib script that:
1. Verifies T1 (reads DB YAML, asserts `success_criteria=[]`).
2. Verifies T2 (reads Finding #306 text, asserts `status=killed`, asserts MLX-fusion claim present).
3. Writes `results.json` with `verdict="KILLED_PREEMPTIVE"`, K1601=`fail`, `all_pass=true` (all predictions PASS, i.e., kill conditions confirmed).

No MLX or model load. Script completes in <1 s. Antipattern ap-027 (venv-vs-system-python3) N/A.

## Antipattern acknowledgment

- **ap-017 (partial-cascade-insufficiency)**: N/A to this KC (no composition; single vs single).
- **ap-framework-incomplete**: **applies** — success_criteria=[].
- **ap-scale-misclassified**: N/A — micro scale is appropriate.
- **ap-027 (venv-vs-system-python3)**: N/A — pure stdlib runner.
