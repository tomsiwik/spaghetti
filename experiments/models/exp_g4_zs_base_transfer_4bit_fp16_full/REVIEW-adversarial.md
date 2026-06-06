# REVIEW-adversarial — exp_g4_zs_base_transfer_4bit_fp16_full

## Verdict

**PROVISIONAL** (`supported_target_only` per F#666; finding-about-the-proxy disposition).

K1814 (proxy / PPL-gain ratio, inherited from parent F#680): FAIL — median R_ppl = 0.946 < 0.95.
K1815 (target / task-accuracy ratio, NEW): PASS — median R_task = 1.139 ≥ 0.95; min R_task = 1.029 ≥ 0.85.

Per F#666 truth table: proxy-FAIL + target-PASS = finding about the proxy, not a kill on the
mechanism. PROVISIONAL is the correct schema-bridge (CLI does not accept `supported_target_only`
on `complete`; provisional preserves both halves of the F#666 discriminator).

## Adversarial checklist

Consistency:
- (a) results.json verdict=PROVISIONAL, verdict_internal=supported_target_only, all_pass=false ✓ internally consistent.
- (b) all_pass=false alongside non-killed verdict — correct under F#666 (proxy-FAIL is a finding, not a kill).
- (c) PAPER.md verdict line "PROVISIONAL (supported_target_only per F#666)" matches results.json ✓.
- (d) is_smoke=false; n_eval_per_domain=50 (matches parent T2.1 protocol) ✓.

KC integrity:
- (e) MATH.md is unchanged from claim-time (`git diff --stat HEAD` empty) — KC IDs locked at registration ✓.
- (f) No tautology: K1814 inherits a real measured ratio from parent on disk; K1815 is a paired
  evaluation of the same adapter on two distinct precision realizations of the base. Single-adapter
  eval, no composition.
- (g) K-IDs in code measure the quantities described in MATH.md and DB row ✓.

Code ↔ math:
- (h) No `sum(lora_A)` / `add_weighted_adapter` / safetensor key summation — single-adapter eval ✓.
- (i) LORA_SCALE = 6 inherited from `exp_p1_t2_single_domain_training` adapters; below the 12-threshold ✓.
- (j) Routing: N/A.
- (k) No `shutil.copy` re-labeling ✓.
- (l) K1815 `pass` derived from `(median_r_task >= 0.95) and (min_r_task >= 0.85)` boolean (run_experiment.py L269); not hardcoded ✓.
- (m) Target model: Gemma 4 E4B (MLX 4-bit + 8-bit). Code uses `MODEL_4BIT` and `MODEL_8BIT` (run_experiment.py L242, L248). MATH.md §Theorem matches ✓.
- (m2) Skill invocation: MATH.md §"Skills invoked" cites `/mlx-dev` and `/fast-mlx` per guardrail 1012;
  code uses `mlx.core` memory-safety pattern (`mx.set_memory_limit`, `mx.set_cache_limit`,
  `mx.clear_cache` between phase 1 and phase 2) — phased execution memory hygiene confirmed by
  pueue task log: `[MEM phase1-end] active=0.00GB cache=0.00GB` after Phase 1 cleanup ✓.

Eval integrity:
- (n) Base accuracies on 4-bit: HumanEval 70%, GSM8K 72%, MedQA 68% — non-zero, non-truncated ✓.
- (o) n=50 per domain (≥ 15) ✓.
- (p) No synthetic padding; HumanEval/GSM8K/MedQA-USMLE-4-options are real benchmarks ✓.
- (q) No baseline drift: K1814 baseline IS the parent's measurement, inherited verbatim ✓.
- (t) Target-gated kill (F#666): NOT being applied as a kill. Proxy-FAIL + target-PASS is the
  canonical "finding about the proxy" disposition; `complete --status killed` is unjustified.
- (u) No scope-changing fixes. Original SFT/LoRA/n=50/protocol preserved ✓.

Deliverables:
- (r) PAPER.md prediction-vs-measurement table present for both K1814 and K1815 ✓.
- (s) No math errors detected. The `r_task = a8 / a4` definition (run_experiment.py L264) matches
  MATH.md §Quantities exactly. Median is computed via `sorted(r_task.values())[1]` (L267) which
  is correct for a 3-element list. Per F#666 truth table application at L273–280 matches MATH §"Per F#666 truth table" exactly.

## Headline finding (worth registering)

**The 4→8-bit PPL-gain ratio is not a reliable transfer-fidelity proxy for adapter benefit on
Gemma 4 E4B.** Behavioral task-accuracy ratio is functionally lossless and in fact strictly
*higher* in every domain measured (HumanEval 70→80, GSM8K 72→82, MedQA 68→70 ≡ R_task = 1.143,
1.139, 1.029). The proxy understated transfer fidelity by a wide margin. This empirically falsifies
the proxy direction implied by parent F#680 ("marginally failing at 5% PPL threshold") and confirms
the project guidance `r ≈ 0.08(PPL, task)` (guardrail 1006).

## Assumptions (judgment calls logged)

1. K1814 is inherited byte-for-byte from parent's `results.json`; not re-measured. The parent's
   median_transfer_ratio = 0.945859805369008 and min_transfer_ratio = 0.8998956945668609 were
   verified on disk to match the values surfaced in this experiment's results.json.
2. The PROVISIONAL disposition under `supported_target_only` is treated as schema-bridge, not as
   a kill. This follows the F#666 truth table (proxy-FAIL + target-PASS → finding about the proxy)
   and the reviewer hat workaround for verdicts the `complete` CLI does not accept directly.
3. The (m2) skill-invocation evidence is satisfied by MATH.md §0 citation + observable `mx.clear_cache`
   pattern in code, not by hat-runtime invocation. Phased-execution memory hygiene
   (`[MEM phase1-end] active=0.00GB cache=0.00GB`) at pueue log confirms the mlx-core pattern works.

## Disposition

PROCEED → analyst (LEARNINGS.md). The PROVISIONAL workflow is being executed:
1. `experiment update --status provisional` (done — DB row shows status=provisional).
2. `experiment evidence` claim recorded.
3. `experiment finding-add --status provisional` (registers headline finding).
4. Verify finding landed via `experiment finding-list`.

No follow-up `_full_full` is filed — this experiment IS the `_full` follow-up of the parent F#680.
The bf16 / retrain-per-precision / composition-under-precision-change follow-ups suggested in
PAPER.md §"Suggested follow-ups" are content-level next steps, not workflow-required.
