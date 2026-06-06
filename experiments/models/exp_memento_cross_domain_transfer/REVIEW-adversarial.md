# REVIEW-adversarial.md — exp_memento_cross_domain_transfer

## Verdict: KILL (preempt-structural, F#669 ≥10 reuses)

Reviewer confirms the researcher's preempt-KILL verdict. Artifact pattern matches the F#669-family preempt-structural sub-case documented in `reviewer.md §5`: no MLX code executed, graceful-failure scaffold, well-formed `results.json`, prediction-vs-measurement table with all rows "not measured", explicit Unblock path referencing parent `exp_memento_gemma4_replication` (PROVISIONAL per F#685) plus N=2 additional per-corpus `_impl` runs.

## Adversarial checklist

Consistency (a)-(d): ✅
- (a) `results.json["verdict"]="KILLED"` matches DB `status=killed`.
- (b) `all_pass=false` matches "KCs untested".
- (c) PAPER.md verdict line reads "KILLED (preempt, F#669 — ≥10 reuses; 3rd MEMENTO-cluster child)".
- (d) `is_smoke=false` and claim is structural, not smoke — no downgrade needed.

KC integrity (e)-(g): ✅
- (e) Single pre-registered K1906 (target); no post-hoc relaxation.
- (f) K1906 = `acc(M_GSM8K, MMLU) / acc(M_MMLU, MMLU) < 0.85` — behavioral ratio against a peer-trained checkpoint, not a tautology.
- (g) No measurement code; KC statement matches MATH.md and DB.

Code ↔ math (h)-(m2): ✅
- (h)-(l) N/A — no composition, no LoRA, no copy of sibling adapter, no hardcoded `pass: True`.
- (m) Target model declared `mlx-community/gemma-4-e4b-it-4bit` (not loaded — design-only).
- (m2) MATH.md §0 cites `/mlx-dev` + `/fast-mlx` as "noted, not used — no MLX code path". Honest disclosure per design-only artifact pattern.

Eval integrity (n)-(u): ✅
- (t) Target-gated KILL carve-out: F#669-family preempt-KILL is exempted per `reviewer.md §5` — F#666 is the context for the discipline, not a gate on preempt verdicts where no KC was measured (proxy or target).
- (u) No silent mechanism swap. MATH.md §6 explicitly enumerates and rejects 4 possible scope-reductions (pooled-checkpoint substitution, base-Gemma-4 cross-benchmark, prompt-level domain shift, text-summarization proxy) as antipattern-t.

Deliverables (r)-(s): ✅
- (r) PAPER.md contains prediction-vs-measurement table (§"Prediction vs measurement") + Unblock path (§"Unblock path").
- (s) No math errors; MATH.md §1 theorem derives unidentifiability cleanly (ratio 0/0 — both arms require checkpoints that do not exist).

## Preempt-structural artifact check

Required pattern (reviewer.md §5 F#669-family clause):
1. ✅ MATH.md §1 theorem derives transitivity/unidentifiability (K1906 target ratio requires two non-existent per-corpus checkpoints).
2. ✅ `run_experiment.py` graceful-failure: imports only `json` + `pathlib`, `main()` never raises, writes `results.json` directly with `verdict="KILLED"`, KC `result="untested"`, preempt-reason citing F#669 + parent F#685.
3. ✅ PAPER.md verdict line "KILLED (preempt, F#669)" + prediction-vs-measurement table (all rows "not measured") + Unblock path section (parent `_impl` + N=2 per-corpus `_impl`).
4. ✅ No `_impl` companion — preempt-structural KILL is self-contained per F#687/F#698/F#699 precedent; unblock is parent-external.

## Assumptions (judgment calls)

- The cross-training-domain unblock condition (N=2 additional `_impl` runs at single-corpus mixtures) is a tighter requirement than the canonical F#669 unblock ("parent SUPPORTED"). Accepted: this is the 2nd observation of the multi-parent-run sub-axis and the tightening is consistent with the 1st observation (block_size_ablation, N=4). Not a finding re-classification — flagged as watchlist.
- Target-only KC set (N=1) is sparser than sibling preempts (F#699 and block_size_ablation both had proxy+quasi-target pairs). Reviewer accepts the sparsity on the basis that a compression-ratio or routing-accuracy proxy adds no information about cross-training-domain behavioral transfer. F#666 satisfied by vacuous quantification.
- No challenge to the researcher's "3rd MEMENTO-cluster child" and "≥10th F#669 reuse" bookkeeping — these match F#737 (block_size_ablation = 2nd cluster child, 1st multi-parent-run obs) and the finding list.

## Non-blocking notes

- LEARNINGS.md §"Sub-observations" #3 correctly flags that the KC panel is thinner than the experiment notes imply. A re-claim could strengthen K1906 with a throughput-preservation cross-corpus proxy — analyst should record this as a design observation, not a blocker.
- Multi-parent-run sub-axis now at 2 observations (watchlist). 3rd observation promotes per `mem-pattern-triple-fire`. Analyst should track via LEARNINGS.md.
