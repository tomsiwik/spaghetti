# LEARNINGS — `exp_hedgehog_behavior_adapter_politeness`

**Status:** PROVISIONAL (design locked, implementation deferred).

## What this iteration produced

- MATH.md §0 (platform skills + version pin — closes the JEPA sibling's non-blocking reviewer flag preemptively) + unchanged theorem/proof/KCs.
- Rewritten `run_experiment.py` in the JEPA-pattern graceful-failure mode:
  - Phase 0 (neutral-prompt curation), Phase B (Hedgehog training loop), Phase E (K4 ablation) all `NotImplementedError` with structured blocker messages.
  - Phase C/D skipped cleanly when adapter not produced.
  - `main()` never raises; always writes `results.json` with `PROVISIONAL` verdict and per-KC `"untested"`.
  - Records `mlx_lm_version` at runtime.
- `results.json` with `"PROVISIONAL"` verdict, 5 KCs `"untested"`, 5 blockers enumerated.
- PAPER.md with full verdict-consistency pre-flight (all 6 checks clear for PROVISIONAL).
- REVIEW-adversarial.md researcher self-pass (reviewer hat will overwrite with independent pass).

## What was *not* produced (honest reporting)

- Phase B Hedgehog cos-sim training loop (teacher+student forward passes with per-layer attention-output hooks, per-layer cos-sim loss, AdamW). Requires a custom MLX training loop that mlx-lm's `lora` CLI does not support. Estimated 2-4h careful MLX engineering.
- Phase 0 neutral-prompt curation (UltraChat filtered to drop politeness-markers). Load-bearing for K2 judge scoring — unfiltered prompts collapse the Δ metric.
- Phase E ablation retrain (same blocker as Phase B).
- MMLU/HumanEval evaluation (Phase D — depends on trained adapter).
- Claude/GPT-4 paired-compare judge scoring for K2/K4 (depends on adapter + neutral prompts).

Filing Phase A alone would be a partial-scope antipattern (no adapter means no K1–K5 measurement). PROVISIONAL across the whole design is the honest status.

## Takeaways for the next researcher iteration

1. **Novel-mechanism single-iteration scope antipattern fired again.** This is now the 2nd instance (JEPA was 1st) of a novel-mechanism design PROVISIONAL. The `mem-antipattern-novel-mechanism-single-iteration-scope` memory directly applies to hedgehog_*, memento_*, rdt_*, and any distillation-with-aux-loss mechanism in the P≤2 backlog. The correct pattern when the claim picker hands you one:
   - lock MATH.md (incl. §0 skill invocation + version pin),
   - scaffold with NotImplementedError,
   - mark PROVISIONAL + add inconclusive evidence,
   - file `_impl` follow-up at P3 (keeps P≤2 bucket unchanged while preserving the design).

2. **Claim-picker is tag-blind.** Event handoff said "prefer standard-mechanism candidates (g4_adapter_class_composition_full, memento_*, p1_* LoRA sweeps)". Claim picker returned hedgehog (novel-mechanism) on the very next pick. Release → reclaim loops onto the same head because the picker lacks a tag-exclude axis. This is analogous to the "claim-time cohort saturation" antipattern raised previously but on a different saturation axis (`tags` instead of `cohort`). Candidate `type: fix` antipattern: **"claim-time tag saturation"**.

3. **F#666 pairing is cheap when written into the scaffold.** K#1782 (proxy) and K#1783 (target) are locked as a pair at KC-registration time; K#1784 (non-interference) and K#1785 (ablation) are additional target-side gates. Even if the eventual full run kills K#1782, the target-side gates prevent a proxy-alone kill. Writing this discipline into the scaffold (not retrofitting during review) is the pattern that should propagate to every new experiment.

4. **MATH.md §0 preempts reviewer flag (m2).** The JEPA sibling left §0 missing, triggering a reviewer non-blocking flag + making it blocking for the `_impl` follow-up. Adding §0 (platform skills + mlx-lm version pin + adapter-target citation + scope-preservation note) during the PROVISIONAL iteration closes the gap preemptively. Same discipline should propagate to all novel-mechanism PROVISIONAL scaffolds.

## Not learned / open

- Whether per-layer attention-output cos-sim actually captures the polite-teacher behavior on Gemma 4 E4B (the central claim) — requires Phase B completion in `_impl`.
- Whether `v_proj + o_proj` rank-8 is sufficient capacity (F#627 validated rank-6 for domain adapters; behavior adapters may need different rank — unclear a priori).
- Whether UltraChat-filtered neutrality is the right corpus (or a curated synthetic neutral set would be tighter).
- Whether Claude 3.7 and GPT-4 agree on politeness rubric — `_impl` should cross-validate judges.

## References

- Moudgil/Apple+MILA 2026 arxiv:2604.14191 §3.1 eq. 6 — Hedgehog Stage-1 recipe, per-layer cos-sim loss form.
- Zhang et al. 2024 arxiv:2402.04347 — Hedgehog feature-map, cos > 0.99 existence for attention-output matching.
- Finding #627 — `v_proj + o_proj` is the proven Gemma 4 E4B adapter target.
- Finding #666 — Target-gated kill rule.
- Finding #328 / #330 — `LORA_SCALE` safety bound (≤ 8).
- Finding #673 — `/mlx-dev` skill-invocation discipline (biggest cause of broken MLX code).
- `mem-antipattern-novel-mechanism-single-iteration-scope` — directly applies to this experiment class.
- Prior instance: `exp_jepa_adapter_residual_stream` (PROVISIONAL, same pattern).
