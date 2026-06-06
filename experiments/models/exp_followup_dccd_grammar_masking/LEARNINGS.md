# LEARNINGS — exp_followup_dccd_grammar_masking

## Core Finding
**KILL validated.** Theorem 1 (structural SOAP via forced-header mask → `L(G)`
membership) trivially confirmed at 100% (N=10). Theorem 2 (semantic
preservation through the free-content channel) **untested** — two pre-declared
confounds.

## Why
1. **Medical q_proj r=6 adapter missing** (only `adapter_config.json`); Phase 1
   fell back to base model → "medical-adapter draft" Theorem 2 requires never
   existed.
2. **Thinking-mode pollution both phases.** No `</think>` strip/stop; Gemma 4
   base emits `<|channel>thought\nThinking Process:` in Phase 1 draft and
   Phase 2 sectional continuation. K1558b (keywords ≥7.4 → 0.00) invalidated;
   K1558c PASS is thinking-mode-ASCII artefact. Exact map onto
   mem-antipattern-008 (auto-injected, ignored).

Theorem 1 PASS is structural-by-construction, not a finding — forced-header
decoding guarantees `{S,O,A,P}` regardless of free-slot content.

## Implications for Next Experiment
Rerun prerequisites (BOTH must land in `run_experiment.py` before KC pre-reg):
1. **Restore medical q_proj r=6 adapter** at
   `micro/models/exp_p1_t2_single_domain_training/adapters/medical/`.
2. **Strip thinking-mode.** Either `</think>` / Gemma-4 `<channel|>` stop
   tokens to `mlx_lm`, or switch to in-process `mlx_lm.generate` +
   `logits_processor` channel mask (also yields token-level FSM matching
   Theorem 1 `A_t = {header_token}` literally).

**Vacate-condition pre-reg** (avoid tautological-KC trap): if post-strip
`avg_thinking_chars == 0` on restored-adapter base is NOT achieved, experiment
is *vacated*, not killed — confound still dominates.

K1558b threshold (≥7.4) must be re-derived from restored adapter's own-domain
distribution, not reused from `exp_p5_dccd_format_conditioning` (different
generation path).

No new antipattern: maps onto mem-antipattern-008 + missing-artifact confound.
