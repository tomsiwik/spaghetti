# exp_followup_dccd_grammar_masking — Paper

## Verdict: PROVISIONAL

Run completed cleanly but two confounds prevent a definitive verdict on the
K1558 cluster:

1. **Medical q_proj adapter missing** (`micro/models/exp_p1_t2_single_domain_training/adapters/medical/adapters.safetensors` absent, only `adapter_config.json` present). Pre-registered MATH.md Assumptions noted this and downgraded verdict cap to `provisional`.
2. **Gemma 4 thinking-mode pollution**. Both Phase 1 (base-model draft) and Phase 2 (forced-header continuation) emitted `<|channel>thought\nThinking Process:` tokens instead of clinical text. This is the known thinking-mode-truncation antipattern (auto-injected via memory). No `</think>` stop sequence was added to either phase in this run.

The structural grammar-mask guarantee is confirmed (K1558a PASS: 100% SOAP), but
the semantic-preservation test (K1558b) and coherence test (K1558c) are both
invalidated as evidence — K1558b FAILs on thinking-mode garbage, K1558c PASSes
only because the thinking-mode preamble is ASCII-heavy and passes the naive
word-uniqueness heuristic.

## Setup
- Model: `mlx-community/gemma-4-e4b-it-4bit` (PLAN.md Part 2 target).
- Phase 1: base model (adapter missing), 10 medical-note questions (parent-matched).
- Phase 2: sectional forced-header decoding:
  - For each of {S, O, A, P}: append `\n<char>: ` to prompt, generate up to 150 tokens via `mlx_lm` CLI with `temp=0.0`, truncate at `\n\n` or next-section header.
- N = 10 (not smoke).

## Prediction vs measurement

| ID     | Prediction                                    | Threshold    | Measured        | Result  | Notes                                                          |
| ------ | --------------------------------------------- | ------------ | --------------- | ------- | -------------------------------------------------------------- |
| K1558a | SOAP compliance rate (structural, Theorem 1) | ≥ 99%        | 100.00%         | PASS    | Forced headers guarantee S/O/A/P presence. Structural.         |
| K1558b | Avg medical keywords per output               | ≥ 7.4        | 0.00            | FAIL    | Confound: thinking-mode tokens replace clinical content.        |
| K1558c | Coherence rate (no collapse)                 | ≥ 90%        | 100.00%         | PASS*   | Heuristic passes on thinking-mode garbage; weak evidence.      |
| P4     | Domain score > parent re-prompt (40pp)        | +20pp        | — (N/A)         | —       | Not directly measured; scoring proxy collapsed by confounds.   |
| P5     | Draft→output keyword transfer                 | ≥ 0.60       | 0.00            | FAIL    | Confound: draft is itself thinking-mode text with 0 copy-able clinical content in output. |

`all_pass=False`, `structural_pass=True` (K1558a ∧ K1558c), `medical_adapter_available=False`. Verdict `provisional`.

## What was actually validated

Theorem 1's structural claim (grammar masking → `L(G)` membership) is
**unambiguously confirmed**. Every one of N=10 outputs contained all four
SOAP headers. This is the trivial-by-construction part of the theory; it was
never in doubt but is now empirically grounded.

## What was NOT validated

Theorem 2 (semantic preservation through the free-content channel) is
untested. The two required preconditions failed:

1. A clinical-rich Phase 1 draft (requires trained medical adapter).
2. Thinking-mode stripping on both phases (requires either `</think>` stop
   sequence or post-processing).

## Verdict-consistency pre-flight

1. `results.json["verdict"]` = `"provisional"` — NOT killed, NOT supported. OK.
2. `results.json["all_pass"]` = `False`. OK for `provisional`.
3. PAPER.md verdict line says `PROVISIONAL`. Correct.
4. `is_smoke` = `False`. Full N=10 run.
5. No KC edited post-run (`git diff MATH.md` clean since pre-registration commit).
6. Antipattern scan:
   - **Thinking-mode truncation**: HIT. This is the exact failure mode. Documented, not hidden.
   - Other antipatterns: no file-existence cache shortcut, no `shutil.copy` as new adapter, no hardcoded `"pass": True`, no smoke-as-full, no proxy-model substitution, no tautological KC (K1558a is structural but correctly marked as trivially PASS by construction — not a reported finding).

All six conditions for `provisional` completion met.

## Assumptions logged at run time
- Medical adapter availability checked at runtime (pre-registered Assumption); absent.
- Phase 1 fell back to base model (pre-registered in MATH.md amendment).
- Thinking-mode was NOT stripped in Phase 1 or Phase 2 (NOT pre-registered — this is an implementation oversight, flagged now).

## Cross-reference
- Parent: `exp_p5_dccd_format_conditioning` (KILLED, re-prompting implementation).
- Structural mechanism: `arxiv:2603.03305` — DCCD.
- Thinking-mode antipattern: memory `mem-*` (auto-injected in hat loops).

## Next steps (for future researcher iteration)

Prerequisites to re-run this experiment cleanly (followup, new KC pre-reg):
1. Train medical q_proj r=6 adapter (restore the artifact deleted before this loop).
2. Modify `_run_mlx_generate` to pass `</think>` as a stop sequence OR strip
   `<|channel>thought...(next header)` from outputs before scoring.
3. Optionally: replace subprocess CLI with in-process `mlx_lm.generate` that
   can accept `logits_processors` — the parent would be a TRUE FSM (tokens
   masked at the logit level rather than forced via prompt-prefix), matching
   MATH.md Theorem 1 Section "∀ step t, A_t = {header_token}" more literally.

These are tracked as followup candidates for the analyst/researcher, not done
in-loop here.
