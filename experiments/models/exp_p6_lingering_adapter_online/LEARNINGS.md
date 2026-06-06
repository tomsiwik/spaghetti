# LEARNINGS: exp_p6_lingering_adapter_online

## Audit Rerun Status (2026-04-18)
**KILLED** under strict PLAN.md §1 verdict-consistency pre-flight. Tags:
`audit-2026-04-17-rerun`, `metric-swap`. Reconstruction from 2026-04-11 PAPER.md
measurements — code not re-executed because the antipattern is structural (KC
text "MMLU" vs code measuring 20 hand-curated trivia questions), not a
transient bug. Re-running unchanged code would reproduce the same swap.

## Core Finding (after audit)
Two of three pre-registered KC pass cleanly (K1285 project QA +60pp, K1286
latency 110ms), but K1287 pre-registered "MMLU within 2pp" was operationalised
as keyword-match on 20 trivia questions — not MMLU. Under strict pre-reg rules
the experiment is KILLED; the substantive behavioral story (online rank-4 LoRA
encodes project facts from 20 turns at <200ms/turn) is preserved as a substantive
finding but not credited as a full closure of K1287.

## Pre-rerun claim (kept for context; do not cite as supported)
Rank-4 LoRA with single-gradient-step online updates can encode project-specific
knowledge from 20 conversation turns: +60pp project QA accuracy (0→60%), 110ms/turn
latency. "Zero general knowledge cost" is **not** established — the measurement
vehicle is a 20-question trivia proxy with binomial σ ≈ 11pp, not MMLU.

## Why It Works
Intrinsic dimensionality of a single project's facts (~10 facts) is far below LoRA
capacity (rank-4 × 8 layers = 32 dimensions), so convergence is guaranteed even with
non-convex LoRA loss in practice (arXiv:2012.13255). Online GD regret bound (Zinkevich
2003) explains frequency-accuracy correlation: facts seen 3-4× all learned; 2× mixed.

## What Breaks
- **Repetitive generation**: adapted outputs show pathological repetition (token cycling)
  — rank-4 constraint forces the model to loop rather than search for alternatives.
- **Character-level details lost**: separators ("zf:" → "zf_"), units ("256KB" → "256
  bytes"), version suffixes ("Python 3.12" → "Python"). Adapter encodes semantic
  category but not precise token identity.
- **Non-convexity caveat**: formal regret guarantee requires convex loss; the LoRA
  → logits mapping is non-convex. Empirical convergence holds but formal bound is loose.

## Implications for Next Experiment
First: close K1287 cleanly in a v2 experiment. Two ways:
  (a) use `lm-eval-harness` MMLU (or a pre-declared subset) — pre-register at
      MATH.md time; never swap after data comes in.
  (b) explicitly pre-register "trivia-proxy keyword matching on N≥200 general
      knowledge questions" as the metric — not MMLU. Cannot be done via this
      experiment retroactively (would be KC-swap antipattern).

Downstream behavioral follow-ups (predicated on a clean K1287 rerun):
- Repetitive generation: add repetition penalty or longer response windows.
- Character-level recall: rank=8 or k=3–5 gradient steps per turn.
- O(1/√T) scaling predicts ~80% at 40 turns — directly testable.

## Lesson Codified
**Metric-swap antipattern**: if KC text names a specific benchmark (e.g. MMLU,
MedQA, HumanEval), the code must measure THAT benchmark — not a hand-curated
proxy. The author intended "general knowledge" and wrote "MMLU"; the code chose
the lightest proxy it could run fast. This disjunction is the metric-swap
pattern now documented for exp_p1_t2 (MedQA vs MedMCQA) and multiple T2.1
downstream kills. Fix: at MATH.md time, either (a) write the actual benchmark
integration, or (b) name the proxy explicitly in the KC.

## References
- arXiv:2411.13405 — PLUM: conversation-to-QA augmentation
- arXiv:2012.13255 — Intrinsic dimensionality (Aghajanyan et al.)
- Zinkevich 2003 — Online convex optimization regret bounds
- Finding #576/#577/#578/#579/#581/#584 — related metric-swap / precondition-probe kills
- Original Finding #490 (retired) — pre-audit SUPPORTED status no longer valid
