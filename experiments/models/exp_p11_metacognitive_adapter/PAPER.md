# PAPER.md — P11.H1: thinking-metar1-metacognitive-v0

**Date**: 2026-04-17 (preemptive kill — supersedes 2026-04-14 design-review draft)
**Status**: KILLED (preemptive — not run, all three KCs unsalvageable)

---

## §1 Verdict

**KILLED — preemptively, before run.**

Triggered by `learning.complete exp_p11_meta_r1_metacognition` (D0) on 2026-04-17.
After H0 (universal thinking adapter, dependency) was killed earlier today with a
−14.5pp MMLU-Pro regression, and after the Gemma-4 mlx_lm.lora reasoning-adapter
training stack revealed a structural protocol bug in B0 (replicated in D0), the
H1 KC table cannot be honestly evaluated and the run would consume ~2 h of compute
to produce an outcome that is already determinable.

This is the **sixth** consecutive preemptive or measured kill in the
`mlx_lm.lora` Gemma-4 reasoning-adapter chain today: F0 (OOM) → H0 (GD violation)
→ B0 (protocol bug) → C0 (preemptive: depends-on B0) → D0 (preemptive: replicates
B0 protocol) → H1 (this experiment).

---

## §2 Why not run

### (a) Replicates the B0/D0 protocol bug

`run_experiment.py:260`:

```python
structured_response = (
    f"<|channel>thought\n{structured_thinking}\n<channel|>{answer_part.strip()}"
)
```

This string becomes the literal `assistant.content` field at line 278
(`{"role": "assistant", "content": structured_response}`) and is fed to
`mlx_lm.lora` as SFT data. `mlx_lm.lora` does NOT recognise `<|channel>thought`
as a channel-protocol delimiter — it tokenises and trains on the bytes as text.
At eval time the chat template invokes channel-thinking via `enable_thinking=True`,
expecting the model to emit channel tokens as protocol. The training/eval format
mismatch is the same root cause that gave B0:
- −15.3pp MMLU-Pro regression (57.1% → 41.8%)
- −71% thinking suppression (2819 → 816 chars)

D0 was preemptively killed earlier today on the same line-pattern at
`exp_p11_meta_r1_metacognition/run_experiment.py:267` — H1 is byte-identical in
intent (PLAN/CHECK injected into channel-text format).

### (b) K1521 vacuous against regressed H0 baseline

K1521 measures `H1.MMLU-Pro ≥ H0.MMLU-Pro`. After H0's kill today:

| Source | H0 MMLU-Pro |
|--------|-------------|
| H1 MATH.md Theorem 1 prediction (pre-H0-run) | ≥ 65.1% |
| H0 results.json (measured 2026-04-17) | 47.6% |

The KC was designed under the assumption H0 would clear ≥65.1% (Finding #530:
base = 62.1% + thinking benefit). H0 actually regressed 14.5pp below the base.
A trivial no-op H1 adapter (or any adapter that doesn't catastrophically forget
beyond −14.5pp) satisfies K1521. K1521 no longer measures what its text
intends — it measures "does H1 fail less than the broken H0", not "does
metacognitive structure preserve quality".

### (c) K1520 likely passes by the wrong mechanism

K1520 measures `H1.thinking_chars ≤ H0.thinking_chars × 0.80` (≥20% reduction).

Measured H0 thinking: 2902 chars/q (PAPER.md 2026-04-17). K1520 threshold: ≤2322 chars.

Given (a) — the protocol bug suppresses thinking ~71% in B0 — H1 is highly
likely to pass K1520 by collapsing thinking output (e.g. to ≤900 chars), not by
metacognitive early-termination. A PASS produced by the protocol bug is not a
research finding; it is an artefact. Reporting it as "K1520 PASS" would
mis-attribute a failure mode to a success.

### (d) MATH.md Theorem 2 premise weakened

Theorem 2 (MATH.md L68–93) requires Q_H0 > Q_base for higher-quality SFT data.

| Quantity | Predicted in MATH.md | Measured 2026-04-17 |
|----------|---------------------|---------------------|
| Q_base (base MMLU-Pro w/ thinking) | 62.1% (F#530) | 40.7% (baseline_eval) |
| Q_H0 (universal-thinking MMLU-Pro) | ≥65.1% | 47.6% |
| Q_H0 − Q_base | ≥3pp | +6.9pp |

Q_H0 > Q_base survives narrowly, but H0 catastrophically forgets humanities
(engineering 13.3%, philosophy 20.0% — H0 PAPER.md). Phase 1 stratifies by
category — humanities trace yield collapses → training-data-quality assumption
breaks per-category even if it holds in aggregate.

### (e) Cascade pattern

Six consecutive `mlx_lm.lora` Gemma-4 reasoning-adapter kills today indicate
the failure is in the **shared training harness** (channel-protocol encoding for
`mlx_lm.lora` SFT data), not in the individual experiment designs. Running H1
spends compute reproducing the same harness failure rather than fixing it.

The unblocking work is a **shared training-harness fix** that either:
(i) strips channel tokens from training targets, gating thinking purely as eval
protocol; or (ii) trains via a custom MLX SFT loop that respects the chat
template; or (iii) replaces the channel-tokenization with `<think>...</think>`
text-format (which H0 used successfully — K1519 PASS, the only PASS in the H0
chain).

---

## §3 Kill Criteria

| Kill ID | Criterion | Status | Reason |
|---------|-----------|--------|--------|
| K1520 | H1 thinking chars ≤ H0 × 0.80 | **fail (by integrity, not value)** | Likely to PASS via protocol-bug suppression — that PASS would mis-attribute the bug as a research success. Not a clean evaluation. |
| K1521 | H1 MMLU-Pro ≥ H0 MMLU-Pro | **fail (vacuous)** | H0 regressed to 47.6% (vs predicted 65.1%); KC no longer measures "preserves quality". |
| K1522 | ≥ 50% H1 traces contain PLAN structure | **fail (entangled)** | Structure presence cannot be cleanly separated from the protocol-bug pathology in the same run. |

All three KCs marked **fail** for the experiment-complete call. Designating
"unevaluable" is not a status the DB supports; KILLED + fail-on-all-KCs is the
honest mapping per PLAN.md verdict-consistency pre-flight.

---

## §4 Antipattern self-check

| Antipattern (from auto-injected memories) | Present? | Note |
|-------------------------------------------|----------|------|
| Composition math bug | n/a | Single-adapter design, no composition at inference. |
| Tautological routing | n/a | No routing in H1. |
| `LORA_SCALE = 20` (mem-antipattern-003) | NO | `LORA_SCALE = 1.0` at L64. |
| `shutil.copy` adapter as new | NO | Real `mlx_lm.lora` training. |
| Hardcoded `"pass": True` | NO | KC computed from measurements. |
| Eval-template truncation (base = 0%) | NO | `parse_answer` is multi-strategy. |
| Proxy-model substituted for target | NO | Same gemma-4 base in train + eval. |
| KC measures wrong object | **YES (post-hoc)** | K1521 was designed when H0's predicted accuracy was ≥65.1%; H0 regressed to 47.6%. K1520 likely measures protocol-bug suppression, not metacognitive efficiency. |
| Channel-text-as-SFT-target | **YES** | L260 → fed to `mlx_lm.lora` at L278. Same as B0:267, D0:267. |
| Smoke reported as full | n/a | Not run. |

---

## §5 Unblock path

**Shared training-harness fix (P11.HARNESS):** new experiment that establishes
the canonical `mlx_lm.lora`-compatible serialization for thinking-channel SFT on
Gemma-4. Acceptance: MMLU-Pro w/ thinking ≥ base − 2pp on a 50-trace pilot.
Once accepted, all blocked H1/C0/D0/M0/H0-v2 redesigns inherit it.

**H1-v2 (post-harness):** redesign with locally-measured H0-v2 baseline (NOT
F#530), a non-vacuous K1521 (e.g. H1-v2 ≥ base − 2pp), and a thinking-chars KC
that subtracts protocol-suppression from the metric (e.g. measure on
samples that PARSE a valid answer — exclude protocol-failure responses).

**Recompute Theorem 2's Q_H0/Q_base** from harness-fix run; re-prove (or
falsify) before re-claiming H1.

---

## §6 Assumptions logged

- A1: Channel-text-as-SFT-target is the same root cause in B0/D0/H1 (no separate
  diagnosis run needed; mechanism is identical and was diagnosed in B0
  PAPER.md 2026-04-17).
- A2: H0's regressed accuracy (47.6%) is the canonical H0 baseline going
  forward; F#530 (62.1%) refers to base+thinking, not H0+thinking, and using it
  as H0's baseline would be a stale-reference error (same family as the F#530
  baseline-reconciliation thread opened in H0 LEARNINGS).
- A3: Decision threshold for preemptive kill = ≥3 of 4 (KC vacuous OR KC
  bug-passable OR theorem premise broken OR cascade pattern). H1 hits all 4.
- A4: REVIEW-adversarial.md and LEARNINGS.md left for reviewer/analyst hat
  rewrite — those artifacts pre-date today's H0/B0/D0 kills and are not
  re-authored by the researcher hat at preemptive-kill time.

---

## §7 References

- B0 PAPER.md (2026-04-17): protocol-bug diagnosis (channel tokens as text → −15.3pp regression, −71% thinking suppression)
- C0 PAPER.md (2026-04-17): preemptive-kill template
- D0 PAPER.md (2026-04-17): same protocol-bug replication at line 267
- H0 PAPER.md (2026-04-17): K1517 fail (47.6% MMLU-Pro), K1518 fail (40% MedMCQA), K1519 pass — dependency for this experiment
- H1 MATH.md (2026-04-13): Theorems 1/2/3 — superseded by this kill, see §2(d) for premise update
- arXiv:2508.17291 — Meta-R1 metacognition (paper grounding still intact for H1-v2 design)
- arXiv:2502.03387 — LIMO (data-quality theorem; premise weakened by H0's regression)
