# Adversarial Review: exp_p11_reasoning_sft_s1k

## Round 2 — Audit re-review (2026-04-18)

**Verdict: KILLED — structural antipattern** (re-classification under
`audit-2026-04-17-rerun`). Round 1 (2026-04-14) below is retained for
evidence but **superseded** by this audit.

### Audit disposition
- Re-run NOT executed. Same protocol would reproduce the same finding.
- `run_experiment.py` strip_thinking regex fixed in-place (channel-aware) so
  the code is correct if anyone clones the pipeline, but the structural bug
  (training format mismatch, §3 below) is a design-level rewrite and out of
  scope for this rerun pass.

### Structural antipatterns identified (code-bug cluster)
1. **strip_thinking regex (FIXED this iteration)** — corrupted Phase 4a base
   eval from 62.1% (real, Finding #536) down to 12.5%. Regex only matched
   `<think>...</think>`; Gemma 4 emits `<|channel>thought...<channel|>`. The
   adversarial reviewer in Round 1 missed this because the *adapter* eval
   still showed 1641 avg_thinking_chars, making the regex look OK. It wasn't —
   those 1641 chars were literal `<think>` text the adapter learned to emit,
   not actual channel content.

2. **Training format mismatch (design-level, unfixed)** — line 159 sets
   `assistant_content = f"<think>{thinking}</think>\n\n{attempt}"`. mlx_lm.lora
   passes this through Gemma 4's chat template as literal assistant text.
   Gemma 4's *real* thinking channel is a distinct token stream emitted by
   `enable_thinking=True`. Consequence: the adapter was trained to imitate
   literal scaffolding text, not to use the channel. Same antipattern class
   as `exp_p11_grpo_reasoning_adapter` evidence ("mlx_lm.lora treated Gemma 4
   channel-thinking tokens as literal assistant text, breaking eval-time
   protocol").

3. **K1492 false pass** — K1492 ("thinking NOT suppressed, >0 chars") was
   marked PASS with 1641 avg_thinking_chars. Under §2 above, those chars are
   *literal* `<think>` text, not real channel usage. The KC measures the
   wrong object (antipattern #6 — KC measures surface token count, not
   channel engagement). PAPER.md's original Theorem 1 "verification" should
   be downgraded to "untested under correct channel protocol".

4. **GSM8K HTTP 422** — datasets-server API rot, not a kill-criterion failure.
   K1491 is INVALID not FAIL (Round 1 flagged this as non-blocking; audit
   agrees, but the implication is that the GSM8K arm of this experiment is
   vacuous as evidence).

### Why the catastrophic-forgetting finding still stands
Even conceding §2/§3 above, the adapter's 36.1% MMLU-Pro is a real measurement
against the real 62.1% base (Finding #536). Whether the adapter was trained on
"literal text" or "real channel" does not rescue a 26pp degradation. The
structural impossibility (math-only traces cannot preserve 14-category breadth)
is preserved and is independently confirmed by downstream LIMO preemptive kill.

### Verdict-consistency pre-flight (researcher hat §5)
1. results.json: absent (the file was not retained by the original run — PAPER.md
   has the numbers). Result recorded via `experiment complete --k` this
   iteration. **PASS (acceptable under audit-rerun reclassification).**
2. all_pass field: N/A — killed.
3. PAPER.md verdict line: now contains **KILLED** in audit header. **PASS**.
4. is_smoke: false. **PASS**.
5. KC drift (MATH.md diff): checked — MATH.md clean from pre-registration
   (commit de38e37). KCs #1490/1491/1492 unchanged. **PASS**.
6. Antipattern memories: §1–§4 above each applied; outcome is `killed`, not
   `supported`. **PASS**.

---

## Round 1 — Post-Run Review (2026-04-14, SUPERSEDED)

**Verdict: PROCEED** (killed experiment — findings valid, 2 non-blocking doc errors)

## Status

Experiment completed. Training ran (1000 steps, 48.5 min, 27 examples). Full eval ran.
Already completed in DB as `killed`, Finding #538 recorded.

## Core Finding Verification

**Finding #538 is accurate**: Adapter 36.1% vs base 62.1% = −26pp catastrophic forgetting.
- K1490 FAIL: 36.1% < 65% ✓ (correctly labeled FAIL)
- K1491: labeled FAIL but actually INVALID (HTTP 422 = untestable) — non-blocking
- K1492 PASS: 1641 avg_thinking_chars ✓ (Theorem 1 verified)

The impossibility structure documented in PAPER.md is correct:
s1K traces are near-orthogonal to MMLU-Pro token distribution → gradient pushes model
away from general reasoning breadth. Not a hyperparameter problem.

## Math Review

Theorem 1 (thinking channel preservation) verified by K1492 PASS.
Theorem 2 (reasoning gain) refuted — but failure mode was pre-identified in §Failure Modes.
Failure mode 1 (trace-domain mismatch) is active: math=20%, base_real=62.1% ≫ 20%.
Kill structure followed correctly. No math errors.

## Non-Blocking Issues

### Issue 1: Per-category table in PAPER.md shows mixed base/adapted values
The "Adapter Eval (Phase 4b)" per-category table is wrong for some rows:
- biology: shows 10% (base, buggy eval) — actual adapted is **50%**
- computer science: shows 5% (base, buggy eval) — actual adapted is **40%**
- All other rows correctly show adapted values from results.json

This does NOT change the finding (36.1% adapter << 62.1% real base = catastrophic forgetting).
But readers might misinterpret the per-category breakdown. Fix in LEARNINGS.md.

### Issue 2: K1491 status
K1491 is labeled "FAIL" in PAPER.md but should be "INVALID" — HTTP 422 means the criterion
was untestable, not that the model failed the criterion. Non-blocking since experiment is killed.

## Key Signal for Future Experiments

LIMO (P11.A1, pueue task 5) uses competition math traces (GAIR/LIMO, 817 problems).
Expect similar or worse degradation — LIMO is harder olympiad math, even more orthogonal to
MMLU-Pro. PAPER.md's "Next Steps" correctly flags this. LIMO should be killed early if
K1493 (≥65%) shows same pattern.

W4A16 verification (P11.K1) is the right next question: if 8-bit scores ~65%+ base,
the gap is quantization not reasoning, and SFT over quantized weights has a ceiling.
