# PAPER.md — P11.I0: Synthetic Reasoning Data Generation Loop (STAR)

**Date**: 2026-04-17 (preemptive-kill determination, supersedes 2026-04-14 design-phase draft)
**Status**: KILLED (preemptive, pre-run)
**Verdict**: KILLED

---

## §1 — Verdict

**KILLED (preemptive, pre-run).** Three independent drivers, any one of which is sufficient to kill:

1. **Replicated protocol bug (antipattern-018):** `run_experiment.py:278-281` embeds Gemma 4's raw `<|channel>thought...<channel|>` tokens as literal text inside `assistant.content` for `mlx_lm.lora` SFT. Byte-equivalent to B0 (`exp_p11_grpo_reasoning_adapter:267`), D0 (`exp_p11_meta_r1_metacognition:267`), and H1 (`exp_p11_metacognitive_adapter:260`). B0 measured −15.3pp MMLU-Pro regression and −71% thinking-length suppression from this exact pattern; D0 and H1 inherited the same mechanism and were preemptively killed.
2. **Upstream dependency dead:** DB `depends_on: exp_p11_grpo_improve` (G0). G0 is `status=killed` (2026-04-17) because ITS upstreams F0/F1 are both killed. There is no usable "best reasoning adapter" to serve as the v0 baseline implied by K1523.
3. **Structurally unreachable target:** MATH.md K1545 requires R1 adapter ≥ 59% MMLU-Pro. Measured base (baseline_eval 2026-04-17) is **40.7%** with thinking, not the 62.1% cited in MATH.md line 33 (Finding #530, which is stale — see F#560 baseline-reconciliation thread). STAR's published gains on MMLU-class tasks are +3–6pp from much larger trace corpora. Lifting from 40.7% to 59% is +18pp from ~30–40 correct-trace examples: unsupported by any prior measurement in this repo or the STAR paper.

**Two simultaneous mutually-exclusive KC sets** also block this experiment:
- DB-registered KCs: K1523 (v1 ≥ v0 +2pp on GSM8K), K1524 (cycle < 4h), K1525 (yield ≥ 50%).
- MATH.md KCs (git `de38e37`): K1544 (yield ≥ 45%), K1545 (R1 ≥ 59%), K1546 (R2 yield ≥ R1 − 5pp).

Per PLAN.md §1 kill-criteria discipline, divergence between pre-registered DB KCs and in-file MATH.md KCs is itself a definitional problem: which set governs `--status supported`? Under either set, the run fails for the reasons above.

---

## §2 — Why not run

### §2.1 Protocol bug replicated at line 278-281

```python
# run_experiment.py:276-281 (prepare_sft_data)
thinking = t["thinking"]      # contains "<|channel>thought ... <channel|>"
answer_letter = t["correct_letter"]
if thinking:
    assistant_content = f"{thinking}\nThe answer is {answer_letter}."
```

`t["thinking"]` is the raw span returned by `strip_thinking` at line 91 — it includes the literal `<|channel>thought` open tag and `<channel|>` close tag. When this string becomes `assistant.content` for an `{role:assistant, content:...}` record at line 286, `mlx_lm.lora` tokenises it as plain text, not as a control sequence. The adapter then learns to emit channel literals as output tokens rather than as a generation-protocol control. At eval time, with `enable_thinking=True`, the chat template introduces a real channel; the adapter's learned literals conflict → either answer-only collapse (like B0, −71% thinking) or malformed output.

**Prior measurements of this exact mechanism:**
- B0 (2026-04-17): MMLU-Pro 57.1% → 41.8% (−15.3pp); thinking 2819 → 816 chars (−71%).
- D0 (2026-04-17): preemptively killed, same code at line 267.
- H1 (2026-04-17): preemptively killed, same code at line 260; antipattern-018 canonical reference.

**Contrast (working pattern):**
- H0 (`exp_p11_thinking_adapter_universal`) used `<think>...</think>` OpenThoughts format; thinking-preservation K1519 **PASSED**. This format tokenises cleanly in Gemma 4.

### §2.2 Upstream chain dead

DB `depends_on: exp_p11_grpo_improve` → `status=killed` (2026-04-17, evidence: "KILLED preemptive/dependency-chain: upstream F0 (s1k) and F1 (LIMO) both killed; no usable reasoning-SFT adapter exists"). No v0 adapter exists to populate K1523's comparison.

The I0 code mitigates this partially — Phase 1 uses the BASE model, not the v0 adapter — but the MOTIVATION for I0 (build v1 on top of a reasoning-trained v0) is undermined. Phase 1 yield from base is a different experiment: STAR-from-scratch, not the self-improvement loop I0 was designed to validate.

### §2.3 K1545/K1525 structurally unreachable

MATH.md §33 cites "base accuracy ρ ≈ 0.62 → 62% of generated traces are correct" from Finding #530. But:
- baseline_eval (exp_p11_baseline_eval) measured **40.7%** on the same model (Gemma 4 E4B 4bit) with thinking on 2026-04-17.
- F#560 (from H0 LEARNINGS today) flagged baseline reconciliation as unresolved.

Under measured base 40.7%:
- Expected Phase 1 yield: ~28 / 70 correct = 40% (not 62%). K1544 borderline; K1525 (50% yield) likely FAIL.
- Required lift for K1545 (59% ≥ R1 ≥ 59%): +18pp from ~28 SFT examples. STAR's published gains on ARC/GSM8K were +2–5pp from hundreds of examples on the same base. No supporting prior.
- K1545 cannot be satisfied by the experiment as designed, regardless of harness correctness.

### §2.4 Kill-criteria divergence between DB and MATH.md

DB KCs (registered pre-MATH rewrite):
- K1523: synthetic v1 ≥ v0 + 2pp on GSM8K
- K1524: cycle < 4h on M5 Pro
- K1525: ≥ 50% trace yield

MATH.md KCs (current):
- K1544: yield ≥ 45%
- K1545: R1 ≥ 59% MMLU-Pro
- K1546: R2 yield ≥ R1 − 5pp

Pre-registration integrity requires ONE canonical set. The DB was not updated when MATH.md was rewritten; per §1011 and the verdict-consistency pre-flight, the run cannot be marked `supported` against KCs the DB doesn't hold.

---

## §3 — KC table (both sets)

| KC | Source | Pre-flight determination |
|----|--------|--------------------------|
| K1523 (v1 ≥ v0 + 2pp GSM8K) | DB | **FAIL**: v0 (G0 adapter) does not exist. Even if trivialised to "any adapter", comparison is vacuous against regressed upstream (B0 pattern: mlx_lm.lora adapter regresses base; no-op "v1 = base" passes trivially but has zero research value — H1 K1521 precedent). |
| K1524 (cycle < 4h) | DB | **N/A for kill**: would likely PASS on wall-clock (~2–3h), but protocol bug makes timing meaningless; "fast regression" is not a result. |
| K1525 (yield ≥ 50%) | DB | **FAIL**: measured base accuracy 40.7% < 50%. Phase 1 expected yield matches base; yield ≥ 50% not reached. |
| K1544 (yield ≥ 45%) | MATH.md | **BORDERLINE/FAIL**: expected ~40%, within ±5pp of threshold; likely FAIL. |
| K1545 (R1 ≥ 59% MMLU-Pro) | MATH.md | **FAIL**: +18pp from measured base with ~30 SFT examples is unsupported. Protocol bug makes regression (not gain) the likely outcome — per B0, expect 20–30% R1 accuracy. |
| K1546 (R2 yield ≥ R1 − 5pp) | MATH.md | **Vacuous**: if R1 suffers thinking suppression (−71% like B0), R2 generation collapses along with R1 yield; relative KC passes trivially on joint collapse. |

---

## §4 — Antipattern self-check (PLAN.md §1 pre-flight)

1. `results.json["verdict"]`: would be KILLED if run. ✓
2. `all_pass`: would be False. ✓
3. PAPER.md verdict line: KILLED. ✓
4. `is_smoke`: N/A (not running).
5. MATH.md git diff: single commit `de38e37`, no KC edits post-registration. However, DB KCs and MATH.md KCs diverge (pre-registration integrity issue). ⚠
6. Antipattern match:
   - **mem-antipattern-018 (CHANNEL-TOKENS-AS-SFT-TEXT)**: ✗ FAILS at line 278-281. Canonical replication of the B0/D0/H1 pattern. BLOCKING.
   - mem-antipattern-003 (LORA_SCALE > 8): ✓ scale=1.0 at line 58.
   - mem-antipattern-008 (eval truncation): ✓ MAX_TOKENS=2048, thinking not truncated.
   - mem-antipattern-001 (composition math): ✓ N/A (no composition).
   - mem-antipattern-012 (`shutil.copy` as new adapter): ✓ grep-clean.
   - mem-antipattern-013 (hardcoded `"pass": True`): ✓ grep-clean; KCs computed from measurements.

One blocking antipattern (018). Cannot mark `supported`.

---

## §5 — Unblock path

The entire P11 reasoning-adapter chain (B0/C0/D0/H1/I0/J0/M0) shares the same mlx_lm.lora training harness with the channel-tokens-as-SFT-text bug. Fix this ONCE; rerun each experiment.

**Fix options (decreasing invasiveness):**
1. **Strip channel tokens from `assistant_content` before SFT** (minimal change): in `prepare_sft_data`, extract *inner* thinking content without `<|channel>thought` / `<channel|>` tags, then either drop it entirely OR wrap it in `<think>...</think>` (H0 working format). Concretely, replace line 276-281 with:
   ```python
   # Strip channel wrapper, keep inner thinking body if non-empty
   inner = re.sub(r'<\|channel>thought\s*|\s*<channel\|>', '', thinking, flags=re.DOTALL).strip()
   if inner:
       assistant_content = f"<think>{inner}</think>\nThe answer is {answer_letter}."
   else:
       assistant_content = f"The answer is {answer_letter}."
   ```
   H0 evidence (K1519 PASS) confirms this format trains cleanly on Gemma 4.
2. **Switch to plain-prompt SFT** (no thinking in target, `enable_thinking=False` at both train and eval): abandons thinking-preservation claim but isolates knowledge-injection variable. Not suitable for I0's self-improvement claim.
3. **Custom MLX SFT loop** (bypass `mlx_lm.lora` CLI): full control over tokenisation, can feed channel tokens as true control sequences. High implementation cost; only justified if (1) and (2) both fail on a downstream experiment.

**Separate: KC reconciliation.** Before rerunning I0-v2:
- Re-measure base MMLU-Pro in-run (not F#530 proxy). Expect ~40%.
- Re-calibrate K1545 target as `base + 2pp` (STAR-compatible floor) rather than absolute 59%.
- Reconcile DB KCs ↔ MATH.md KCs; commit one canonical set to both.

---

## §6 — Assumptions (autonomous decisions logged per §1007)

1. **Preemptive kill rather than harness-fix + run in this claim:** the harness fix is a cross-experiment change affecting 7 experiments; fixing it inside I0's claim conflates "fix harness" with "verify I0 hypothesis". A separate `exp_p11_harness_fix` (or explicit unblock entry) is the correct atomic unit. This experiment is completed as the 7th instance of the same structural failure, finalising the data for Analyst to synthesise a cross-cut lesson.
2. **Use DB KCs (1523/1524/1525) for `--k` flag, not MATH.md KCs:** DB is authoritative for `experiment complete`; evidence row records both sets and notes the divergence.
3. **K1525 classified FAIL (not borderline):** measured base 40.7% < 50% threshold; yield tracks base. Even if thinking generation artificially inflates some category yields, aggregate structurally cannot exceed base absent some specialised selection.
4. **Not running smoke-mode rerun:** the 2026-04-14 smoke (PAPER.md §39-53) confirmed API correctness. Re-running smoke does not address the three kill drivers. No learning gained vs kill time.

---

## §7 — References

- **STAR** (`arxiv:2203.14465`) — Zelikman et al. 2022. Theorem 1 (filtered self-generation provides useful gradient signal); MATH.md §19-42.
- **B0 protocol-bug data** — `micro/models/exp_p11_grpo_reasoning_adapter/PAPER.md` (2026-04-17): measured −15.3pp MMLU-Pro, −71% thinking suppression.
- **H1 antipattern-018 canonical** — `micro/models/exp_p11_metacognitive_adapter/PAPER.md` (2026-04-17): same pattern, preemptively killed.
- **H0 working contrast** — `micro/models/exp_p11_thinking_adapter_universal/PAPER.md` (2026-04-17): `<think>...</think>` format + K1519 PASS.
- **Baseline reconciliation (F#560 open thread)** — measured 40.7% (baseline_eval 2026-04-17) vs cited 62.1% (F#530).
- **G0 dependency kill** — `micro/models/exp_p11_grpo_improve/PAPER.md` (2026-04-17).

---

## §8 — Handoff

Emitting `experiment.done` with `status=killed` payload. Reviewer should verify:
- Protocol-bug replication at line 278-281 (grep for `<|channel>thought`).
- Measured-base-vs-cited-base gap (40.7% vs 62.1%, F#560).
- DB/MATH.md KC divergence (rare pre-registration failure mode).

Open threads for Analyst:
- Cross-cut LEARNINGS across 7-experiment chain (F0/H0/B0/C0/D0/H1/I0) calling for a single P11.HARNESS unblock experiment with explicit scope.
- Memory update: antipattern-018 now has 4 confirmed instances (B0/D0/H1/I0) and 2 preemptively-killed same-code-path (C0/M0 pending claim); worth moving from "new pattern" to "established canonical antipattern" tier in memory.
