# PAPER: P11.D0 — Meta-R1 Metacognition (Planning, Regulation, Early Stopping)

**Model**: mlx-community/gemma-4-e4b-it-4bit
**Date**: 2026-04-17 (preemptive-kill pre-flight; supersedes 2026-04-14 PROCEED)
**Status**: **KILLED (preemptive, pre-run)**

---

## 1. Verdict Summary

Preemptive kill, no full run executed. KCs will not be materialized because the training
code contains the same protocol bug that killed `exp_p11_grpo_reasoning_adapter` (B0) on
2026-04-17 (−15.3pp MMLU-Pro regression, −71% thinking suppression). Combined with a
format-injection design that structurally contradicts K1502, running the experiment would
burn ~2.5–3 hours of compute reproducing a known failure mode.

All three KCs are marked fail in the DB with evidence citing: (a) B0 protocol bug replicated
at `run_experiment.py:267`, (b) format injection adds chars rather than reducing them, and
(c) K1503 baseline reference is now stale (`BASE_ACCURACY_REFERENCE = 0.621` cites Finding
\#530, but `baseline_eval` measured 40.7% on this model in-run).

This pattern parallels the preemptive kill of `exp_p11_thinkpo_polish` (C0) on the same day.

---

## 2. Why Not Run

### 2.1 Protocol bug replicated from B0 (line 267)

```python
structured_response = f"<|channel>thought\n{structured_thinking}\n<channel|>{answer_part.strip()}"
```

`structured_response` becomes the `assistant` content in the jsonl fed to `mlx_lm.lora`. The
Gemma 4 chat template does **not** recognize `<|channel>thought` / `<channel|>` as special
tokens when they appear inside the content string — they tokenize as literal text. The
adapter trains to emit those characters as output, not to invoke the thinking channel.

This is the exact bug diagnosed in `exp_p11_grpo_reasoning_adapter/PAPER.md` (2026-04-17):
B0's training fed identical `<|channel>thought...<channel|>{answer}` payloads to
`mlx_lm.lora`, and at evaluation the adapter produced −71% thinking chars (2819 → 816) and
−15.3pp accuracy (57.1% → 41.8%) with catastrophic per-category drops (math −42.9pp,
physics −42.9pp, 9/14 cats regressed >5pp).

D0 uses the same training stack (`mlx_lm.lora` + Gemma 4 E4B + this serialization at
line 267). There is no reason to expect a different outcome at inference.

Contrast: H0 (`exp_p11_thinking_adapter_universal`) used the standard `<think>...</think>`
OpenThoughts format inside assistant content, which tokenizes cleanly and DID produce
thinking at eval (K1519 PASS, 2902 chars/q). H0 was killed on a different axis (gradient
diversity, Theorem 1 violation). The protocol bug is specific to embedding the raw
Gemma-channel tokens as literal text.

### 2.2 Format injection contradicts K1502 (30% reduction)

`restructure_trace` at line 98 wraps the raw correct thinking trace:

```
META_PLAN_PREFIX  = "PLAN: I need to analyze this problem carefully.\n"        # ~55 chars
META_CHECK_SUFFIX = "\nCHECK: Based on my analysis, the answer is confirmed."  # ~55 chars
```

Training traces become `PLAN_prefix + raw_thinking + CHECK_suffix`. At base thinking ~3086
chars (Finding \#530 reference), training traces are **longer** than base by ~110 chars,
not shorter. The MATH.md Theorem 1 prediction (T_meta ~ 600–1100 chars) assumes the
model learns to terminate early at CHECK; there is nothing in the 200-step SFT signal that
teaches "stop thinking at CHECK" as an exit rule — the training examples show the model
producing the full long trace, then CHECK, then answer. The 2026-04-14 smoke-round-2
reviewer flagged this explicitly: "K1502 LIKELY FAIL — format injection adds overhead"
(see REVIEW-adversarial.md round 1, lines 62–68).

Structural lesson: the MATH.md derivation of T_meta ~ 600–1100 chars requires training
traces to BE 600–1100 chars. Training on 3086-char traces cannot teach 600-char behavior
without either curriculum (short-only traces) or RL-style length reward.

### 2.3 K1503 baseline reference is stale

```python
BASE_ACCURACY_REFERENCE = 0.621  # Finding #530 — base model 62.1% MMLU-Pro with thinking
```

But `exp_p11_baseline_eval` measured 40.7% on the same model in the same 2026-04-17
reporting round (see H0 PAPER.md baseline reconciliation note). Whichever baseline is
canonical, K1503 as coded compares against a frozen 62.1% that may not reflect this run's
environment (chat-template version, mlx-lm version, seed). A live K1503 evaluation would
need an in-run `base_model` phase (which the code does have — `phase3a_base`) and gate on
that local number, not the hardcoded 0.621. This is a design issue but not the primary
kill reason.

### 2.4 Cascade pattern: four consecutive mlx_lm.lora Gemma 4 reasoning adapters killed

| Experiment | Status | Kill cause |
|---|---|---|
| P11.F0 (`exp_p11_s1k_reasoning_train_eval`) | killed 2026-04-17 | OOM + 0% yield data |
| P11.H0 (`exp_p11_thinking_adapter_universal`) | killed 2026-04-17 | Theorem 1 GD violation (2 STEM domains) |
| P11.B0 (`exp_p11_grpo_reasoning_adapter`) | killed 2026-04-17 | Protocol bug (channel tokens as text) |
| P11.C0 (`exp_p11_thinkpo_polish`) | killed 2026-04-17 | Preemptive (stacked on B0) |
| **P11.D0 (this)** | **killed 2026-04-17** | **Preemptive (same protocol bug as B0)** |

The structural issue is the training stack, not the individual experiment designs. All
downstream reasoning-adapter work (`exp_p11_metacognitive_adapter` H1,
`exp_p11_adapter_composition_thinking` J0, `exp_p11_full_pipeline_v2` M0) depends on
first fixing the protocol bug in a shared training harness.

---

## 3. Prediction vs Measurement Table (updated for post-run state)

| Kill Criterion | 2026-04-14 prediction | 2026-04-17 determination | Result |
|---|---|---|---|
| K1502: avg thinking ≤ 2160 chars | LIKELY FAIL (format injection) | FAIL — protocol bug suppresses thinking to near-zero (per B0) *or* format injection inflates beyond boundary | fail |
| K1503: accuracy ≥ 62.1% | UNCERTAIN | FAIL — B0-class regression expected (−10 to −15pp from base); stale baseline reference | fail |
| K1504: ≥ 50% traces with PLAN structure | LIKELY PASS trivially | FAIL — if adapter learns to emit literal `<|channel>thought` prefix, `has_metacognitive_structure` regex won't match that as PLAN; if adapter suppresses thinking entirely, there is no trace to structure | fail |

No full-run data was collected. Rows marked fail on design-level evidence, not measurement.
If a v2 experiment is designed with the protocol fix, K1502–K1504 should be re-specified
against a locally-measured baseline and a training format that does not embed channel
tokens as literal content.

---

## 4. Assumptions (logged, not verified in this run)

- B0's diagnosis is correct: the `<|channel>thought...<channel|>` literal-text antipattern
  is the dominant kill cause and would carry over to D0 with the same magnitude. If B0's
  diagnosis is later revised (e.g. the bug turns out to be a chat-template version issue
  unrelated to content-embedded channel tokens), this preemptive kill is over-conservative
  and D0 should be re-claimed.
- Format-injection KC semantics (K1502) cannot be satisfied by 200 steps of LoRA SFT on
  traces that are longer than the target length.
- Finding \#530's 62.1% is either measurement drift or environment difference; resolving
  this is a baseline-hygiene issue (tracked for the whole P11 chain, not D0-specific).

---

## 5. Unblock Path

The blocking bug for D0 and the entire P11 reasoning-adapter chain is "how do we train
adapters that use Gemma 4's thinking channel via `mlx_lm.lora`?" Options:

1. **Custom chat-template fork** (medium effort): Patch the chat template so that assistant
   content containing `<|channel>thought` is tokenized with the channel special tokens.
   Requires an upstream PR to `mlx-lm` or a local tokenizer subclass.
2. **Plain-prompt SFT** (low effort, low upside): Strip thinking entirely, train adapter
   on `{user: question, assistant: answer_letter}`. Abandons the thinking-mode emergence
   hypothesis — what H0 was trying to prove.
3. **Actual GRPO with custom MLX loop** (high effort, highest upside): Implement GRPO
   reward-on-answer-letter with the thinking channel invoked during generation only; no
   channel tokens in training content. Matches Meta-R1 paper's training setup.

For D0's specific goal (token reduction via metacognitive structure), path 3 is the only
one that preserves the research question. Option 2 abandons thinking mode; option 1 fixes
the protocol but still trains on long traces, so K1502 remains structurally blocked
unless combined with short-trace curriculum.

Recommendation for the P11 backlog: ship a shared training-harness experiment
(B0-v2 or dedicated) that resolves the protocol bug once, then re-spec D0, C0, H1, J0, M0
on that harness.

---

## 6. Antipattern Self-Check

- **mem-antipattern-002 (tautological routing)**: N/A — this experiment has no routing, single adapter.
- **mem-antipattern-003 (LORA_SCALE=20)**: CLEAN — `LORA_SCALE = 1.0` at line 65.
- **B0 protocol bug (channel tokens as literal text)**: **MATCH at line 267** — primary kill cause.
- **Composition math bug**: N/A — no composition.
- **shutil.copy as new adapter**: N/A — trains a new adapter via `mlx_lm.lora`.
- **Hardcoded `pass: True`**: CLEAN — KCs computed from live metrics (though baseline reference is stale).
- **Eval-template truncation producing base=0%**: CLEAN — baseline run is `phase3a_base` with same prompt as adapter eval.
- **Proxy model substitution**: CLEAN — target Gemma 4 E4B 4-bit is used throughout.
- **KC measures wrong object**: PARTIAL — K1503 measures meta-r1 accuracy vs a hardcoded 62.1% instead of the in-run base. Not the primary kill cause but worth fixing in any v2.
- **N=smoke reported as full**: CLEAN — `IS_SMOKE` guard active.

Pre-flight fails on the B0 protocol-bug match. Running the experiment cannot produce a
supported verdict on K1502 or K1503.

---

## References

- `exp_p11_grpo_reasoning_adapter/PAPER.md` — B0 kill analysis (2026-04-17), protocol bug diagnosis
- `exp_p11_thinking_adapter_universal/PAPER.md` — H0 kill analysis (2026-04-17), baseline reconciliation
- `exp_p11_thinkpo_polish/PAPER.md` — C0 preemptive kill (2026-04-17), precedent pattern
- Finding \#553 (tautological routing antipattern, does not apply here)
- Finding \#559 (dispatch axis, does not apply here)
- Finding \#560 (gradient diversity, applies to H0; D0's cascade is protocol, not GD)
- arXiv:2508.17291 — Meta-R1 metacognitive framework (paper hypothesis preserved for v2)
- arXiv:1612.00796 — EWC forgetting (Theorem 2 support preserved for v2)
