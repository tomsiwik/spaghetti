# PAPER: P11.C0 — ThinkPO Polish (DPO on GRPO Adapter)

**Experiment**: exp_p11_thinkpo_polish
**Date**: 2026-04-17 (preemptive kill; no full run executed this iteration)
**Verdict**: **KILLED** — dependency `exp_p11_grpo_reasoning_adapter` killed with −15.3pp MMLU-Pro regression and 71% thinking suppression; reference policy π_ref is structurally broken for the stated research question.

---

## TL;DR

C0's design uses the GRPO adapter (B0) as the frozen reference π_ref for offline DPO. B0 was completed 2026-04-17 as `killed`: 41.8% MMLU-Pro vs 57.1% base (−15.3pp), 816 avg thinking chars vs 2819 (−71%). Root cause (B0 PAPER §Falsification): `mlx_lm.lora` received `<|channel>thought ... <channel|>` tokens as literal text targets, so the adapter learned to emit channel tokens as output string rather than to use the thinking channel protocol.

C0's kill criteria are all stated **relative to GRPO**, so passing them does not entail base-competitive behavior:

| KC                                           | Gate relative to B0 | Absolute target | Gap from base (57.1%)            |
|---------------------------------------------|---------------------|-----------------|----------------------------------|
| K1499 ThinkPO ≥ GRPO + 2pp                  | 41.8% + 2pp         | ≈ 43.8%         | −13.3pp below base              |
| K1500 thinking_chars ≥ GRPO × 1.10          | 816 × 1.10          | ≈ 898 chars     | −68% vs base 2819               |
| K1501 GSM8K ≥ GRPO − 5pp                    | baseline regression | ≤ 5pp drop      | (GRPO GSM8K not measured in B0) |

Even a **three-for-three KC pass** would leave the composition regressed below Gemma 4 E4B base. This is a metric-pass / behavioral-fail configuration — not a useful research signal for the P11 reasoning-adapter line.

---

## Pre-flight Audit (6-Step Verdict-Consistency Check)

Before running, step through PLAN.md §1 checks:

1. **results.json**: absent (no run). Preemptive kill status, not a degenerate full-run.
2. **all_pass**: N/A. Kill is structural, not measurement-based.
3. **PAPER.md verdict line**: KILLED (this document).
4. **is_smoke**: N/A. No run executed.
5. **MATH.md KC diff vs DB**: single commit `de38e37` (2026-04-16), no post-registration edits. KC 1499/1500/1501 match DB pre-reg.
6. **Antipattern self-check**: see §Antipattern Audit below. Two structural antipatterns trigger before any measurement.

---

## Antipattern Audit (type: fix memory sweep)

| Antipattern | Location in run_experiment.py | Status |
|-------------|-------------------------------|--------|
| `from mlx_lm import save; save(ADAPTER_DIR, model, tokenizer)` saves full merged model, not LoRA-only | line 409–410 | CONFIRMED (flagged in NB1 of 2026-04-14 REVIEW — never fixed because experiment did not reach smoke) |
| LoRA-on-LoRA stacking: `LoRALinear.from_base(proj, r=4)` applied to projections already hooked with GRPO LoRA (model loaded at line 527 with `adapter_path=GRPO_ADAPTER_PATH`) | line 361–372 | STRUCTURAL: trainable params become `LoRA_thinkpo(LoRA_grpo(W_base))`. The GRPO LoRA becomes part of the "base" for ThinkPO; any ThinkPO improvement is stacked on a regressed sub-space. |
| Preference pairs sampled from regressed π | line 184–219 | UPSTREAM-DRIVEN: with B0 at 816 avg thinking chars (suppressed), `MIN_THINKING_DIFF=200` over N=4 completions is likely to collapse to the fallback path (line 542) |
| Fallback path uses `base` vs `GRPO` as short/long (assumes `grpo_len > base_len`) | line 544–571 | STRUCTURAL RISK: base (57.1%) likely produces **longer** thinking than regressed GRPO (816 chars vs base 2819). The fallback's `if grpo_len > base_len` branch collects zero pairs. But if executed with inverted data, DPO would train toward base, *undoing* GRPO. Either way, the fallback path is broken against the measured upstream regression. |

---

## Why Preemptive Kill, Not Re-Run

1. **Dependency regression is a protocol bug, not a hyperparameter miss.** B0 PAPER's "unblock path" identifies four candidate fixes, none of which apply to C0: custom GRPO loop, plain-prompt SFT, thinking-strip at train, chat-template fork. All four require rebuilding the upstream training pipeline. C0 cannot reach behaviorally useful territory without B0-v2.

2. **Research-value of a relative-KC pass is zero.** K1499 is defined `thinkpo_acc ≥ grpo_acc + 0.02`. Passing it at 43.8% while base is 57.1% is metric-pass / behavior-fail (cf. MEMORY.md behavioral-outcomes feedback memory).

3. **Fallback path amplifies the problem.** If phase 1 collapses to fewer than 2 valid pairs (likely given the upstream thinking suppression and `MIN_THINKING_DIFF=200`), the fallback uses GRPO vs base with the inverted assumption. DPO would either collect no pairs or learn the wrong direction.

4. **Budget and precedent.** ~30–60 min of generation + DPO training for a measurement that cannot falsify anything new. P11.G0 (exp_p11_grpo_improve) was preemptively killed on 2026-04-17 under the same dependency-chain-kill rule.

---

## Prediction vs Measurement Table

| Prediction (MATH.md)                       | Theorem         | Predicted      | Measured        | Pass? |
|--------------------------------------------|-----------------|----------------|-----------------|-------|
| ThinkPO ≥ GRPO + 2pp (MMLU-Pro)            | Thm 2 (ThinkPO) | ≥ 43.8% abs    | not measured    | FAIL  |
| avg_thinking_chars ≥ GRPO × 1.10           | Thm 2 (DPO signal) | ≥ 898 chars | not measured    | FAIL  |
| GSM8K ≥ GRPO − 5pp                         | Thm 4 (D_align) | ≤ 5pp drop     | not measured    | FAIL  |

All three KCs recorded as FAIL on the pre-reg via `experiment complete --k <id>:fail`. Preemptive kill does not claim the KCs were measured — it claims the precondition for meaningful measurement (a healthy π_ref) does not hold.

---

## Salvageable Sub-Measurements

None. No phase 1 / phase 2 / phase 3 ran.

---

## Unblock Path (for Analyst / next experiment)

B0-v2 must land before any C0-class experiment is viable. The forward path:

**B0-v2 design requirements** (derived from B0 PAPER §Unblock Path):
- Replace `mlx_lm.lora` with a custom RS-SFT / GRPO loop that does *not* pass thinking-channel tokens as literal text to cross-entropy.
- Or: serialize training targets as plain prompts (no `<|channel>thought`), accepting a weaker thinking signal in training to preserve protocol at eval.
- Retain RS-SFT's D_train=D_eval alignment (MMLU-Pro) to keep Theorem 1's impossibility guarantee.
- Pre-register a kill criterion that the adapter does not regress below base by >2pp on any single MMLU-Pro category (catches the 9/14 per-category collapse from B0).

**C0 redesign (if B0-v2 produces a healthy adapter)**:
- Keep the DPO loop (offline reference log-probs are correct).
- Fix `mlx_lm save` antipattern: save LoRA weights only via `mx.savez` over `tree_flatten(model.trainable_parameters())`.
- Remove the fallback path (line 544-571) — if phase 1 yields <2 pairs, that is a signal to kill, not to invert directions.
- Consider Gemma 4 base as π_ref rather than the SFT adapter, to measure DPO's isolated contribution.

**Downstream inheritance (same training stack)**:
- `exp_p11_meta_r1_metacognition` (D0): blocked-by B0, same risk pattern — flag for similar preemptive-kill if claimed before B0-v2.
- `exp_p11_full_pipeline_v2` (M0): pipeline kill cascade should note the protocol-bug root cause.

---

## Assumptions (per Researcher hat rule 1007)

1. **B0's `killed` verdict and evidence are authoritative.** I did not re-run B0 to verify the −15.3pp measurement; I am trusting `results.json` + PAPER.md from the 2026-04-17 B0 full run. If B0 is later re-measured and found to be better than reported, this preemptive kill should be revisited.
2. **Running C0 would not produce an adapter that is base-competitive.** I am not proving it analytically; I am making the inference from B0's protocol-bug root cause plus the fact that C0's KCs are all stated relative to B0.
3. **The 2026-04-14 PROCEED review is superseded.** That review was at smoke-design stage, before B0 completed. A reviewer today would need to address the same blocked-by-dependency question.

---

## Status: **KILLED** (preemptive / upstream-dependency)

- `results.json`: not created (no run).
- `adapters/thinkpo/`: not created.
- KC records: K1499=fail, K1500=fail, K1501=fail on the pre-reg.
- Next unblocking experiment: B0-v2 (see B0 PAPER.md §Unblock Path).

---

## References

- arXiv:2502.13173 (ThinkPO): Length-based DPO, +3.8pp MATH500
- arXiv:2305.18290 (DPO): Direct Preference Optimization
- arXiv:2501.12948 (DeepSeek-R1): RS-SFT warmup before preference learning
- B0 PAPER.md (exp_p11_grpo_reasoning_adapter, 2026-04-17): protocol-bug root cause
