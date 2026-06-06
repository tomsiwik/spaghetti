# Adversarial Review: P11.C0 — ThinkPO Polish

**Reviewer**: Adversarial (post-kill determination)
**Date**: 2026-04-17
**Verdict**: **KILL** (endorse researcher's preemptive kill)

The 2026-04-14 PROCEED review (for pre-run design) is **superseded** by this
review, which assesses the experiment's preemptive kill on 2026-04-17 given
the 2026-04-17 kill of dependency `exp_p11_grpo_reasoning_adapter` (B0).

---

## Verdict Summary

Dependency `exp_p11_grpo_reasoning_adapter` (B0) was killed 2026-04-17 with
−15.3pp MMLU-Pro regression (57.1%→41.8%) and −71% thinking suppression
(2819→816 chars) from a protocol bug in `mlx_lm.lora` training. C0's
research question (does ThinkPO improve over GRPO reference policy?) is
structurally broken because π_ref is regressed below base. Running C0 would
consume ~30–60 min for a metric-pass / behavioral-fail configuration with
no falsification power. Preemptive kill is the correct action under
precedent set by P11.G0 (2026-04-17).

---

## Independent Verification

**B0 dependency state (verified via `experiment get`):**
- Status: `killed` (2026-04-17)
- K1496=fail (41.8% < 64%), K1497=fail (41.8% < 56.1%), K1498=fail
- Evidence row confirms −15.3pp MMLU-Pro regression

**C0 DB state (verified via `experiment get`):**
- Status: `killed`, updated 2026-04-17
- K1499=fail, K1500=fail, K1501=fail
- Evidence: preemptive kill citing upstream regression

**MATH.md KC integrity (git log):** single commit `de38e37`. No
post-registration KC edits. K1499/1500/1501 match DB pre-reg.

**Code antipatterns (re-verified):**
- `mlx_lm save` at `run_experiment.py:409-410` saves merged model, not LoRA.
  CONFIRMED (matches NB1 in superseded review).
- LoRA-on-LoRA stacking at `:361-372` on top of adapter loaded at `:527`.
  CONFIRMED: `LoRALinear.from_base(proj, r=4, scale=1.0)` applied after
  `load(MODEL_ID, adapter_path=str(GRPO_ADAPTER_PATH))`. Trainable params
  become `LoRA_thinkpo(LoRA_grpo(W_base))` — stacked on a regressed subspace.
- Fallback path at `:544-571` uses `if grpo_len > base_len` branch
  (`:555`). B0's measured 816 chars vs base's 2819 inverts the assumption
  → fallback collects zero pairs or, if collected, trains DPO toward base,
  *undoing* GRPO. CONFIRMED.
- `MIN_THINKING_DIFF=200` at `:51` × N=4 completions likely collapses
  phase 1 to <2 pairs given upstream suppression. STRUCTURAL RISK.

---

## Adversarial Checklist

**Consistency (a-d):**
- (a) `results.json` absent (preemptive, no run); DB `killed` matches PAPER
  verdict `KILLED`. ✓
- (b) N/A (no run); preemptive kill is structural.
- (c) PAPER.md line 5 reads `KILLED`. Matches DB status. ✓
- (d) `is_smoke` N/A (no run).

**KC integrity (e-g):**
- (e) MATH.md single commit `de38e37`; no post-reg KC relaxation. ✓
- (f) No tautology in the KCs themselves. PAPER.md correctly flags that
  K1499's *relative* framing (≥GRPO+2pp) means passing at 43.8% still
  sits −13.3pp below base — a behavioral-fail signal, not a tautology.
  This is a kill argument, not a design bug, under the
  behavioral-outcomes memory rule.
- (g) K1499/1500/1501 in code (`run_experiment.py:9-11`) match MATH.md
  `Kill Criteria` §130-142 and DB. ✓

**Code ↔ math (h-m):**
- (h) No `sum(lora_A` / `add_weighted_adapter` / independent safetensor
  summing. ✓
- (i) `scale=1.0` at `:368`. No LORA_SCALE=20. ✓
- (j) No single-sample routing applied to all (no routing in ThinkPO).
- (k) No `shutil.copy` of sibling adapter.
- (l) No hardcoded `{"pass": True}`.
- (m) `MODEL_ID = mlx-community/gemma-4-e4b-it-4bit` at `:40` matches
  MATH.md Theorem 2 prediction for E4B-4bit. ✓
- (m2) MATH.md does not cite `/mlx-dev` or `/fast-mlx` invocation.
  Non-blocking for preemptive kill; would be blocking for a run.

**Eval integrity (n-q):** N/A (no run).

**Deliverables (r-s):**
- (r) PAPER.md has prediction-vs-measurement table at §63-68 with
  "not measured" for all three predictions. Correct for preemptive kill. ✓
- (s) Kill argument is four-fold and each leg verified: (1) B0 protocol bug
  not re-runnable under current mlx_lm.lora, (2) C0 KCs relative to a
  regressed π_ref, (3) fallback inverted vs measured B0 thinking length,
  (4) LoRA-on-LoRA on regressed subspace. Sound.

---

## Kill Robustness

Preemptive kill does not depend on any single antipattern — it is robust
to the following alternative scenarios:

- **If researcher ran C0 anyway:** K1499 would likely pass at ~44% absolute
  (below base 57.1%) — behavioral-fail. K1500 would pass (DPO toward long
  traces) but still −68% vs base. K1501 would pass or near-pass.
  Metric-pass / behavioral-fail outcome confirms kill reasoning.
- **If fallback path collected pairs:** `if grpo_len > base_len` is false
  (B0 suppressed thinking), so fallback yields zero pairs and main path
  also collapses. Exit before training.
- **If fallback path somehow collected pairs:** DPO trains toward base →
  *undoes* GRPO. Outcome is closer to base than GRPO. Still no research
  signal for ThinkPO-on-healthy-GRPO.

All three branches of the decision tree end in "no research value."

---

## Assumptions

1. B0's killed verdict is authoritative. DB entry and B0 PAPER.md are both
   consistent at 2026-04-17; did not re-run B0 to verify the −15.3pp
   measurement. If B0 is re-measured and found healthy, revisit this kill.
2. The 2026-04-14 PROCEED review is superseded, not retracted for design
   bugs. At that time the design was mathematically sound and the dependency
   was unresolved; C0 is killed on the state of the B0 dependency *now*,
   not on a design flaw in C0 itself.

---

## Open Threads for Analyst

- **Cascade inheritance:** `exp_p11_meta_r1_metacognition` (D0) and
  `exp_p11_full_pipeline_v2` (M0) also depend on B0 or its output adapter.
  Expect similar preemptive-kill analysis if claimed before B0-v2 lands.
- **No new finding needed:** The preemptive-kill-on-upstream-regression
  pattern was already established by P11.G0 on 2026-04-17 (same iteration).
  The B0 protocol bug itself is already captured in B0's LEARNINGS and
  downstream findings.
- **Unblock path:** B0-v2 design requirements are documented in C0
  PAPER.md §Unblock Path and B0 PAPER.md §Unblock Path; both require
  replacing `mlx_lm.lora` thinking-channel handling.

---

## Passed Checks (carried from 2026-04-14 design review)

- Theorem 1 (DPO): standard Rafailov reparameterization, correct.
- Theorem 3 (offline DPO): reference log-probs as constants — sound.
- Theorem 4 (distribution alignment): D_train=D_eval argument valid.
- Failure modes 1-3 documented with detection + response.
