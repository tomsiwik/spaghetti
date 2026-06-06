# MATH.md — exp_hedgehog_behavior_adapter_conciseness

**Claim.** Per-layer cosine-similarity distillation (Hedgehog) of the 26B Gemma 4
teacher under a concise-output system prompt (π_Concise) into a rank-8 LoRA on
`(v_proj, o_proj)` of Gemma 4 E4B trains a **conciseness adapter** that (a)
reduces mean output length by ≥ 20 % on neutral held-out prompts (K1881 target
— behavioral acquisition) AND (b) preserves task accuracy within a 3 pp
degradation budget (K1882 target — substance-safety, one-sided). This is the
**3rd behavior-axis instance** in the Hedgehog-framework (cousin of F#683
politeness, F#724 formality), and the **2nd dual-target / zero-proxy KC
design** in the super-family (1st was F#724). K1882 is an *asymmetric safety*
KC class — one-sided degradation-only — which differentiates it from F#724
K1880 two-sided orthogonality and extends the zero-proxy design pattern.

**This filing triggers the behavior-axis sub-cluster standalone memory
promotion** per analyst's 3-instance-on-same-sub-cluster threshold (F#683 1st
instance → F#724 2nd instance → THIS 3rd instance).

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns,
  `mx.eval` discipline at step boundaries, `mx.clear_cache` between phases,
  `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval,
  bandwidth-aware kernels). Both MUST be invoked before the MLX training-loop
  code lands in the `_impl` follow-up. Hard gate per Finding #673 and
  2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed
  at run time.
- **Student model:** `mlx-community/gemma-4-e4b-it-4bit` (HF repo id).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` (sequential-phase
  eviction per F#673 on 48 GB M5 Pro).
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Axis pinning.** Conciseness axis is orthogonal to politeness (F#683) and
  to formality (F#724). Concise-output prompts (short-form, bullet-like, no
  hedging) vs verbose-output prompts (long-form, elaborative, multi-paragraph)
  — a length/output-structure axis independent of register style.
- **Scope-preservation (antipattern-t).** PROVISIONAL filing: do NOT
  substitute smaller teacher, different axis, or different step count. Full
  Phase A/B/C/D pipeline lands in `_impl`.

## 1. Failure mode

Primary degenerate behavior the experiment guards against: **"Conciseness cuts
content."** A conciseness adapter trained via per-layer cos-sim distillation on
a concise-teacher capture may shift attention routing in ways that truncate
*substantive answer content*, not just *output length*. Under this failure
regime, the adapter reduces length (K1881 PASS) but degrades task accuracy by
> 3 pp (K1882 FAIL). The most insidious instantiation: the adapter learns to
stop generating *before the answer is complete* (e.g., on a multi-step math
question it emits just the setup then terminates), which looks concise but is
substantively wrong.

Secondary failure: **"Conciseness is null."** Cos-sim distillation captures
attention-output structure but not output-length routing. If concise-vs-
verbose teacher capture differs only in post-attention sampling behavior (e.g.,
EOS probability at head layers) — which rank-8 LoRA on `(v_proj, o_proj)`
cannot influence directly — then training is a no-op: K1881 < 5 % length
reduction AND K1882 ≈ 0 pp accuracy drift. Under this mode, conciseness is a
*decoder-head* phenomenon not a *routing* phenomenon, and Hedgehog cos-sim
distillation is the wrong inductive bias. Detection: K1881 < 5 % paired with
K1882 < 0.5 pp change ⇒ adapter is learning nothing; rank or training-step
budget is misspecified.

Tertiary failure: **"Conciseness conflates with truncation."** If the teacher
under π_Concise simply emits shorter sequences via early-EOS rather than via
denser, more informative tokens, the student learns the same truncation
behavior — giving a K1881 length reduction that is structurally equivalent to
`max_tokens` hard-capping. Detection: the K1881 reduction should track
*information density per token* staying roughly constant or increasing, not
simply fewer tokens with the same density. A follow-up density audit is
deferred to `_impl` (compute mean informative-content per token via
perplexity-weighted coverage).

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim loss
  (baseline definition).
- **Zhang et al. 2402.04347:** cosine loss recovers 99 % of softmax attention
  behavior with MLP feature maps. Evidence for cos-sim as a dense training
  signal.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` of Gemma 4 E4B captures
  domain specialization end-to-end. Parameterization is sufficient at rank 8.
- **Sibling Hedgehog-axis PROVISIONAL precedents:** F#683 politeness
  (behavior axis, 1st instance); F#724 formality (behavior axis, 2nd
  instance); F#684 procedural-refactor, F#696 JS, F#697 Python, F#717 Rust,
  F#718 SQL (domain axes 1-5 — closed after F#718); F#719 KL-loss + F#720
  MSE-loss (loss-variant); F#721 layer-selection-top6; F#722 temperature-
  sweep; F#723 prompt-rephrase-augmentation.
- **F#666 target-gating convention.** K1881 + K1882 are BOTH target KCs (no
  proxy). Verdict matrix gates on both targets. **Pure-proxy preempt-KILL does
  NOT apply** (no proxy KCs); **§5 tautological-inter-variant-delta does NOT
  apply** (KCs grounded to external length measurement + external task
  benchmark, not inter-variant deltas).
- **`mem-impossibility-f666pure-saturation-implies-f702-unavailable` —
  inapplicable** (both KCs target — F#702 hygiene-patch path is AVAILABLE).
- **F#614/F#536:** thinking-mode load-bearing on Gemma 4 (`enable_thinking=True`).
- **F#328/F#330:** LORA_SCALE ≤ 8.
- **F#673 + 2026-04-17 audit:** `mx.clear_cache` between phases; sequential-
  phase eviction for 26B residency on 48 GB.

## 3. Derivation sketch

1. *Existence.* Output length is not a pure sampling-head phenomenon — prior
   work (e.g., length-controlled generation via routing ablation, Jiang et al.
   2023) shows attention patterns differ meaningfully between concise and
   verbose teacher outputs. Per Zhang 2024, attention output is a 99 %-
   faithful proxy for the model's surface-form distribution. A concise-output
   teacher capture varies π_Concise vs π_Null at constant Q content; the
   per-layer attention-output delta encodes length-specific routing. Rank-8
   LoRA on `v_proj+o_proj` has sufficient degrees of freedom to fit this
   length delta (Pierre F#627 precedent at rank 6 fit domain shifts).
2. *K1881 target-gated per F#666.* Behavioral conciseness measured as
   `1 − mean_tokens(adapter) / mean_tokens(base)` on a 50-prompt held-out
   neutral set. K1881 KILL condition: reduction < 20 % (no meaningful
   behavioral acquisition).
3. *K1882 target-gated per F#666 — one-sided safety KC class.* Task accuracy
   on an answer-quality benchmark (MMLU-100 subset seed=42 OR TriviaQA-100
   subset) under the adapter vs base. K1882 KILL condition:
   `accuracy(base) − accuracy(adapter) > 3 pp` (strictly degradation;
   accuracy improvements do NOT violate — they are rare but benign). Note
   asymmetry vs F#724 K1880 (two-sided): the KC canonical text in DB says
   "drops task accuracy > 3pp" which is one-sided.
4. *Verdict matrix (DUAL TARGET, no proxy, K1881 two-sided+ K1882 one-sided).*
   - **SUPPORTED** = K1881 PASS (length reduction ≥ 20 %) ∧ K1882 PASS
     (accuracy drop ≤ 3 pp). Adapter acquired conciseness without substance
     degradation — framework generalizes to a 3rd behavior axis with
     asymmetric safety.
   - **KILLED** = K1881 FAIL ∧ K1882 FAIL. No length reduction AND accuracy
     crashes — adapter learned an unrelated routing perturbation; framework
     does NOT generalize to conciseness.
   - **PROVISIONAL (K1881 PASS + K1882 FAIL)** = conciseness acquired at
     cost of accuracy — finding: cos-sim distillation cannot separate
     length-reduction from content-preservation at this rank/scale; framework
     needs SIGReg or length-vs-content-decomposed loss.
   - **PROVISIONAL (K1881 FAIL + K1882 PASS)** = no length reduction but no
     accuracy drop — adapter is null, training is benign; finding:
     conciseness-axis is sub-threshold for cos-sim distillation at rank 8 /
     800 steps; needs different loss (length-aware CE? EOS-weighted KL?) or
     larger rank.
5. *Bounds.* Length reduction ∈ [0, 1]; 20 % floor based on the rule-of-thumb
   "meaningful behavioral change is ≥ 20 % effect size" and on F#683-analogue
   signal strength. Task accuracy ∈ [0, 100]; 3 pp drift threshold is looser
   than F#724 (2 pp) because conciseness is known to have a stronger content-
   truncation risk mechanism than formality (formality is lexical; conciseness
   involves answer-completeness).

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1881 | `1 − mean_tokens(adapter) / mean_tokens(base)` over 50 held-out neutral prompts | < 0.20 strictly | target — behavioral acquisition |
| K1882 | `accuracy(base) − accuracy(adapter)` on 100-question MMLU subset (seed=42) | > 3 pp strictly (one-sided — degradation only) | target — substance safety (one-sided variant of F#724 K1880 two-sided orthogonality) |

**F#666 target-gating.** Verdict matrix per §3.4 — DUAL-TARGET design (no
proxy KC). Pure-proxy preempt-KILL does NOT apply. §5 tautological-inter-
variant-delta does NOT apply (K1881 grounded to external token count; K1882
grounded to MMLU canonical answers — both external ground truth, not inter-
variant deltas).

**KC-design classification.** Hedgehog-framework KC-design taxonomy:
- **target+proxy paired** (canonical): F#683 (K1 cos-sim proxy + K2/K3/K4
  targets), F#719/720/721/722 (cos-sim proxy + behavioral target).
- **dual-target / zero-proxy, two-sided safety**: F#724 (1st).
- **dual-target / zero-proxy, one-sided safety**: THIS (2nd zero-proxy
  instance, 1st one-sided-safety sub-variant).
- **target+proxy intra-stability**: F#723 (cos-sim variance + behavioral
  target).

This is the **2nd zero-proxy KC design** in the super-family (1st was F#724),
and the **1st one-sided-safety sub-variant**. Justified by K1882 being an
*asymmetric safety* target (only degradation matters; improvements are
benign), which is a structurally distinct safety-target class from F#724's
two-sided orthogonality.

**Behavioral length reduction (K1881).** `mean_tokens(adapter) /
mean_tokens(base)` computed over 50 held-out neutral prompts. Both variants
generated with `max_tokens = 256`, `temperature = 0`, stop tokens identical.
Token count is post-tokenizer count (not character count). Mean over 50
prompts; report `1 − ratio`.

**MMLU subset (K1882).** 100 random questions sampled from MMLU dev split,
seed=42. Categories balanced across STEM, humanities, social-science.
Adapter applied at inference; greedy decoding, max_tokens=64 (same for
K1882 eval to match K1882 budget; K1881 uses 256 because length-headroom is
the measured variable). Same 100 questions for both adapter and base
evaluation.

## 5. Predicted measurements

- K1881: length reduction `ρ ∈ [+10 %, +40 %]`; mean prediction +25 %.
  **Expected PASS** based on teacher-behavior dynamics (π_Concise typically
  halves teacher output lengths) and F#683-analogous signal strength.
- K1882: `Δ_accuracy ∈ [+1, −6] pp`; mean prediction −2.5 pp degradation.
  **Expected PASS but borderline.** The content-truncation failure mode (§1
  primary) is a known concern; if truncation triggers on multi-step answers,
  accuracy drop can exceed 3 pp.

**Most likely outcome:** SUPPORTED (both PASS) at probability ~45 %;
PROVISIONAL (K1881 PASS + K1882 FAIL — content truncation) at ~35 %;
PROVISIONAL (K1881 FAIL + K1882 PASS — sub-threshold) at ~15 %; KILLED at
~5 %.

If SUPPORTED, finding: "Hedgehog cos-sim distillation generalizes to a 3rd
behavior axis (conciseness) with asymmetric safety at rank 8. Behavior-axis
sub-cluster now has 3 supported axes — standalone memory promoted for
behavior-axis sub-cluster."

If PROVISIONAL (K1881 PASS + K1882 FAIL), finding: "Conciseness acquisition
truncates task-critical content; asymmetric safety requires SIGReg-style
explicit constraint or length-vs-content loss decomposition. Cos-sim alone is
insufficient for safety on length axes."

If PROVISIONAL (K1881 FAIL + K1882 PASS), finding: "Conciseness is sub-
threshold for cos-sim distillation at rank 8 / 800 steps; length control may
require EOS-aware loss or sampling-head fine-tuning. Behavior-axis sub-cluster
does not generalize trivially to length axes."

If KILLED, finding: "Length is a decoder-head phenomenon — cos-sim
distillation on attention outputs does not shift EOS probability. Hedgehog
cos-sim is the wrong inductive bias for conciseness; behavior-axis sub-
cluster requires non-attention-output methods for length control."

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Neutral prompt curation.** Generate 250 neutral prompts (200
   train + 50 held-out) covering knowledge questions, descriptive tasks,
   problem-solving — prompts that are register-neutral AND length-neutral
   (no implicit "in one sentence" or "in detail" cue). Reuse pipeline pattern
   from F#683/F#724 Phase 0 but NEW prompt content; register-neutral set may
   overlap with F#724 if register-neutrality is preserved.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + π_Concise + Q in
   context (concise-output system prompt: "You are a concise assistant.
   Reply in the fewest words necessary. Never elaborate beyond what the
   question requires. No preamble, no disclaimers."). Capture
   `{layer_idx: attn_output}` for all 42 layers per (Q, A) pair. Sequential-
   phase eviction per F#673; pre-compute offline.
3. **Phase B — Student training.** Rank-8 LoRA on `(v_proj, o_proj)` with
   per-layer cos-sim loss: `L = mean_l (1 − cos(A_t_l, A_s_l))`. 800 steps
   on N_TRAIN = 200 (Q, A) pairs (4 epochs). AdamW, `mx.eval` +
   `mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)`
   functional gradients.
4. **Phase C — K1881 length reduction.** 50 held-out neutral prompts. Both
   adapter and base generate completions (max_tokens=256, temperature=0).
   Report `ρ = 1 − mean_tokens(adapter) / mean_tokens(base)`.
5. **Phase D — K1882 task accuracy.** 100 MMLU questions (seed=42 random
   sample). Adapter and base both answer with greedy decoding (max_tokens=64).
   Score against canonical MMLU answer. Report `Δ_accuracy =
   accuracy(base) − accuracy(adapter)` (one-sided; positive = degradation).
6. **Phase E — KC resolution.** Apply F#666 verdict matrix (§3.4); write
   results.json with kc dict + verdict + sub-type + super-family ledger.

## 7. Locked KCs — no edits after data collection

KCs K1881, K1882 pre-registered in DB verbatim. Any post-hoc relaxation
invalidates the run (verdict-consistency check #5). Concise system prompt
(π_Concise) locked as in Phase A (a single canonical concise prompt). MMLU
subset random seed locked at 42. Held-out neutral prompt set locked at 50
prompts. Max-tokens budgets locked: 256 for K1881 length eval, 64 for K1882
MMLU eval.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1 (axis is novel-cousin to politeness/formality, not redundant).**
  Behavior-axis sub-cluster currently has 2 instances (F#683 politeness,
  F#724 formality). Adding conciseness opens a 3rd behavior-axis cousin —
  this is the **sub-cluster standalone-memory promotion trigger**. Domain-axis
  sub-cluster closed at 5 (F#684/696/697/717/718). Conciseness is structurally
  distinct from register style (one can be formal-verbose or informal-
  concise).
- **A2 (dual-target / zero-proxy KC design — 2nd instance, 1st one-sided-
  safety sub-variant).** F#724 was 1st zero-proxy with two-sided orthogonality
  (K1880). K1882 one-sided degradation-only is a structurally distinct safety-
  target class, extending the zero-proxy design pattern.
- **A3 (concise teacher prompt is single-canonical).** π_Concise is a single
  fixed system prompt. Multi-prompt conciseness teacher (e.g., bullet-format
  + twitter-style + TL;DR formal) is out of scope; if conciseness varies
  meaningfully across sub-styles, file as a follow-up sweep.
- **A4 (K1882 power constraint).** MMLU n=100 has binomial CI ±5 pp at
  p≈0.5. The 3 pp one-sided KC is chosen specifically because at n=100,
  detecting a 3 pp drop at α=0.05 one-sided requires n ≈ 300; the 3 pp
  threshold is thus a **necessary** check — empirical drop > 3 pp at n=100
  is strong evidence of real degradation. `_impl` may scale to n=300 if
  K1882 lands borderline.
- **A5 (scope).** Researcher-hat single-iteration cap (30 min / 40 tool
  calls) means full Phase 0 + Phase A (teacher capture) + Phase B (training
  loop) + Phase C/D/E ~ 8–10 h pipeline is out of scope. PROVISIONAL with
  `_impl` follow-up is the right filing.
- **A6 (LORA_SCALE = 6.0 ≤ 8** per F#328/F#330).
- **A7 (KC-count scope).** Only K1881 + K1882 pre-registered. No cross-axis
  interference KCs (conciseness-vs-politeness/formality orthogonality), no
  information-density audit KC — those are sibling follow-ups, NOT retro-
  attached KCs.
- **A8 (hygiene-patch — F#702).** DB row shipped with 3 hygiene defects
  (success_criteria=[], references=[], platform=~). F#702 hygiene-patch
  PROVISIONAL is APPLICABLE. `mem-impossibility-f666pure-saturation-implies-
  f702-unavailable` does NOT fire — both KCs are targets (not F#666-pure).
  Hygiene corrections applied via DB update before `experiment complete`.
- **A9 (behavior-axis sub-cluster promotion — 3RD INSTANCE MILESTONE).**
  F#683 1st, F#724 2nd, THIS 3rd. Sub-cluster standalone-memory promotion
  threshold MET. Analyst hat owns the actual memory write; researcher filing
  flags the milestone.
- **A10 (10th Hedgehog-framework PROVISIONAL).** Hard-defer pile is 9
  designs / 0 measurements (post-F#724). This filing is 10th design / still
  0 measurements. 26B teacher cache remains standalone-prereq-task candidate
  blocking 10+ dependents including this one.
- **A11 (transitive blocker).** This adapter's `_impl` depends on Phase 0
  neutral-prompt curation (NEW set, not F#683 reuse — length-neutrality
  matters more than register-neutrality for this axis). It does NOT
  transitively depend on F#683 or F#724 `_impl`. Phase A teacher capture
  pipeline pattern reusable from F#683/F#724.
- **A12 (KC-design bifurcation rule axis-invariant).** Paired-target →
  PROVISIONAL; pure-proxy → KILL. Dual-target → PROVISIONAL (both two-sided
  and one-sided safety sub-variants, extending from F#724).
