# MATH.md — exp_hedgehog_behavior_adapter_formality

**Claim.** Per-layer cosine-similarity distillation (Hedgehog) of the 26B Gemma 4
teacher under a formal-register system prompt (π_Formal) into a rank-8 LoRA on
`(v_proj, o_proj)` of Gemma 4 E4B trains a **formality adapter** that (a) raises
auto-judge formality by ≥ +10 pp on neutral held-out prompts (K1879 target —
behavioral acquisition) AND (b) preserves factual accuracy within ±2 pp (K1880
target — style/substance orthogonality, anti-leak safety). This is the **2nd
behavior-axis instance** in the Hedgehog-framework (cousin of F#683 politeness),
and the **FIRST dual-target / zero-proxy KC design** in the super-family — every
prior Hedgehog-framework PROVISIONAL paired a target with a proxy, but K1879 and
K1880 are BOTH target KCs. K1880 is a NEW KC class — *style/substance
orthogonality* — absent from F#683, F#684, F#696, F#697, F#717, F#718, F#719,
F#720, F#721, F#722, F#723.

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
- **Axis pinning.** Formality axis is independent of politeness axis (F#683).
  Formal-register prompts (academic, technical, professional) vs informal-
  register prompts (casual, conversational, slang) — an orthogonal style axis.
- **Scope-preservation (antipattern-t).** PROVISIONAL filing: do NOT substitute
  smaller teacher, different axis, or different step count. Full Phase A/B/C/D
  pipeline lands in `_impl`.

## 1. Failure mode

Primary degenerate behavior the experiment guards against: **"Style leaks into
substance."** A formality adapter trained via per-layer cos-sim distillation on
a formal-teacher capture may shift attention routing in ways that perturb
*content selection*, not just *register selection*. Under this failure regime,
the adapter raises formality auto-judge (K1879 PASS) but degrades factual
accuracy on knowledge-bench tasks like MMLU or TriviaQA by > 2 pp (K1880 FAIL).
The most insidious instantiation: formal register correlates statistically with
hedging, qualification, and academic-style citations; the adapter learns to
*qualify* answers ("It is generally believed that…", "Some sources suggest…")
rather than to *answer* them, which lowers binary-correctness rates while
sounding more formal.

Secondary failure: **"Formality is null."** Cos-sim distillation captures
attention-output structure but not register-specific routing. If formal-vs-
informal teacher capture differs only in token-level lexical choice (not in
attention-routing patterns), the rank-8 LoRA cannot learn the distinction —
both K1879 and K1880 fail (no formality gain AND no factual change). Under this
mode, formality is a *lexical* phenomenon not a *routing* phenomenon, and
Hedgehog cos-sim distillation is the wrong inductive bias. Detection: K1879 < 5
pp delta paired with K1880 < 0.5 pp factual change ⇒ adapter is learning
nothing; rank or training-step budget is misspecified.

Tertiary failure: **"Formality conflates with tone of voice."** If the formal
teacher prompt (π_Formal) implicitly encodes politeness (formal speech is often
polite, e.g., "I would respectfully suggest…"), the adapter learns a polite-
formal mixture — and the auto-judge for formality (K1879) actually scores
politeness-correlated features. Detection: cross-axis interference test in
`_impl` — apply formality adapter, judge politeness; if politeness-judge ALSO
moves > +5 pp, the formality-vs-politeness axes are not orthogonal at this
training scale, and the F#683-style behavior axis is broader than presumed.

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim loss
  (baseline definition).
- **Zhang et al. 2402.04347:** cosine loss recovers 99% of softmax attention
  behavior with MLP feature maps. Evidence for cos-sim as a dense training
  signal.
- **Pierre F#627:** rank-6 LoRA on `v_proj+o_proj` of Gemma 4 E4B captures
  domain specialization end-to-end. Parameterization is sufficient at rank 8.
- **Sibling Hedgehog-axis PROVISIONAL precedents:** F#683 politeness (behavior
  axis, 1st instance — closest cousin); F#684 procedural-refactor, F#696 JS,
  F#697 unspecified-domain, F#717 Rust, F#718 SQL (domain axes 1-5 — closed
  after F#718); F#719 KL-loss + F#720 MSE-loss (loss-variant); F#721 layer-
  selection-top6; F#722 temperature-sweep; F#723 prompt-rephrase-augmentation.
- **F#666 target-gating convention.** K1879 + K1880 are BOTH target KCs (no
  proxy). Verdict matrix gates on both targets. **Pure-proxy preempt-KILL does
  NOT apply** (no proxy KCs); **§5 tautological-inter-variant-delta does NOT
  apply** (KCs grounded to external auto-judge + factual benchmark, not inter-
  variant deltas).
- **`mem-impossibility-f666pure-saturation-implies-f702-unavailable` —
  inapplicable** (both KCs target — F#702 hygiene-patch path is AVAILABLE).
- **F#614/F#536:** thinking-mode load-bearing on Gemma 4 (`enable_thinking=True`).
- **F#328/F#330:** LORA_SCALE ≤ 8.
- **F#673 + 2026-04-17 audit:** `mx.clear_cache` between phases; sequential-
  phase eviction for 26B residency on 48 GB.

## 3. Derivation sketch

1. *Existence.* Formality is a register dimension distinct from semantic
   content. Per Zhang 2024, attention output is a 99%-faithful proxy for the
   model's surface-form distribution. A formal-register teacher capture varies
   π_Formal vs π_Null at constant Q content; the per-layer attention-output
   delta encodes register-specific routing. Rank-8 LoRA on `v_proj+o_proj` has
   sufficient degrees of freedom to fit this register delta (Pierre F#627
   precedent at rank 6 fit domain shifts).
2. *K1879 target-gated per F#666.* Behavioral formality is the independent
   target metric. Auto-judge on a 50-prompt held-out neutral set scores
   register on a 0–100 rubric (academic → casual). Δ = score(adapter) −
   score(base) on neutral prompts. K1879 KILL condition: Δ < +10 pp (no
   meaningful behavioral acquisition).
3. *K1880 target-gated per F#666 — NEW KC class.* Factual accuracy on a
   knowledge benchmark (MMLU 100-question subset OR TriviaQA 100-question
   subset) under the adapter vs base. K1880 KILL condition: |accuracy(adapter)
   − accuracy(base)| > 2 pp. **Two-sided** — both degradation (style-leak into
   substance) AND improvement (the adapter is doing more than register-
   shifting; e.g., hedging changes answer distribution) violate K1880.
4. *Verdict matrix (DUAL TARGET, no proxy).*
   - **SUPPORTED** = K1879 PASS (Δ ≥ +10 pp formality) ∧ K1880 PASS
     (|Δ_factual| ≤ 2 pp). Adapter acquired formality without disturbing
     substance — the framework generalizes to a 2nd behavior axis with safety.
   - **KILLED** = K1879 FAIL ∧ K1880 FAIL. No formality gain AND substance
     drift — adapter learned an unrelated routing perturbation; framework does
     NOT generalize cleanly to formality.
   - **PROVISIONAL (K1879 PASS + K1880 FAIL)** = formality acquired but at the
     cost of substance — finding: cos-sim distillation cannot orthogonalize
     style from substance at this rank/scale; framework needs SIGReg or
     orthogonality-constrained loss.
   - **PROVISIONAL (K1879 FAIL + K1880 PASS)** = no formality gain but no
     substance drift — adapter is null, training is benign; finding:
     formality-axis is sub-threshold for cos-sim distillation; needs different
     loss (KL? CE-on-style-tokens?) or larger rank.
5. *Bounds.* Formality auto-judge ∈ [0, 100]; +10 pp JND from F#683 power calc
   (50-prompt paired-judge MDE ~+8 pp at α=0.05). Factual accuracy ∈ [0, 100];
   ±2 pp threshold based on MMLU subset reliability (binomial CI at n=100,
   p≈0.5 ⇒ ±5 pp; ±2 pp requires n≈400 — see A4 power note).

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1879 | Δ = formality-judge(adapter) − formality-judge(base) on 50 held-out neutral prompts (0–100 rubric) | Δ < +10 pp strictly | target — behavioral acquisition |
| K1880 | |accuracy(adapter) − accuracy(base)| on 100-question MMLU subset | > 2 pp strictly (two-sided) | target — style/substance orthogonality (NEW KC class) |

**F#666 target-gating.** Verdict matrix per §3.4 — DUAL-TARGET design (no proxy
KC). Pure-proxy preempt-KILL does NOT apply. §5 tautological-inter-variant-
delta does NOT apply (K1879 grounded to external auto-judge baseline; K1880
grounded to MMLU canonical answers — both external ground truth, not inter-
variant deltas).

**KC-design classification.** Hedgehog-framework KC-design taxonomy (in this
super-family):
- **target+proxy paired** (canonical): F#683 (K1 cos-sim proxy + K2/K3/K4
  targets), F#719/720/721/722 (cos-sim proxy + behavioral target).
- **dual-target / zero-proxy** (NEW, this experiment): K1879 + K1880 both
  target. No structural proxy.
- **target+proxy intra-stability** (F#723): cos-sim variance proxy + behavioral
  target.

This is the **1st zero-proxy KC design** in the super-family. Justified by
K1880 being a SAFETY target (orthogonality), which by construction belongs in
the "target" column (we want substance NOT to drift), not in the "proxy"
column.

**Behavioral formality auto-judge (K1879).** 0–100 rubric: (a) lexical register
(formal vocabulary score), (b) syntactic complexity (subordinate-clause
density), (c) hedging/qualification (academic citation patterns), (d)
contraction-rate (informal markers — inverse weighting). Held-out 50-prompt
neutral set (NOT politeness-axis prompts; NEW set generated for this axis).
Blind-paired adapter-vs-base presentation; swap order 50/50 for position-bias
control.

**MMLU subset (K1880).** 100 random questions sampled from MMLU dev split,
seed=42. Categories balanced across STEM, humanities, social-science. Adapter
applied at inference; greedy decoding, max_tokens=64. Same 100 questions for
both adapter and base evaluation.

## 5. Predicted measurements

- K1879: `Δ_formality ∈ [+5, +18] pp`; mean prediction +12 pp. **Expected
  PASS** based on F#683 precedent (politeness adapter achieved +20-35 pp
  predicted; formality is closer to politeness than to domain — register-axis
  Hedgehog adapters work).
- K1880: `|Δ_factual| ∈ [0, 4] pp`; mean prediction 1.8 pp degradation.
  **Expected PASS but borderline.** The hedging-style failure mode (§1
  primary) is a known concern; if hedging shows up at scale, factual accuracy
  drift can exceed 2 pp.

**Most likely outcome:** SUPPORTED (both PASS) at probability ~50%; PROVISIONAL
(K1879 PASS + K1880 FAIL — hedging leak) at ~30%; PROVISIONAL (K1879 FAIL +
K1880 PASS — sub-threshold) at ~15%; KILLED at ~5%.

If SUPPORTED, finding: "Hedgehog cos-sim distillation generalizes to a 2nd
behavior axis (formality) with style/substance orthogonality at rank 8.
Behavior-axis sub-cluster of Hedgehog-framework opens beyond politeness."

If PROVISIONAL (K1879 PASS + K1880 FAIL), finding: "Formality acquisition
leaks into factual hedging; orthogonality requires SIGReg-style explicit
constraint or content-vs-style loss decomposition. Cos-sim alone is
insufficient for safety on register axes."

If PROVISIONAL (K1879 FAIL + K1880 PASS), finding: "Formality is sub-threshold
for cos-sim distillation at rank 8 / 800 steps; needs alternative loss or
more capacity. Behavior-axis sub-cluster does not generalize trivially."

If KILLED, finding: "Formality is not a routing phenomenon — it's a token-
level lexical phenomenon. Hedgehog cos-sim is the wrong inductive bias;
behavior-axis sub-cluster requires non-attention-output methods for formality."

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Neutral prompt curation.** Generate 250 neutral prompts (200
   train + 50 held-out) covering knowledge questions, descriptive tasks,
   problem-solving — NEUTRAL register baseline. Manually verify register
   neutrality (no implicit formal/informal cue). Reuse pipeline pattern from
   F#683 Phase 0 (politeness neutral curation), but NEW prompt content.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + π_Formal + Q in
   context (formal-register system prompt: "You are a formal-register
   academic assistant. Reply in formal English with academic tone, no
   contractions, full sentences with subordinate clauses where appropriate.").
   Capture `{layer_idx: attn_output}` for all 42 layers per (Q, A) pair.
   Sequential-phase eviction per F#673; pre-compute offline.
3. **Phase B — Student training.** Rank-8 LoRA on `(v_proj, o_proj)` with
   per-layer cos-sim loss: `L = mean_l (1 − cos(A_t_l, A_s_l))`. 800 steps on
   N_TRAIN = 200 (Q, A) pairs (4 epochs). AdamW, `mx.eval` + `mx.clear_cache`
   between batches. `nn.value_and_grad(student, loss_fn)` functional
   gradients.
4. **Phase C — K1879 formality auto-judge.** 50 held-out neutral prompts.
   Both adapter and base generate completions (max_tokens=128, temperature=0).
   Blind-paired auto-judge (Claude 3.7 or GPT-4) scores each pair on 0–100
   formality rubric. Position swap 50/50. Report `Δ = mean(adapter) −
   mean(base)`.
5. **Phase D — K1880 factual accuracy.** 100 MMLU questions (seed=42 random
   sample). Adapter and base both answer with greedy decoding. Score against
   canonical MMLU answer. Report `|Δ_factual| = |acc(adapter) − acc(base)|`.
6. **Phase E — KC resolution.** Apply F#666 verdict matrix (§3.4); write
   results.json with kc dict + verdict + sub-type + super-family ledger.

## 7. Locked KCs — no edits after data collection

KCs K1879, K1880 pre-registered in DB verbatim. Any post-hoc relaxation
invalidates the run (verdict-consistency check #5). Formal-register
prompt (π_Formal) locked as in Phase A (a single canonical formal system
prompt). MMLU subset random seed locked at 42. Held-out neutral prompt set
locked at 50 prompts.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1 (axis is novel-cousin to politeness, not redundant axis-extension).**
  Behavior-axis sub-cluster of Hedgehog-framework currently has 1 instance
  (F#683 politeness). Adding formality opens a 2nd behavior-axis cousin.
  Domain-axis sub-cluster is closed at 5 (F#684/696/697/717/718). The
  axis-extension super-saturation note (post-F#718) does NOT block
  behavior-axis additions because they were under-represented (1 instance vs 5
  domain instances). Formality is a clearly distinct register dimension from
  politeness (formality ⊥ politeness on canonical register theory; one can be
  formal-impolite or informal-polite).
- **A2 (dual-target / zero-proxy KC design is novel within the super-family).**
  Justified by K1880 being a SAFETY target (orthogonality / non-interference)
  which belongs in the target column by construction. No proxy KC was needed
  because the design tests two distinct target properties (acquisition + non-
  interference). If reviewer pushes for a proxy KC, propose adding cos-sim
  proxy K1879p as a follow-up (post-PROVISIONAL).
- **A3 (formal teacher prompt is single-canonical).** π_Formal is a single
  fixed system prompt. Multi-prompt formality teacher (e.g., academic + legal
  + medical formal styles) is out of scope; if formality varies meaningfully
  across formal sub-styles, file as a follow-up sweep.
- **A4 (K1880 power constraint).** MMLU n=100 has binomial CI ±5 pp at
  p≈0.5, so detecting 2 pp drift at α=0.05 requires n≈400. The ±2 pp KC is
  thus a **necessary** (not sufficient) check — empirical |Δ_factual| ≤ 2 pp
  with n=100 is consistent with true drift up to ~5 pp, but |Δ_factual| > 2
  pp at n=100 is strong evidence of a real degradation. The asymmetric power
  is acceptable for an initial PROVISIONAL design-lock; `_impl` may scale to
  n=400 if K1880 lands borderline.
- **A5 (scope).** Researcher-hat single-iteration cap (30 min / 40 tool
  calls) means full Phase 0 (curation) + Phase A (teacher capture) + Phase B
  (training loop) + Phase C/D/E ~ 8–10 h pipeline is out of scope.
  PROVISIONAL with `_impl` follow-up is the right filing.
- **A6 (LORA_SCALE = 6.0 ≤ 8** per F#328/F#330).
- **A7 (KC-count scope).** Only K1879 + K1880 pre-registered. No cross-axis
  interference KCs (formality-vs-politeness orthogonality), no rank-sweep
  KCs, no temperature-ablation KCs — those are sibling follow-ups, NOT retro-
  attached KCs. Cross-axis interference filed separately as
  `exp_hedgehog_cross_axis_interference` (already in DB, P=2 open).
- **A8 (hygiene-patch — F#702).** DB row shipped with 2 hygiene defects
  (success_criteria=[], platform=~, references=[]; "platform: ~" counts).
  F#702 hygiene-patch PROVISIONAL is APPLICABLE here. `mem-impossibility-
  f666pure-saturation-implies-f702-unavailable` does NOT fire — both KCs are
  targets (not F#666-pure). Hygiene corrections applied via DB update before
  `experiment complete`.
- **A9 (behavior-axis sub-cluster opening).** F#683 was 1st behavior axis;
  this is 2nd. Sub-cluster promotion to standalone memory triggers at 3rd
  behavior-axis instance per analyst's 3-instance threshold precedent.
- **A10 (9th Hedgehog-framework PROVISIONAL).** Hard-defer pile is 8 designs
  / 0 measurements (post-F#723). This filing is 9th design / still 0
  measurements. 26B teacher cache remains standalone-prereq-task candidate
  blocking 9+ dependents including this one. No measurement without _impl +
  teacher cache + curated neutral prompt set.
- **A11 (transitive blocker).** This adapter's `_impl` depends on Phase 0
  neutral-prompt curation (NEW set, not F#683 reuse — different axis requires
  different neutral baseline). It does NOT transitively depend on F#683
  `_impl`. Phase A teacher capture pipeline pattern reusable from F#683.
- **A12 (KC-design bifurcation rule axis-invariant).** Paired-target →
  PROVISIONAL; pure-proxy → KILL. Dual-target also → PROVISIONAL (extends
  the rule to zero-proxy designs, which are more conservative than target+
  proxy because they lack the structural proxy short-circuit).
