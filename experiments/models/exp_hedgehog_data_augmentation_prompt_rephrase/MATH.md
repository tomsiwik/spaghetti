# MATH.md — exp_hedgehog_data_augmentation_prompt_rephrase

**Claim.** Rephrasing each Hedgehog training prompt 5× with temperature 1.2 — all
other Hedgehog components held fixed (politeness axis reuse from F#683, 26B Gemma 4
teacher with π_Polite in context, rank-8 LoRA on `(v_proj, o_proj)`, cos-sim per-
layer loss, 800 steps, same seed) — produces a student adapter whose (a) downstream
politeness-axis behavioral quality exceeds the non-augmented adapter's quality by
> 3 pp (K1877 target), and (b) per-layer cos-sim variance across training batches
is > 0.10 (K1878 proxy — training-stability signal). This is a DATA-AUGMENTATION
ABLATION on the Hedgehog training procedure — NEW 5th sub-type of the Hedgehog-
ablation super-family, cousin of loss-variant / layer-selection / hyperparameter
ablations. It tests whether prompt-rephrasing augmentation is a net-positive
regularizer or a net-negative noise injector on cos-sim distillation.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns,
  `mx.eval` discipline at step boundaries, `mx.clear_cache` between phases,
  `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval,
  bandwidth-aware kernels). Both MUST be invoked before the MLX training-loop code
  lands in the `_impl` follow-up. Hard gate per Finding #673 and 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed at
  run time.
- **Student model:** `mlx-community/gemma-4-e4b-it-4bit` (HF repo id).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` (sequential-phase
  eviction per F#673 on 48 GB M5 Pro).
- **Rephrasing model:** reuse teacher (26B Gemma 4) for prompt-rephrase generation
  at temperature 1.2. Do NOT introduce a third model class — rephrasing model drift
  would confound the ablation.
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Axis pinning.** Both variants train on the SAME axis (politeness — F#683
  precedent with the most mature teacher-capture pipeline among siblings). Only
  the training-data presentation differs (5× rephrased vs 1× original). All other
  hyperparameters identical; seed identical.
- **Scope-preservation (antipattern-t).** If either arm cannot land in a single
  iteration, file PROVISIONAL; do NOT substitute smaller teacher, different axis,
  or different step count in ONE arm — that breaks the A/B.

## 1. Failure mode

Primary degenerate behavior the ablation guards against: "Prompt rephrasing does
NOT increase effective training diversity — the 5× rephrases collapse to
semantically-equivalent strings at temperature 1.2 (Gemma 4 teacher is already
over-regularized), producing a training corpus with 5× redundancy instead of 5×
diversity. Under this degenerate regime, augmented training is strictly worse
than non-augmented because batch-level signal is diluted while compute is 5×.
K1877 FAIL + K1878 PASS (high variance with no gain) = augmentation is actively
harmful at this rephrase temperature."

Secondary failure: "Rephrasing drifts off-axis. At temperature 1.2, some
rephrases will invert politeness polarity (polite prompt → rude rephrase) or
shift the domain (politeness → formality). If drift-rate > 20%, the augmented
training signal becomes polluted with off-axis examples, and the adapter
partially learns a different behavior. Detection: per-rephrase judge scores
original-vs-rephrase semantic-equivalence before training; discard rephrases
below threshold. The _impl must implement and test this quality gate."

Tertiary failure: "Cos-sim variance (K1878) is a stability signal, not a quality
signal. High variance in the augmented arm would normally suggest unstable
training dynamics — but if K1877 also passes, the finding is 'augmentation
helps via regularization DESPITE high variance.' Variance and quality are
orthogonal here, not substitutable. K1878 is paired diagnostic for K1877 under
F#666; verdict is target-gated."

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim loss
  (baseline definition). Moudgil does not study data-augmentation variants.
- **Wei et al. 2024 "Prompt rephrasing for SFT" (arxiv:2404.06666-class ref):**
  prompt-rephrasing at temperature 1.0–1.5 provides +1–3 pp improvement on GSM8K
  / HumanEval-family benchmarks. Effect is rank-dependent and dataset-dependent.
- **Alpaca / Self-Instruct (Stanford, 2023):** self-rephrased prompts at
  temperature 1.0 generate effective SFT training diversity. Canonical data-
  augmentation-via-LLM precedent.
- **Finding #469 (this repo):** diversity-via-rephrase beat 1.3× compute gain on
  political-bias axis; did NOT beat on domain-of-code axis. Effect is **axis-
  dependent**. Politeness axis is a linguistic-register axis — closer to bias-axis
  precedent than to code-axis — so prior favors K1877 positive.
- **Sibling Hedgehog-axis PROVISIONAL precedents:** F#683, F#684, F#696, F#697,
  F#717, F#718 (axis-extension sub-type); F#719, F#720 (loss-variant-ablation
  sub-type); F#721 (layer-selection-ablation sub-type); F#722 (hyperparameter-
  ablation sub-type — temperature).
- **F#666 target-gating convention;** K1877 is target-paired with K1878 proxy;
  `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — **inapplicable
  here** (K1877 is a target KC, not F#666-pure).
- **Pierre F#627** (v_proj+o_proj LoRA sufficiency); **F#614/F#536** (thinking-mode
  load-bearing on Gemma 4); **F#328/F#330** (LORA_SCALE ≤ 8); **F#673**
  (mx.clear_cache between phases, MLX audit 2026-04-17).

## 3. Derivation sketch

1. *Existence.* Data augmentation via prompt rephrasing is a standard SFT
   technique. Gemma 4 at temperature 1.2 can produce stylistic variants of a
   given politeness-axis prompt; these variants share axis semantics but vary
   surface form. This is well-posed over the Hedgehog training-step budget.
2. *Effective-training-set size.* Non-augmented arm sees N_TRAIN = 200 unique
   (Q, A) pairs over 800 steps (4 epochs). Augmented arm sees N_TRAIN × 5 = 1000
   (Q_rephrased, A) pairs over 800 steps (0.8 epochs). Epoch count differs by
   5× — fewer repeats per pair, broader coverage. This IS the canonical
   regularization regime (Wei 2024).
3. *K1877 target-gated per F#666.* Behavioral quality (auto-judge on held-out
   axis prompts, F#683 rubric) is the independent target metric. Each variant
   evaluated independently on the SAME held-out slice; the 3 pp delta is the
   effect size. Behavioral quality is NOT defined by the augmentation regime —
   it's downstream task performance measured against external rubric — so the
   comparison is grounded, NOT tautological per §5.
4. *K1878 training-stability proxy.* Cos-sim variance across training batches
   (measure over the last 200 steps) tracks the training-dynamics instability
   induced by broader prompt diversity. High variance (> 0.10) without quality
   gain = augmentation is adding noise, not signal. High variance WITH quality
   gain = augmentation is regularizing via noise (a known regularizer regime).
5. *Bounds.* Behavioral quality ∈ [0, 100] on rescaled rubric; +3 pp JND from
   F#683 power calc. Cos-sim variance ∈ [0, 1] empirically over training
   batches; > 0.10 is "noticeable instability" threshold.

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1877 | mean behavioral-quality-judge score: (augmented adapter) − (non-augmented adapter) on 50 held-out paired prompts | Δ > +3 pp strictly (augmented beats non-augmented by > 3 pp) | target — paired with K1878 per F#666 |
| K1878 | variance of per-layer cos-sim across last 200 training batches in the augmented arm | variance > 0.10 strictly | proxy — training-stability diagnostic |

**F#666 target-gating.** Verdict matrix:
- **SUPPORTED** = K1877 PASS ∧ K1878 PASS (augmentation adds quality AND has
  noticeable stability cost — regularization-via-noise regime; finding: prompt
  rephrasing is net-positive with quantifiable variance cost to schedule around).
- **KILLED** = K1877 FAIL ∧ K1878 FAIL (augmentation provides no quality gain
  AND training is stable — finding: augmentation is a compute-no-op; skip it).
- **PROVISIONAL (target-FAIL + proxy-PASS)** = no quality gain but high variance —
  augmentation is strictly harmful; skip it and log the variance signature.
- **PROVISIONAL (target-PASS + proxy-FAIL)** = quality gain without variance —
  augmentation regularizes "for free"; preferred result; adopt as default.

**K1878 flavor disclosure.** Cos-sim variance is an INTRA-VARIANT stability
threshold, NOT an inter-variant delta. §5 tautological-inter-variant-delta does
NOT fire on K1878 (it's single-variant absolute threshold). K1877 is inter-
variant but grounded to external ground truth (held-out rubric judge), so §5
does NOT fire on K1877 either.

**F#666-pure disclosure.** K1878 is a proxy (training stability, not behavioral
quality). Without the K1877 target pairing, this would be F#666-pure-standalone
and subject to preempt-KILL. With K1877 paired, F#666 is satisfied; verdicts gate
on the target.

**Behavioral quality judge (K1877).** 0–100 rescaled auto-judge rubric on the
**politeness axis** (reusing F#683 rubric verbatim): (a) register appropriateness,
(b) indirection/hedging appropriate to context, (c) tonal fidelity. N = 50 held-
out axis prompts, blind-paired (augmented-adapter vs non-augmented-adapter)
presentation to judge to reduce absolute-score bias. Rubric reused verbatim
from F#683 — same axis, same rubric, same held-out set (A1).

## 5. Predicted measurements

- K1877: `Δ_behavioral = (augmented) − (non-augmented) ∈ [−1.0, +5.0] pp`; mean
  prediction +2.0 pp. The kill threshold is +3 pp, so the *expected* outcome is
  PROVISIONAL (target-FAIL — Δ < +3 pp) per the §3 derivation: politeness-axis
  rephrases at temperature 1.2 collapse partially to equivalent stylistic
  variants (Gemma 4 is over-regularized), yielding partial but not decisive
  improvement.
- K1878: `variance(augmented cos-sim over last 200 steps) ∈ [0.06, 0.18]`;
  mean prediction 0.11. The kill threshold is 0.10, so K1878 is expected PASS
  (variance > 0.10). Broader prompt diversity will increase batch-to-batch
  cos-sim variance as expected.

If K1877 and K1878 both PASS (augmented > +3 pp AND variance > 0.10), the
finding is: "Prompt rephrasing provides regularization-via-noise benefit on the
politeness axis — adopt as default with variance-aware LR scheduling." Most
positive outcome, low-to-medium prior.

If K1877 and K1878 both FAIL (no quality gain AND stable training), the
finding is: "Prompt rephrasing at temperature 1.2 collapses to semantically-
equivalent redundancy on the politeness axis — skip augmentation for cost
savings." Low prior.

If PROVISIONAL (target-FAIL + proxy-PASS, the mean-prediction case): "Augmented
training adds variance without quality — avoid." This is the most likely outcome.

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset + rephrasing.** Reuse F#683 politeness axis corpus
   (N_TRAIN = 200 base (Q, A) pairs). Generate 5 rephrases of each Q at
   temperature 1.2 using 26B Gemma 4 teacher (no π_Polite context during
   rephrasing — axis-neutral rephrasing). Held-out eval slice unchanged (n = 50,
   F#683 rubric). Store rephrase corpus to disk.
2. **Phase 0.5 — Rephrase QA gate.** Semantic-equivalence auto-judge on each
   (Q_original, Q_rephrase) pair; discard rephrases below threshold (e.g., ≥ 0.7
   semantic similarity). Record drift-rate; fail-abort if drift-rate > 20%
   (tertiary-failure gate from §1).
3. **Phase A — Teacher attention capture.** 26B Gemma 4 + π_Polite + Q in context.
   Capture `{layer_idx: attn_output}` for all 42 layers on BOTH corpora (original
   + augmented). Cos-sim distillation needs attn_output only (not attn_weights).
   Sequential-phase eviction per F#673; pre-compute offline.
4. **Phase B_base — Student training (non-augmented arm).** Rank-8 LoRA on
   `(v_proj, o_proj)` with per-layer cos-sim loss. 800 steps on N_TRAIN = 200
   base pairs (4 epochs), AdamW, `mx.eval + mx.clear_cache` between batches.
   `nn.value_and_grad(student, loss_fn)` functional gradients.
5. **Phase B_aug — Student training (augmented arm).** SAME protocol as B_base
   but trained on N_TRAIN × 5 = 1000 (Q_rephrased, A) pairs for 800 steps (0.8
   epochs). ALL other hyperparameters identical to B_base — same rank, targets,
   scale, steps, seed, optimizer state-init, seqlen. Only training-data
   presentation differs.
6. **Phase C — K1878 cos-sim variance.** During Phase B_aug, log per-layer
   cos-sim every step. Compute variance over last 200 steps (approx last 25% of
   training). Report `variance(augmented cos-sim)`.
7. **Phase D — K1877 behavioral quality delta.** Blind-paired 50-prompt auto-
   judge on held-out F#683 politeness-axis set. Swap order 50/50 for position-
   bias control. Report `Δ = score(augmented) − score(non-augmented)` on 0–100
   rescaled rubric.

## 7. Locked KCs — no edits after data collection

KCs K1877, K1878 pre-registered in DB verbatim. Any post-hoc relaxation
invalidates the run (verdict-consistency check #5). Rephrase temperature locked
at 1.2. Rephrase depth locked at 5×. Rephrase model locked at 26B Gemma 4
teacher (no π_Polite context — axis-neutral generation). Any deviation
invalidates the A/B.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1 (axis reuse).** Politeness axis has the most mature teacher-capture
  pipeline among Hedgehog siblings (F#683 PROVISIONAL, rubric published).
  Reusing it minimizes variance from axis-novelty and isolates the
  augmentation-effect signal. If another axis lands an _impl first, the
  ablation can be re-run on that axis as a robustness check — not a blocker.
- **A2 (rephrasing model = teacher).** Uses 26B Gemma 4 as the rephraser to
  avoid adding a third model class. Rephrasing does NOT use π_Polite context —
  axis-neutral rephrase (preserves Q's original register; does not re-polish).
- **A3 (5× rephrase depth chosen).** Wei 2024 finds 3–7× rephrase depth optimal
  for SFT. 5× is median; allows later sweep (3× / 7×) as follow-up if K1877
  lands ambiguous.
- **A4 (paired-judge blind presentation has adequate power for Δ ≥ +3 pp).**
  50 held-out pairs × 2 conditions × rubric MDE ~ +3 pp at α=0.05 — matches F#683
  rubric power calculation.
- **A5 (scope).** Researcher-hat single-iteration cap (30 min / 40 tool calls)
  means full Phase 0 (rephrase generation) + Phase 0.5 (QA gate) + Phase A
  (teacher capture) + Phase B_base + Phase B_aug + Phase C + Phase D ~ 12–15 h
  two-adapter pipeline is out of scope. PROVISIONAL with `_impl` follow-up is
  the right filing.
- **A6 (LORA_SCALE = 6.0 ≤ 8** per F#328/F#330).
- **A7 (KC-count scope).** Only K1877 + K1878 pre-registered. No cross-axis,
  cross-rank, or rephrase-temperature-sweep KCs — those can be sibling follow-
  ups, NOT retro-attached KCs.
- **A8 (drift gate locked at ≥ 0.7 semantic similarity, drift-rate ≤ 20%).**
  Below threshold ⇒ rephrase discarded. Over 20% of rephrases failing ⇒
  fail-abort (tertiary-failure from §1).
- **A9 (hygiene-patch — F#702).** DB row shipped with 3 hygiene defects
  (success_criteria=[], platform=~, references=[]). F#702 hygiene-patch
  PROVISIONAL is applicable here. `mem-impossibility-f666pure-saturation-
  implies-f702-unavailable` does NOT fire — K1877 is a target KC (not F#666-
  pure). Hygiene corrections applied via DB update before `experiment complete`.
- **A10 (data-augmentation-ablation is NEW 5th sub-type).** Super-family ledger:
  axis-extension (F#683/684/696/697/717/718, 6 instances — closed after F#718);
  loss-variant-ablation (F#719/720, 2 instances); layer-selection-ablation
  (F#721, 1 instance); hyperparameter-ablation (F#722, 1 instance temperature);
  **data-augmentation-ablation (this, 1 instance, NEW)**. Super-family now 5
  sub-types / 11+1 instances. KC-design bifurcation (paired-target → PROVISIONAL;
  pure-proxy → KILL) axis-invariant across super-family; this filing is paired-
  target → PROVISIONAL.
- **A11 (8th Hedgehog-framework PROVISIONAL).** Hard-defer pile was 7 designs /
  0 measurements; this is 8th design / still 0 measurements. 26B teacher cache
  remains standalone-prereq-task candidate blocking 8+ dependents including
  this one. No measurement without _impl + teacher cache + F#683 corpus.
- **A12 (transitive blocker).** This ablation's `_impl` depends on F#683 `_impl`
  having landed (politeness corpus + rubric). If F#683 stalls, this cascades.
  Re-scope to whichever Hedgehog axis `_impl` lands first if F#683 blocks
  indefinitely.
