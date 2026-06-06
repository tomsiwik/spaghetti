# MATH.md — exp_hedgehog_loss_variant_kl_div

**Claim.** Replacing the per-layer cos-sim distillation loss with a KL-divergence loss
on attention-map distributions — all other Hedgehog components held fixed (teacher
attention-output capture, student rank-8 LoRA on `(v_proj, o_proj)`, same axis, same
training schedule) — produces a student adapter whose (a) student-to-teacher per-layer
cos-sim lags cos-loss adapter's cos-sim (K1870 proxy), and (b) downstream behavioral
quality on the Hedgehog axis trails cos-loss adapter's behavioral quality by > 3 pp
(K1871 target). This is a LOSS-VARIANT ABLATION (not an axis-extension) on the
Hedgehog training procedure; its purpose is to test whether cos-sim is **load-bearing**
or whether any attention-divergence surrogate works.

---

## 0. Platform skills + versions (PLAN.md §1011/1012)

- **Skills required before coding:** `/mlx-dev` (array/nn/training patterns, `mx.eval`
  discipline at step boundaries, `mx.clear_cache` between phases,
  `nn.value_and_grad` functional gradients) + `/fast-mlx` (compile, lazy eval,
  bandwidth-aware kernels). Both MUST be invoked before the MLX training-loop code
  lands in the `_impl` follow-up. Hard gate per Finding #673 and the 2026-04-17 audit.
- **mlx-lm version pin:** record `results.json["mlx_lm_version"]` as installed at run
  time.
- **Base model (student):** `mlx-community/gemma-4-e4b-it-4bit` (exact HF repo id).
- **Teacher model:** `mlx-community/gemma-4-26b-a4b-it-4bit` (larger variant;
  sequential-phase eviction per F#673 on 48 GB M5 Pro).
- **Adapter targets:** `v_proj + o_proj` (Pierre F#627).
- **LoRA scale:** 6.0 (≤ 8 per F#328/F#330).
- **Axis pinning.** Both variants trained on the SAME axis (politeness — F#683
  precedent has the most mature teacher-capture pipeline among Hedgehog siblings). All
  other hyperparameters held identical across the two arms — the ONLY difference is
  the loss function. Anything else changed silently invalidates the ablation.
- **Scope-preservation (antipattern-t).** If either arm cannot land in a single
  iteration, file PROVISIONAL; do NOT substitute smaller rank, smaller teacher,
  different axis, or different step count in ONE arm — that breaks the A/B.

## 1. Failure mode

Primary degenerate behavior the ablation guards against: "Cos-sim is **not** load-
bearing — any attention-divergence surrogate recovers the same behavioral quality.
If K1870 FAILS-as-written (KL-div achieves cos-sim ≥ 0.70, nearly matching cos-loss)
AND K1871 FAILS-as-written (KL-div behavioral quality is NOT > 3 pp worse), the
finding is **loss choice is NOT load-bearing** — the Hedgehog framework generalizes
beyond cos-sim to other f-divergences. This is a *negative* kill (the null hypothesis
holds — no loss-choice effect), which is a finding about the Hedgehog framework's
robustness, not a bug."

Conversely, if K1870 KILL ∧ K1871 KILL both fire (KL-div cos-sim < 0.70 AND
behavioral quality > 3 pp worse), then cos-sim IS load-bearing — switching losses
degrades both proxy and target. Default to cos-sim.

Secondary failure: "KL-divergence on attention-map distributions requires row-
normalization after softmax — numerical instability on near-zero rows (padding,
attention-sink head) produces NaN gradients. Fixed by adding small ε to the post-
softmax distributions before KL. If ε is too large, KL collapses to uniform — if too
small, NaNs. The _impl must implement and test this."

Tertiary failure: "KL-divergence direction mixing — KL(teacher || student) vs
KL(student || teacher) vs symmetric-KL(Jensen-Shannon) produce different gradients.
The KC text doesn't specify; the _impl must fix to KL(teacher || student) (forward-
KL, mode-seeking, analogous to teacher-forcing in SFT) and document this as a locked
choice."

## 2. Cited prior math / findings

- **Moudgil arxiv:2604.14191 §3.1 eq. 6:** Hedgehog per-layer cos-sim. Moudgil's
  ablation of alternative attention-matching losses is partial; the paper primarily
  defends cos-sim via empirical performance on a different benchmark family.
- **Zhang 2402.04347:** cosine loss recovers 99% attention behavior with a small
  student. KL-divergence variant not reported in Zhang; a direct gap.
- **Hinton 2503.02531 (knowledge distillation):** canonical KL-div-on-logits
  distillation. Analog to attention-KL here but at logit level; mechanism transfers.
- **Sibling Hedgehog-axis precedents:** `exp_hedgehog_behavior_adapter_politeness`
  (F#683 PROVISIONAL), `exp_hedgehog_procedural_adapter_refactor` (F#684
  PROVISIONAL), `exp_hedgehog_domain_adapter_js` (F#696 PROVISIONAL),
  `exp_hedgehog_adapter_python_domain` (F#697 PROVISIONAL),
  `exp_hedgehog_adapter_rust_domain` (F#717 PROVISIONAL),
  `exp_hedgehog_adapter_sql_domain` (F#718 PROVISIONAL).
- **F#702 hygiene-patch PROVISIONAL**; **F#666 target-gating convention**;
  `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — **inapplicable
  here** (K1871 is a target KC, not F#666-pure).
- **Pierre F#627** (v_proj+o_proj LoRA sufficiency); **F#614/F#536** (thinking-mode
  load-bearing on Gemma 4); **F#328/F#330** (LORA_SCALE ≤ 8); **F#673**
  (mx.clear_cache between phases).

## 3. Derivation sketch

1. *Existence.* Both cos-sim and KL-div are bounded differentiable functions of
   student/teacher attention state. At rank-8 on `v_proj+o_proj`, the adapter DOF
   suffice to minimize either loss to a local optimum. The A/B comparison is well-
   posed.
2. *Cos-sim is the explicit training target for cos-loss.* Trivially,
   cos-loss trained adapter achieves higher student-to-teacher cos-sim than KL-
   trained adapter, because cos-loss directly optimizes cos-sim while KL-div does
   not. This makes K1870 the **tautological-for-cos-loss** proxy — it informs about
   the KL arm's fidelity *on cos-sim specifically*, not about whose adapter is "better"
   in any absolute sense.
3. *K1871 target-gated per F#666.* Behavioral quality (auto-judge on held-out axis
   prompts) is the independent target metric. Each variant evaluated independently;
   the delta is a valid A/B effect. Behavioral quality is NOT defined by either loss
   function — it's downstream task performance — so the comparison is NOT tautological.
4. *Bounds.* KL(teacher || student) ≥ 0 with equality iff distributions match.
   Cos-sim ∈ [−1, 1] with equality to 1 iff directions match (magnitude invariant).
   The two losses weight attention-distribution mismatches differently: KL penalizes
   probability-mass errors (high where teacher has high mass); cos-sim penalizes
   direction errors (magnitude-invariant). Under the hypothesis that Hedgehog learns
   via *direction* of attention (Moudgil §3.1 claim), cos-sim should dominate KL on
   behavioral quality.

## 4. Kill-criterion map

| KC | Measured quantity | Kill condition (KILL if TRUE) | Type |
|---|---|---|---|
| K1870 | student-to-teacher per-layer cos-sim for KL-div arm vs cos-loss arm | cos-sim(KL) < 0.70 AND cos-sim(cos-loss) > 0.80 | proxy — **tautological-for-cos-loss** by construction; see §5 of `mem-antipattern-tautological-inter-variant-delta` |
| K1871 | mean behavioral-quality-judge score: (cos-loss adapter) − (KL-div adapter) | Δ > +3 pp strictly | target (pair K1870 per F#666) |

**F#666 target-gating.** Verdict matrix:
- **SUPPORTED** = K1870 PASS ∧ K1871 PASS (cos-sim IS load-bearing — both proxy
  tautology confirms and target effect size is > 3 pp).
- **KILLED** = K1870 FAIL ∧ K1871 FAIL (cos-sim is NOT load-bearing — null hypothesis
  holds; switch defaults to whichever is cheaper/simpler).
- **PROVISIONAL (proxy-PASS + target-FAIL)** = K1870 confirms cos-loss wins on cos-sim
  (trivially true; tautological) BUT behavioral Δ < +3 pp — finding: **cos-sim is a
  tautological proxy but NOT a behavioral discriminator**; the Hedgehog framework is
  loss-robust on behavioral outcomes. This is the most informative null.
- **PROVISIONAL (proxy-FAIL + target-PASS)** = KL achieves cos-sim ≥ 0.70 AND
  behavioral Δ > +3 pp — surprising finding: KL is competitive on cos-sim too AND
  cos-loss still wins behaviorally. Would suggest a third mechanism. Low-prior outcome.

**K1870 §5-awareness.** K1870 is tautological-for-cos-loss by construction (cos-loss
explicitly optimizes cos-sim). This is *disclosed* in the KC text — the experiment
design acknowledges the asymmetry and uses K1870 only as a "did training converge"
signal for the KL arm, NOT as a claim of cos-loss superiority. Per F#666 doctrine,
the target K1871 is what gates verdicts; K1870 is retained as a diagnostic.

**Behavioral quality judge (K1871).** 0–10 auto-judge rubric on the **politeness
axis** (reusing F#683 rubric): (a) register appropriateness (formal vs informal
match to prompt context), (b) indirection/hedging appropriate to context, (c)
tonal fidelity. N = 50 held-out axis prompts, blind-paired (cos-loss-adapter vs KL-
adapter) presentation to judge to reduce absolute-score bias. Rubric reused verbatim
from F#683 — same axis, same rubric, same held-out set (assumption A1 below
validates this reuse).

## 5. Predicted measurements

- K1870: `cos-sim(cos-loss adapter) ∈ [0.82, 0.88]`; `cos-sim(KL adapter) ∈ [0.60, 0.75]`
  — cos-loss expected PASS (by construction); KL cos-sim straddles the 0.70 threshold.
- K1871: `Δ_behavioral_quality = (cos-loss) − (KL) ∈ [+0.5, +4.0] pp`; mean prediction
  +1.5 pp. The kill threshold is +3 pp, so the *expected* outcome is K1871 FAIL
  (behavioral delta < +3 pp) — the null hypothesis (no meaningful behavioral effect)
  is favored by the derivation in §3.2 (KL and cos-sim both recover attention
  routing; behavioral downstream signal should be insensitive to which is used).

If K1870 and K1871 both PASS (cos-sim is load-bearing, Δ > +3 pp), the finding is:
"Moudgil's choice of cos-sim matters — switching to KL-div degrades behavioral
quality beyond the 3 pp threshold at rank-8 scale." Surprising but possible.

If K1870 and K1871 both FAIL (null hypothesis holds), the finding is: "Cos-sim is
NOT load-bearing on behavioral quality; KL-divergence is an equivalent alternative.
The Hedgehog framework is loss-agnostic over this family of attention-matching
surrogates." Useful negative result that simplifies future work (either loss is OK).

If PROVISIONAL (proxy-PASS + target-FAIL, the mean-prediction case): "Cos-loss wins on
the cos-sim metric (trivially, by training objective) but NOT on behavioral quality."
This is the most-likely outcome per §3 derivation.

## 6. Experimental protocol (locked before implementation)

1. **Phase 0 — Dataset.** Reuse F#683 politeness axis corpus (teacher-with-context
   π_Polite prompts + held-out axis eval set, n=50 blind-paired held-out for K1871).
   No new dataset curation needed — axis-locking to politeness is the variance-
   reduction choice for the ablation.
2. **Phase A — Teacher attention capture.** 26B Gemma 4 + `π_Polite` + Q in context.
   Capture `{layer_idx: (attn_output, attn_weights)}` for all 42 layers. Needed for
   BOTH loss variants — cos-sim uses `attn_output`, KL-div uses `attn_weights`
   (post-softmax distributions). Sequential-phase eviction on 48 GB M5 Pro per
   F#673; pre-compute offline and stream from disk during student training.
3. **Phase B_cos — Student training (cos-loss arm).** Rank-8 LoRA on
   `(v_proj, o_proj)` with per-layer cos-sim loss on attention outputs:
   `L_cos = mean_l (1 − cos(A_teacher_l, A_student_l))`. 800 steps, AdamW,
   `mx.eval + mx.clear_cache` between batches. `nn.value_and_grad(student, loss_fn)`
   functional gradients — no Torch-style `.backward()`.
4. **Phase B_kl — Student training (KL-div arm).** Same protocol as Phase B_cos but
   loss is per-layer forward-KL on post-softmax attention distributions:
   `L_kl = mean_l KL(softmax(Q_t K_t^T / √d + ε) || softmax(Q_s K_s^T / √d + ε))`.
   ε = 1e−6 (numerical floor; prevents NaN on padded rows). ALL other hyperparameters
   identical to B_cos — same rank, same targets, same scale, same steps, same seed,
   same batch, same seqlen, same optimizer state-init. Only loss function differs.
5. **Phase C — K1870 student-to-teacher cos-sim.** Both adapters evaluated on the
   same held-out axis eval slice. Compute per-layer cos-sim and aggregate (mean over
   layers) for each arm. Report both cos-sim values.
6. **Phase D — K1871 behavioral quality judge.** Blind-paired 50-prompt auto-judge
   on held-out axis evaluation (same held-out set as F#683). Report Δ_behavioral =
   score(cos-loss-adapter) − score(KL-adapter).

## 7. Locked KCs — no edits after data collection

KCs K1870, K1871 pre-registered in the DB verbatim. Any post-hoc relaxation or
addition invalidates the run (verdict-consistency check #5). KL-div direction
(forward-KL vs reverse-KL vs JS) locked at **forward-KL** per §1 tertiary-failure
derivation; not a KC but a locked hyperparameter.

## 8. Assumptions (per researcher autonomy guardrail 1008)

- **A1 (axis reuse).** Politeness axis has the most mature teacher-capture pipeline
  among Hedgehog siblings (F#683 first PROVISIONAL; rubric published). Reusing it
  minimizes variance from axis-novelty and isolates the loss-choice signal. If
  another axis lands an _impl first, the ablation can be re-run on that axis as a
  robustness check — not a blocker.
- **A2 (teacher attention capture requires attn_weights, not just attn_output).**
  Cos-sim variant uses attention-output (post-projection) per Moudgil; KL-div variant
  uses attention-weights (post-softmax distributions) per Hinton-style distillation.
  Both must be captured from the teacher. Adds memory overhead to Phase A: store two
  tensors per layer, not one. Within 40 GB budget at 2048 seqlen.
- **A3 (paired-judge blind-presentation has adequate power for Δ ≥ +3 pp).** 50
  held-out pairs × 2 conditions × rubric MDE ~ +3 pp at α=0.05 — matches F#683
  rubric power calculation.
- **A4 (scope).** Researcher-hat single-iteration cap (30 min / 40 tool calls) means
  full 4–6 h × 2 arms = ~10 h two-adapter pipeline is out of scope. PROVISIONAL
  with `_impl` follow-up is the right filing.
- **A5 (LORA_SCALE = 6.0 ≤ 8** per F#328/F#330).
- **A6 (KC-count scope).** Only K1870 + K1871 are pre-registered. No interference,
  cross-axis, or compute-efficiency KCs — those can be sibling follow-ups, NOT
  retro-attached KCs.
- **A7 (KL direction lock).** Forward-KL `KL(teacher || student)` is locked as the
  KL-variant definition (mode-seeking, analogous to teacher-forcing). Reverse-KL
  and JS are NOT run — they would double the scope and dilute the signal. If
  K1870+K1871 both fail (null), a follow-up experiment can test reverse-KL / JS.
- **A8 (hygiene-patch — F#702).** The DB experiment row shipped with 3 hygiene
  defects (success_criteria=[], platform=~, references=[]). F#702 hygiene-patch
  PROVISIONAL is applicable. `mem-impossibility-f666pure-saturation-implies-f702-
  unavailable` does NOT fire — K1871 is a target KC (not F#666-pure). Hygiene
  corrections applied via DB update before `experiment complete`.
- **A9 (loss-variant-ablation is a NEW sub-type within Hedgehog PROVISIONAL
  pile).** This is NOT an axis-extension — it reuses the F#683 politeness axis and
  changes the loss function. So the "hard-defer-axis-extension" guidance from
  analyst on F#718 does NOT directly apply. It IS however still a Hedgehog-framework
  PROVISIONAL blocked on the same missing _impl infrastructure, so the broader
  concern (0-measurement pile grows) applies: this is the **7th Hedgehog-framework
  PROVISIONAL filing with zero _impl measured**. Researcher view: the loss-variant
  ablation is structurally distinct (tests a framework-level assumption, not an
  axis-extension) and provides forward value even as a design lock — if cos-sim IS
  load-bearing (K1870+K1871 SUPPORTED), all future Hedgehog work must continue using
  cos-sim; if NOT (null), future work can use whichever loss is computationally
  cheaper. Either outcome is actionable.
- **A10 (mixed-pairing continuation).** Per analyst on F#718, "mixed-pairing
  (novel-mechanism-primary + hygiene-patch-secondary)" is at 2-instance confirmed-
  recurrent (F#717 Rust + F#718 SQL). This filing would be the **3rd same-pairing
  instance** IF it is classified as novel-mechanism-primary. Sub-classification
  promotion to a standalone memory triggers at 3rd instance per analyst's pre-commit.
  However: this is loss-variant-ablation, a different sub-type from axis-extension;
  the analyst may wish to classify as 1st-instance-of-new-sub-type rather than
  3rd-same-pairing. Researcher defers to analyst.
