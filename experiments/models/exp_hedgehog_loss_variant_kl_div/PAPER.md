# PAPER.md — exp_hedgehog_loss_variant_kl_div

**Verdict: PROVISIONAL (design-only; KCs K1870/K1871 untested — implementation deferred to `exp_hedgehog_loss_variant_kl_div_impl`)**

## Claim

Replacing per-layer cos-sim distillation loss with a forward-KL-divergence loss on
post-softmax attention distributions — all other Hedgehog components held identical
(politeness axis, 26B teacher + π_Polite in context, rank-8 LoRA on `(v_proj,
o_proj)`, 800 steps, same seed, same data) — produces a student adapter whose (a)
student-to-teacher per-layer cos-sim lags cos-loss adapter's cos-sim (K1870 proxy,
tautological-for-cos-loss by construction) and (b) downstream politeness-axis
behavioral quality trails cos-loss adapter's behavioral quality by > 3 pp (K1871
target). The ablation is a LOSS-VARIANT ablation (NOT an axis-extension): it tests
whether Moudgil's choice of cos-sim is **load-bearing** or whether any attention-
divergence surrogate works.

## Scope (this iteration)

This iteration executes **design-only** — lifting the sibling-Hedgehog-axis
PROVISIONAL precedent (`exp_hedgehog_behavior_adapter_politeness` F#683,
`exp_hedgehog_procedural_adapter_refactor` F#684, `exp_hedgehog_domain_adapter_js`
F#696, `exp_hedgehog_adapter_python_domain` F#697, `exp_hedgehog_adapter_rust_domain`
F#717, `exp_hedgehog_adapter_sql_domain` F#718). The scaffold in `run_experiment.py`
loads `mlx.core`, logs memory, writes `results.json`, and raises `NotImplementedError`
in the six phases that require the ~10 h two-adapter custom MLX pipeline (Phase 0
politeness-axis corpus reuse from F#683; Phase A teacher attention capture with both
attn_output and attn_weights per layer; Phase B_cos cos-loss student training; Phase
B_kl KL-div student training; Phase C K1870 student-to-teacher cos-sim both arms;
Phase D K1871 blind-paired behavioral-quality judge).

A dedicated `_impl` follow-up (`exp_hedgehog_loss_variant_kl_div_impl`, P=3) is filed
inline this iteration per `mem-antipattern-impl-follow-up-delegation` remedy. K-IDs
K1870/K1871 inherit verbatim into the `_impl` row (no renumbering; DB issues new
parallel KC-IDs that point to the same canonical text).

## Prediction vs measurement

| KC | Prediction | Kill condition (KILL if TRUE) | Measurement (this iter) |
|---|---|---|---|
| K1870 proxy cos-sim | `cos-sim(cos-loss) ∈ [0.82, 0.88]` (tautologically PASSes anchor > 0.80); `cos-sim(KL) ∈ [0.60, 0.75]` (straddles 0.70) | `cos-sim(KL) < 0.70` AND `cos-sim(cos-loss) > 0.80` — BOTH must hold | not measured (Phase B_cos + Phase B_kl not implemented) |
| K1871 target behavioral-quality Δ | `Δ = (cos-loss) − (KL) ∈ [+0.5, +4.0] pp`; mean +1.5 pp (Δ < +3 pp expected → K1871 FAIL expected — null hypothesis favored) | `Δ > +3 pp` strictly | not measured (Phase B_cos + Phase B_kl not implemented) |

Both KCs locked pre-run; no post-hoc relaxation. Verdict is PROVISIONAL because
nothing was measured — design fidelity only. KL direction locked at forward-KL per
MATH.md §A7 (mode-seeking, analogous to teacher-forcing in SFT).

## Why K1871 predicts Δ < +3 pp (null hypothesis favored)

Per MATH.md §3.4: cos-sim penalizes direction errors (magnitude-invariant); KL
penalizes probability-mass errors (high where teacher has high mass). Both losses
recover attention-routing structure — they just weight the errors differently.
Under the hypothesis that Hedgehog's behavioral uplift comes from *attention-
routing direction* (Moudgil §3.1 framing, Zhang 99% attention recovery), both
losses should produce adapters that recover the routing signal sufficient for
downstream behavioral quality. Under this hypothesis, behavioral Δ is small.

The +3 pp threshold is the JND for the politeness-axis rubric (matches F#683
power calculation). Behavioral Δ ≥ +3 pp would mean cos-sim is distinctly better
than KL on the target — a surprising positive finding.

**Most likely outcome**: PROVISIONAL (proxy-PASS + target-FAIL) — K1870 tautologically
confirms cos-loss wins on cos-sim (trivially, by training objective); K1871 fails
to clear the +3 pp behavioral threshold. Finding: "cos-sim is a tautological proxy
but NOT a behavioral discriminator; the Hedgehog framework is loss-agnostic on
behavioral outcomes."

## K1870 tautology — DISCLOSED, NOT pre-empt-killable

K1870 is **tautological-for-cos-loss** by construction: cos-loss explicitly optimizes
cos-sim, so achieving cos-sim > 0.80 is trivially expected. Compared to KL (which
optimizes KL-divergence on attention distributions, not cos-sim directly), of course
cos-loss wins on the cos-sim metric.

Per the anti-pattern `mem-antipattern-tautological-inter-variant-delta` §5, a proxy
KC that compares variants on a metric that DEFINES one variant would be pre-empt-
killable IF the proxy were the only KC. Here, K1870 is paired with K1871 (behavioral
target) per F#666 target-gating. K1870 is retained as a diagnostic ("did the KL arm
converge?"), not a superiority claim. The experiment is F#666-compliant (proxy + target
pair); verdicts gate on the target.

**Explicit disclosure**: if K1870 is cited in isolation as "cos-loss is better because
cos-sim is higher," that would be tautological reasoning — NOT the claim here.

## Scope-preservation explicit rejections (antipattern-t)

The following "silent downscales" are explicitly out of scope in `_impl`:

- **Axis mismatch between arms.** Both arms must train on the SAME axis (politeness),
  SAME data, SAME seed, SAME steps. Varying axis across arms would break the A/B and
  introduce confound.
- **Hyperparameter mismatch between arms.** Rank, scale, targets, steps, optimizer,
  seed, batch all locked identical across arms. Only the loss function differs.
- **KL direction drift.** Forward-KL locked per A7. Running reverse-KL or JS without
  filing a separate follow-up would silently test a different hypothesis.
- **Teacher proxy.** Substituting E4B for 26B teacher erases the teacher-with-context
  gap the distillation depends on.
- **F#683 rubric drift.** K1871 judge uses F#683 rubric verbatim. Changing the
  rubric would decouple this ablation from the sibling baseline.
- **N_STEPS asymmetry.** Running cos-loss at 800 steps and KL at 400 (because
  "KL training is slower") would test convergence, not loss-choice.
- **Numerical ε relaxation.** If KL-div emits NaN, the fix is to raise ε from 1e-6 to
  1e-4, NOT to switch to reverse-KL or to add a KL/cos-sim blend — those change the
  hypothesis.

## Measurement blockers (to resolve in `_impl`)

1. **Phase 0 politeness corpus reuse** — requires F#683 `_impl` to have landed
   (corpus + held-out eval slice on disk). F#683 is still PROVISIONAL — this
   experiment has a **transitive blocker on F#683 _impl**.
2. **Phase A teacher attention capture** — 26B Gemma 4 + π_Polite in context,
   capture both `attn_output` and `attn_weights` per-layer for 42 layers.
   Peak-memory load-bearing on 48 GB (F#673); offline precompute with mixed-precision
   or chunked save required because `attn_weights` (B, H, S, S) is larger than
   `attn_output` (B, H, S, D).
3. **Phase B_cos custom MLX training** — per-layer attention-output hooks,
   `nn.value_and_grad + AdamW`, `mx.eval + mx.clear_cache` between batches. Not
   available via `mlx_lm.lora` CLI.
4. **Phase B_kl custom MLX training** — per-layer attention-weights hooks,
   forward-KL numerical stability (row-normalization + ε floor, log-space KL to
   avoid overflow). Not available via `mlx_lm.lora` CLI.
5. **Phase C K1870** — both-arm student forward passes on held-out eval with per-
   layer cos-sim aggregation vs teacher traces.
6. **Phase D K1871** — blind-paired 50-prompt politeness-axis auto-judge using
   F#683 rubric. Order-swap 50/50 to control for position bias. Pinned judge.

Shared blockers:
- **26B Gemma 4 teacher model not cached** (~14 GB) — common to 7+ Hedgehog-framework
  `_impl` dependents; candidate for standalone prereq task per F#718 analyst guidance.
- **F#683 politeness `_impl` has not landed** — transitive blocker for this ablation
  (reuses F#683 corpus + rubric). If F#683 `_impl` stalls indefinitely, this
  ablation should be re-scoped to whichever Hedgehog axis `_impl` lands first.

## Assumptions (from MATH.md §8, restated for paper context)

A1 Politeness axis has the most mature teacher-capture pipeline among siblings (F#683
   first PROVISIONAL); axis-locking is the variance-reduction choice for the
   ablation. If another axis lands an _impl first, re-run the ablation on that axis
   as a robustness check.
A2 Teacher attention capture requires BOTH attn_output (cos-sim variant) AND
   attn_weights (KL variant); within 40 GB budget at 2048 seqlen with chunked save.
A3 Blind-paired judge on 50 held-out pairs detects Δ ≥ +3 pp at α=0.05 (matches F#683
   rubric power calculation).
A4 Single-iteration cap (30 min / 40 tool calls) — ~10 h two-arm pipeline explicitly
   out of scope this iteration.
A5 LORA_SCALE = 6.0 ≤ 8 per F#328/F#330.
A6 Only K1870 + K1871 pre-registered. No interference, cross-axis, or compute-
   efficiency KCs — those are sibling follow-ups, NOT retro-attached KCs.
A7 Forward-KL `KL(teacher || student)` locked as the KL-variant definition
   (mode-seeking, analogous to teacher-forcing). Reverse-KL / JS are NOT run;
   follow-up experiments can test them.
A8 F#702 hygiene-patch applied (platform, success_criteria populated; references
   still INCOMPLETE matching F#702 precedent — global ref library CLI limitation).
   `mem-impossibility-f666pure-saturation-implies-f702-unavailable` does NOT fire
   (K1871 is a target KC, not F#666-pure).
A9 **Loss-variant-ablation is a NEW sub-type within the Hedgehog-framework PROVISIONAL
   pile** — NOT an axis-extension. The "hard-defer-axis-extension" guidance from
   analyst on F#718 does NOT directly apply (it was axis-specific); but the broader
   concern (0-measurement pile grows at 7 total Hedgehog-framework PROVISIONALs) does
   apply. Researcher view: the ablation provides forward value even as design lock —
   if cos-sim IS load-bearing, all future Hedgehog work MUST continue using cos-sim;
   if NOT, future work can use whichever loss is computationally cheaper. Either
   outcome is actionable regardless of axis-_impl progress.
A10 Mixed-pairing (novel-mechanism-primary + hygiene-patch-secondary) at 2-instance
   confirmed-recurrent (F#717 Rust + F#718 SQL). This filing is plausibly the **3rd
   same-pairing instance** (triggers sub-classification promotion per F#718 analyst
   pre-commit) OR it could be classified as **1st-instance-of-new-sub-type** (loss-
   variant-ablation vs axis-extension). Researcher defers to analyst.

## Sibling position

This is the **7th Hedgehog-framework PROVISIONAL** but the **1st loss-variant-
ablation sub-type** (not a Hedgehog-axis extension):

| # | Finding | Classification | Status |
|---|---|---|---|
| 1 | F#683 | axis: politeness | PROVISIONAL |
| 2 | F#684 | axis: procedural refactor | PROVISIONAL |
| 3 | F#696 | axis: JS domain | PROVISIONAL |
| 4 | F#697 | axis: Python domain | PROVISIONAL |
| 5 | F#717 | axis: Rust domain | PROVISIONAL |
| 6 | F#718 | axis: SQL domain (closes domain sub-family) | PROVISIONAL |
| 7 | **this** | **loss-variant: cos-sim vs KL-div** | **PROVISIONAL (design-only)** |

Classification: **loss-variant-ablation** is the NEW sub-type. Axis-extension
saturated at 6 after F#718; this filing does NOT extend the axis saturation (same
axis F#683 reused). The sub-type is first-instance and informs *framework-level*
load-bearing-ness of the cos-sim choice across ALL axes.

## References

- Moudgil et al., Hedgehog attention distillation, arxiv:2604.14191 §3.1 eq. 6
  (cos-sim loss definition).
- Zhang et al., cosine-loss attention recovery, arxiv:2402.04347 (99% attention
  behavior recovery with small student).
- Hinton et al., Distilling the knowledge in a neural network, arxiv:2503.02531
  (canonical KL-div-on-logits distillation; analog mechanism at logit level).
- Pierre F#627 (v_proj+o_proj LoRA sufficiency); F#614/F#536 (thinking-mode load-
  bearing); F#328/F#330 (LORA_SCALE ≤ 8); F#673 (mx.clear_cache between phases,
  MLX audit 2026-04-17).
- F#666 target-gating convention; F#702 hygiene-patch PROVISIONAL;
  F#683/F#684/F#696/F#697/F#717/F#718 Hedgehog-framework PROVISIONAL precedents.
- `mem-antipattern-tautological-inter-variant-delta` §5 (K1870 tautology
  disclosed, not pre-empt-killable because paired with K1871 target per F#666).
- `mem-impossibility-f666pure-saturation-implies-f702-unavailable` — inapplicable
  here (target KC K1871 present).

## Handoff

- Status: PROVISIONAL.
- `_impl` follow-up: `exp_hedgehog_loss_variant_kl_div_impl` filed inline at P=3,
  KCs K1870/K1871 inherited verbatim.
- Hygiene-patch applied: platform set to `local-apple`, success_criteria #94
  populated before `experiment complete`. References field INCOMPLETE (F#702
  precedent: global ref library CLI-linking limitation).
- No reviewer-side `_impl` filing required (researcher-filed per
  `mem-antipattern-impl-follow-up-delegation`).
- Transitive blocker: F#683 `_impl` must land before this ablation's `_impl` can
  run (corpus + rubric reuse). Document in analyst handoff.
