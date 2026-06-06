# MATH.md — exp_jepa_frozen_encoder_ablation (PREEMPT-KILL, TRIPLE-FIRE, 5th SAME-PARENT-F#682 POST-PROMOTION)

## Verdict: PREEMPT-KILL (triple-fire, 5th same-parent-F#682, 1st post-promotion instance of `mem-promotion-same-parent-repeat-blocker`)

Preempt-killed before any code executes. Fires **three** independent preempt-KILL memories AND is the first post-promotion instance of the same-parent-repeat-blocker memory (promoted at F#728, documenting the 4-child-count threshold).

Triple-fire composition:
1. **Memory: F#666-pure standalone** — the sole KC K1889 is a proxy (MSE ratio) with no target companion. Strict subset of F#666-pure pattern — even more degenerate than K1887/K1888 in F#728 because this is a **single-KC** set (pure; not mixed proxy+safety).
2. **Memory: §5 tautological-inter-variant-delta** — K1889 compares `frozen-encoder JEPA MSE` to `fine-tuned encoder JEPA MSE` with no external anchor. Both are untested realizations of parent F#682's JEPA mechanism.
3. **Memory: F#669 parent-target-unverified** — K1889's `fine-tuned encoder JEPA MSE` RHS is precisely parent F#682's untested K1767 baseline.

**Same-parent-repeat-blocker (post-promotion N+=1):** this is the **5th** child of `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) to hit preempt-KILL — first instance after the watchlist→standalone promotion at F#728. Routing expected to be stable per post-promotion canonical rule.

## §0 Platform / skills / model pins

- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: `v_proj + o_proj` (per F#627) — standard JEPA-adapter injection surface.
- Parent dependency (implicit): `exp_jepa_adapter_residual_stream` (status `provisional`, F#682). DB `depends_on=[]` but notes explicitly frame the ablation as "is adapter-only JEPA viable?" — i.e. ablates parent F#682's encoder-tuning.
- Platform pin at `experiment complete`: `local-apple` (per F#702 hygiene-patch).

## §1 Preempt-KILL theorems (three independent blocks)

### §1.1 F#666-pure standalone (memory: mem-preempt-f666-pure-standalone)

**Theorem (target-gate absence, single-KC variant).** Let `K = {K1889}` be the KC set. A KC is a *target metric* iff its LHS is a behavioral outcome (task accuracy), an oracle-gap, or externally-anchored quality claim (per guardrail 1007).

- **K1889** "Frozen-encoder JEPA MSE > 1.5x fine-tuned encoder JEPA MSE" — LHS is a prediction-loss ratio (structural, internal to the JEPA training objective). No external anchor; not a benchmark score, not a behavioral outcome, not an oracle-gap. **Proxy.**

∴ ∀k ∈ K: k is proxy. No target KC exists. F#666-pure standalone preempt-KILL fires.

Per memory `mem-preempt-f666-pure-standalone`: KILL requires BOTH target AND proxy to fail; a proxy-only KC set cannot produce a SUPPORTED verdict even in principle. This is the **17th** reuse (16 prior: F#700-F#716 scattered + F#728 → 17). More severe than F#728: |K|=1, so there is no companion KC to even re-classify. KC augmentation at re-claim must add a target **de novo**, not re-pair an existing KC.

**QED (block 1).**

### §1.2 §5 tautological-inter-variant-delta (memory: mem-preempt-s5-tautological-inter-variant-delta)

**Theorem (inter-variant delta, single-KC variant).** K1889 is a direct comparison `MSE(frozen-encoder JEPA) > 1.5 × MSE(fine-tuned-encoder JEPA)` where both variants are realizations of the same untested parent F#682 mechanism.

Under §5 canonical, comparing two untested instantiations of the same mechanism produces a tautological delta:

- If fine-tuned-encoder JEPA (= parent F#682 canonical) fails to learn, a "frozen loses 1.5×" outcome is vacuous — both are noise; ratio is dominated by variance.
- If fine-tuned encoder learns but is unvalidated on external target, we cannot distinguish "frozen is worse" from "frozen is just slower / has different bias" vs "neither is behaviorally viable."
- A fixed multiplicative threshold (1.5×) has no external calibration — why 1.5× rather than 1.2× or 2.0×? The threshold is arbitrary without an external loss curve to anchor it.

∴ K1889 is structurally non-informative. §5 preempt-KILL fires. This is the **11th** reuse (10 prior: F#704, F#709, F#711 + others through F#728).

**QED (block 2).**

### §1.3 F#669 parent-target-unverified (memory: mem-f669-preempt-parent-target-unverified)

**Theorem (inter-experiment target unverifiability, encoder-freeze-ablation sub-case).** Parent `P = exp_jepa_adapter_residual_stream` has `status=provisional`, with K1766/K1767 proxies and K1768/K1769 targets all untested per F#682.

- K1889's RHS "fine-tuned encoder JEPA MSE" is precisely parent P's untested canonical training trajectory (K1767 measures L_pred step500/step50 on this same trajectory). Without parent P's K1767 validated, we do not know whether the fine-tuned-encoder MSE baseline has a meaningful value — it could be collapsing, diverging, or learning. Comparing against it produces an unidentifiable sample per F#669 canonical.

∴ ∀k ∈ K: testing `k` while `P.status ∈ {provisional, open}` produces unidentifiable samples. F#669 preempt-KILL fires. This is the **7th** reuse (6 prior: F#669, F#687, F#698, F#699, F#727, F#728).

**QED (block 3).**

### §1.4 Composition: all three blocks are independent

Each block is individually sufficient. F#666-pure (block 1) fires on KC set shape alone. §5 (block 2) fires on K1889's inter-variant structure regardless of parent status. F#669 (block 3) fires on parent F#682 status regardless of KC structure. Each block reinforces the others but none depends on the others for its conclusion.

**Distinct structural severity vs F#728:** F#728 had |K|=2 (proxy + safety-guard). This has |K|=1 (single proxy). Under F#666-pure, K1889 has *no companion to re-pair* — re-claim must add a target from scratch, not re-label an existing KC.

## §2 Prior art

- **Memory: mem-preempt-f666-pure-standalone** — anchored at F#666 (2026-04-18); 16 reuses prior. **17th reuse here.**
- **Memory: mem-preempt-s5-tautological-inter-variant-delta** — anchored at F#704 (2026-04-24); 10 reuses prior. **11th reuse here.**
- **Memory: mem-f669-preempt-parent-target-unverified** — anchored at F#669 (2026-04-19); canonical promoted at F#698 (3rd reuse). Prior reuses: F#669, F#687, F#698, F#699, F#727, F#728. **This is the 7th reuse.**
- **Memory: mem-promotion-same-parent-repeat-blocker** — anchored at F#728 (2026-04-24). Prior same-parent-F#682 preempt-KILLs: F#687, F#698, F#727, F#728 (4 instances at promotion). **This is the 5th — first post-promotion N+=1 confirmation.**
- **Memory: mem-promotion-triple-fire-mode** — anchored at F#721 (2026-04-24). Prior triple-fires: F#714, F#716, F#720, F#721, F#722, F#728. **This is the 7th triple-fire** (post-promotion routing stable across 6 prior).
- **F#682** parent PROVISIONAL: design-only, 4 target-gated KCs untested.
- **F#627** Gemma 4 E4B LoRA baseline (parent K1768 RHS).
- **Guardrail 1007** (F#666): target-gated KILL discipline.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                     | Kind      | Measurement status                                |
| ----- | ------------------------------------------------------------------------- | --------- | ------------------------------------------------- |
| K1889 | Frozen-encoder JEPA MSE > 1.5× fine-tuned encoder JEPA MSE                | proxy     | untested (preempt-blocked, triple-fire)           |

**Observation:** |K|=1. This is a single-KC experiment (even before ablation of safety-guard-as-KC). F#666-pure is *maximally* degenerate here — there is no other KC even in principle to pair with.

## §4 Unblock condition

The preempt-KILL clears when ALL THREE block conditions clear:

1. **F#666-pure block clears** by *adding* a target KC from scratch (not re-pairing — there is only one existing KC). Concrete example: "Frozen-encoder JEPA adapter GSM8K-Hard accuracy ≥ (fine-tuned encoder - 3pp) at matched param budget on Gemma 4 E4B." This is genuine KC-augmentation, not relabeling.
2. **§5 block clears** by anchoring K1889 (or the added target) to an external behavioral baseline — same KC-augmentation resolves §5.
3. **F#669 block clears** when parent `exp_jepa_adapter_residual_stream` reaches `status=supported` via `exp_jepa_adapter_residual_stream_impl` (P=1, filed) with K1767 + K1768 + K1769 all SUPPORTED.

Block-1 and Block-2 are resolved by the same KC augmentation. Block-3 is an external parent-dependency gate.

**Summary unblock action:** (a) add a target KC (not relabel), AND (b) wait for parent F#682 `_impl` to land SUPPORTED.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#727/F#728 precedent + reviewer.md §5. Unblock is a combination of:

- KC-augmentation at re-claim (researcher-owned action at re-claim time)
- Parent-external: `exp_jepa_adapter_residual_stream_impl` (P=1, already filed)

## §6 Scope integrity

No silent objective swap (antipattern-t): scaffold does NOT attempt a "simpler" ablation (e.g. comparing frozen vs fine-tuned on an unrelated synthetic task, or substituting a non-JEPA objective). Both shortcuts would substitute proxy phenomena for the encoder-freeze mechanism K1889 measures.

Pre-registered KC preserved verbatim. DB text matches MATH.md §3 verbatim — no post-claim KC mutation.

## §7 Meta-routing: same-parent-repeat-blocker POST-PROMOTION INSTANCE 1

This is the **5th child** of F#682 PROVISIONAL to hit preempt-KILL — first instance after `mem-promotion-same-parent-repeat-blocker` was promoted at F#728.

| # | Child                                     | Finding | KC structure              | F#666 state   | Preempt memories fired                    |
| - | ----------------------------------------- | ------- | ------------------------- | ------------- | ----------------------------------------- |
| 1 | exp_jepa_router_prediction_error          | F#687   | proxy+proxy               | compound      | F#669                                     |
| 2 | exp_jepa_adapter_attention_output         | F#698   | proxy-only (|K|=2)        | compound      | F#669                                     |
| 3 | exp_jepa_multilayer_prediction            | F#727   | proxy+target              | compliant     | F#669                                     |
| 4 | exp_jepa_contrastive_variant              | F#728   | proxy+safety (|K|=2)      | pure          | F#669 + §5 + F#666-pure (triple)          |
| 5 | **exp_jepa_frozen_encoder_ablation (this)** | **TBD** | **proxy-only (|K|=1)**  | **pure**      | **F#669 + §5 + F#666-pure (triple)**      |

Parent F#682 unblock leverage now **5:1** (one `_impl` landing unblocks 5 child re-claims — modulo each child's own KC-augmentation needs).

**Expected behavior under post-promotion routing:** per `mem-promotion-same-parent-repeat-blocker`, 5th-and-later claims contribute N+=1 only (no new information beyond count). Route as preempt-KILL per standard triple-fire template; do not re-derive memory theorem; note "post-promotion N+=1 confirmation" in finding/evidence. No new memory write.

## §8 Assumptions (A1-A12)

1. **A1:** Parent F#682 remains PROVISIONAL at claim time. Verified via `experiment get exp_jepa_adapter_residual_stream` showing `status=provisional`.
2. **A2:** K1889's "fine-tuned encoder JEPA" refers to parent F#682's canonical mechanism (JEPA with both encoder and adapter trained). Supported by experiment notes framing as "adapter-only" ablation of parent.
3. **A3:** K1889 is a proxy (prediction-loss ratio), not a target metric per guardrail 1007. Verified by inspection: MSE ratio is structural / internal to training objective, no external behavioral anchor.
4. **A4:** DB `depends_on=[]` does not preclude structural dependence on parent F#682. F#669 theorem applies when child KCs *reference* parent's unverified claims, regardless of explicit DB edge. Confirmed by F#687/F#698/F#727/F#728 precedent.
5. **A5:** F#669 post-promotion routing (confirmed stable across 6 prior reuses) applies at 7th reuse without re-derivation.
6. **A6:** Triple-fire-mode post-promotion routing (confirmed stable across 6 prior triple-fires post-F#721 promotion) applies at 7th triple-fire without re-derivation.
7. **A7:** Same-parent-repeat-blocker post-promotion routing (promoted at F#728) applies at this first post-promotion instance without re-derivation or memory update. N=5 logged for census.
8. **A8:** Verdict classification: `killed` per CLI status enum. Kill reason encoded in evidence + MATH.md + PAPER.md.
9. **A9:** No `_impl` follow-up filed. Parent's existing `exp_jepa_adapter_residual_stream_impl` (P=1) is the sole external unblock gate; KC-augmentation at re-claim here requires genuine new target KC (|K| increases from 1 to ≥2).
10. **A10:** All 4 required artifacts (MATH.md, run_experiment.py, results.json, PAPER.md) written per objective success-criteria. LEARNINGS.md + REVIEW-adversarial.md are downstream hats' responsibilities.
11. **A11:** F#702 hygiene-patch APPLIED for `platform`, `dir`, `success_criteria`, `evidence`. `references` array left empty per F#698/F#699/F#727/F#728 precedent for preempt-KILL.
12. **A12:** No new memory promotion triggered. F#669/§5/F#666-pure/triple-fire-mode/same-parent-repeat-blocker all post-promotion; routing stable. N+=1 census updates inline in the finding body.
