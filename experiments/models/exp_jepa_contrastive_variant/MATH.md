# MATH.md — exp_jepa_contrastive_variant (PREEMPT-KILL, TRIPLE-FIRE + SAME-PARENT-REPEAT-BLOCKER PROMOTION)

## Verdict: PREEMPT-KILL (triple-fire)

This experiment is preempt-killed before any code executes. It simultaneously fires **three** distinct preempt-KILL memories AND crosses the 4th-instance promotion threshold for the **same-parent-repeat-blocker** watchlist pattern (F#682 child count).

Triple-fire composition:
1. **Memory: F#666-pure standalone preempt** — neither KC is a target metric.
2. **Memory: §5 tautological-inter-variant-delta** — K1887 directly compares InfoNCE to MSE variant with no external anchor.
3. **Memory: F#669 parent-target-unverified** — K1887's MSE-variant RHS is parent F#682's untested baseline; K1888 tests stability of a design whose behavioral claim is itself unverified.

Same-parent-repeat-blocker (watchlist at F#727): this is the **4th** child of `exp_jepa_adapter_residual_stream` (F#682 PROVISIONAL) to hit preempt-KILL — promotion threshold crossed (memory write deferred to analyst per canonical routing).

## §0 Platform / skills / model pins

- Platform skills: `/mlx-dev` + `/fast-mlx` (per `PLAN.md` Part 2). **Not invoked** — no MLX code written; honest disclosure per reviewer checklist.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: `v_proj + o_proj` (per F#627) — standard JEPA-adapter injection surface.
- Parent dependency (implicit): `exp_jepa_adapter_residual_stream` (status `provisional`, F#682). DB `depends_on=[]` but the experiment notes explicitly test "whether contrastive loss improves JEPA signal" — the JEPA framework *is* parent F#682.
- Platform pin at `experiment complete`: `local-apple` (per F#702 hygiene-patch).

## §1 Preempt-KILL theorems (three independent blocks)

### §1.1 F#666-pure standalone (memory: mem-preempt-f666-pure-standalone)

**Theorem (target-gate absence).** Let `K = {K1887, K1888}` be the KC set. A KC is a *target metric* iff its LHS is a behavioral outcome (task accuracy on an external benchmark), an oracle-gap (distance to ground-truth decision), or otherwise an externally-anchored quality claim (per guardrail 1007). Inspecting both KCs:

- **K1887** "InfoNCE variant next-embedding accuracy < MSE variant" — LHS is *next-embedding accuracy*, a structural prediction-quality metric on the adapter's internal prediction task. No external anchor (not a benchmark score, not a behavioral outcome on downstream generation). **Proxy.**
- **K1888** "InfoNCE training unstable (loss NaN or divergence > 3 epochs)" — LHS is a *training-dynamics safety guard* (NaN detection). No external anchor (not accuracy, not behavioral quality, not oracle-gap). **Proxy / safety-guard.**

∴ ∀k ∈ K: k is proxy or safety-guard. No target KC exists. F#666-pure standalone preempt-KILL fires.

Per memory `mem-preempt-f666-pure-standalone`: KILL requires BOTH target AND proxy to fail; a proxy-only KC set cannot produce a SUPPORTED verdict even in principle. The experiment is structurally unsupportable.

**QED (block 1).**

### §1.2 §5 tautological-inter-variant-delta (memory: mem-preempt-s5-tautological-inter-variant-delta)

**Theorem (inter-variant delta without external anchor).** K1887 is a direct comparison `accuracy(InfoNCE variant) < accuracy(MSE variant)` where both variants are realizations of the same untested JEPA-adapter mechanism (parent F#682 PROVISIONAL). Neither variant has been target-validated against external behavioral quality.

Under §5 canonical: comparing two untested instantiations of the same mechanism is tautological — the delta has no external referent. Specifically:

- If the MSE variant fails to learn (parent F#682 K1768 unverified), an "InfoNCE beats MSE" outcome is vacuous (both are noise; InfoNCE wins by fluctuation, not by learning signal).
- If the MSE variant does learn but has not been validated, we cannot distinguish "InfoNCE is better" from "InfoNCE is also learning, and both are fine" vs "neither learns."
- The kill criterion cannot discriminate these cases without anchoring both to an external target.

∴ K1887 is structurally non-informative. §5 preempt-KILL fires.

**QED (block 2).**

### §1.3 F#669 parent-target-unverified (memory: mem-f669-preempt-parent-target-unverified)

**Theorem (inter-experiment target unverifiability, contrastive-loss-variant sub-case).** Parent `P = exp_jepa_adapter_residual_stream` has `status=provisional`, with K1766/K1767 proxies and K1768/K1769 targets all untested per F#682.

- K1887 compares InfoNCE-variant accuracy to *MSE-variant accuracy*. The MSE variant is exactly parent P's primary mechanism. Without parent P's K1767 (L_pred ratio proxy) and K1768 (GSM8K-Hard target) validated, we do not know whether the MSE-variant baseline is "learning real dynamics." Comparing against an unverified baseline produces an unidentifiable sample (per F#669 canonical).
- K1888 tests whether InfoNCE training is stable — but a stability check on a loss variant of an unvalidated mechanism cannot inform the behavioral-quality question the parent purports to answer. Stability of a useless training objective has no behavioral implication.

∴ ∀k ∈ K: testing `k` while `P.status ∈ {provisional, open}` produces unidentifiable samples. F#669 preempt-KILL fires (6th reuse).

**QED (block 3).**

### §1.4 Composition: all three blocks are independent

Each block is sufficient individually. F#666-pure (block 1) fires on the KC set alone (target absence). §5 (block 2) fires on K1887's inter-variant structure regardless of parent status. F#669 (block 3) fires on parent F#682 status regardless of KC structure. The three blocks reinforce each other but each would independently produce a preempt-KILL verdict.

## §2 Prior art

- **Memory: mem-preempt-f666-pure-standalone** — anchored at F#666 (2026-04-18); 15 reuses prior (F#700-712, scattered). 16th reuse here.
- **Memory: mem-preempt-s5-tautological-inter-variant-delta** — anchored at F#704 (2026-04-24); 9 prior instances (F#704, F#709, F#711, plus other §5 instances). 10th reuse here.
- **Memory: mem-f669-preempt-parent-target-unverified** — anchored at F#669 (2026-04-19); promoted canonical at F#698 (3rd reuse). Prior reuses: F#669, F#687, F#698, F#699, F#727. This is the **6th reuse**.
- **Memory: mem-same-parent-repeat-blocker (watchlist)** — anchored at F#727 (2026-04-24). Prior same-parent-F#682 preempt-KILLs: F#687 (router_prediction_error), F#698 (attention_output), F#727 (multilayer_prediction). This is the **4th** — PROMOTION TRIGGER.
- **Memory: mem-promotion-triple-fire-mode** — anchored at F#721 (2026-04-24). Prior triple-fires: F#714, F#716, F#720, F#721, F#722. This is the **6th triple-fire** (post-promotion routing stable).
- **F#682** parent PROVISIONAL: design-only, 4 target-gated KCs untested.
- **F#627** Gemma 4 E4B LoRA baseline (parent K1768 RHS).
- **Guardrail 1007** (F#666): target-gated KILL discipline.

## §3 Predictions (registered, not measured)

All KC states are **"untested (preempt-blocked)"**:

| KC    | Claim                                                                     | Kind      | Measurement status                                |
| ----- | ------------------------------------------------------------------------- | --------- | ------------------------------------------------- |
| K1887 | InfoNCE variant next-embedding accuracy < MSE variant                     | proxy     | untested (preempt-blocked, triple-fire)           |
| K1888 | InfoNCE training unstable (loss NaN or divergence > 3 epochs)             | safety    | untested (preempt-blocked, triple-fire)           |

## §4 Unblock condition

The preempt-KILL clears when ALL THREE block conditions clear:

1. **F#666-pure block clears** by adding an externally-anchored target KC (e.g. "InfoNCE-variant adapter GSM8K-Hard accuracy ≥ LoRA r=16 baseline") — KC-augmentation. Similar to F#698-attention_output unblock (KC augment).
2. **§5 block clears** by anchoring K1887 to an external target (same KC-augmentation as above — adding a behavioral-quality claim breaks the inter-variant-only pattern).
3. **F#669 block clears** when parent `exp_jepa_adapter_residual_stream` reaches `status=supported` via `exp_jepa_adapter_residual_stream_impl` (P=1, filed) with K1767 + K1768 + K1769 all SUPPORTED.

Both block-1 and block-2 are resolved by the **same** KC augmentation (add a target KC). Block-3 is an external parent-dependency gate.

**Summary unblock action:** (a) augment KC set with one target metric AND (b) wait for parent F#682 `_impl` to land.

## §5 Follow-up

No `_impl` companion filed — preempt-structural kill is self-contained per F#687/F#698/F#699/F#727 + reviewer.md §5. Unblock is a combination of:

- KC-augmentation at re-claim (researcher-owned action at re-claim time)
- Parent-external: `exp_jepa_adapter_residual_stream_impl` (P=1, already filed)

This distinction between novel-mechanism PROVISIONAL (which mandates `_impl`) and preempt-structural KILL (which does NOT spawn `_impl`) is canonical per F#687/F#698/F#699/F#727.

## §6 Scope integrity

No silent objective swap (antipattern-t): scaffold does NOT attempt a "simpler" ablation (e.g. training InfoNCE standalone and reporting absolute accuracy, or substituting a non-JEPA baseline). Both shortcuts would substitute proxy phenomena for the contrastive-loss mechanism the KCs measure.

Pre-registered KCs preserved verbatim. DB text matches MATH.md §3 verbatim — no post-claim KC mutation.

## §7 Meta-routing: same-parent-repeat-blocker PROMOTION TRIGGER

This is the **4th child** of F#682 PROVISIONAL (`exp_jepa_adapter_residual_stream`) to hit preempt-KILL:

| # | Child                                     | Finding | KC structure              | Preempt memories fired |
| - | ----------------------------------------- | ------- | ------------------------- | ---------------------- |
| 1 | exp_jepa_router_prediction_error          | F#687   | proxy+proxy (F#666 compound) | F#669                  |
| 2 | exp_jepa_adapter_attention_output         | F#698   | proxy-only (F#666 compound) | F#669                  |
| 3 | exp_jepa_multilayer_prediction            | F#727   | proxy+target (F#666-compliant) | F#669                  |
| 4 | exp_jepa_contrastive_variant (this)       | TBD     | proxy+safety (F#666-pure)  | **F#669 + §5 + F#666-pure (triple)** |

Parent F#682 unblock leverage: **≥4:1** (one `_impl` landing unblocks 4 child re-claims — modulo each child also needing its own KC-augmentation if it fires F#666-pure/compound independently).

Same-parent-repeat-blocker watchlist threshold reached. Per F#727's canonical note ("If 4th same-parent child of F#682 hits preempt-KILL, promote to standalone memory"), the memory promotion fires now. Memory write is analyst-owned per precedent (researcher writes artifacts + files finding; analyst writes memory promotion).

## §8 Assumptions (A1-A12)

1. **A1:** Parent F#682 remains PROVISIONAL at claim time. Verified via `experiment get exp_jepa_adapter_residual_stream` showing `status=provisional`.
2. **A2:** K1887's "MSE variant" refers to parent F#682's JEPA-adapter trained with MSE loss (parent's canonical mechanism), not some other MSE baseline. Supported by experiment notes: "InfoNCE contrasts predicted vs negative embeddings. Tests whether contrastive loss improves JEPA signal" — "JEPA signal" implies parent F#682 framework.
3. **A3:** K1888 "training unstable" is a safety guard / proxy, not a target metric per guardrail 1007. Verified by inspection: NaN/divergence detection is a training-dynamics property, not a behavioral outcome or external anchor.
4. **A4:** DB `depends_on=[]` does not preclude structural dependence on parent F#682. F#669 theorem applies when child KCs *reference* parent's unverified claims, regardless of explicit DB edge. Confirmed by F#687/F#698/F#727 precedent (same parent, structural dependence despite edge variation).
5. **A5:** F#669 post-promotion routing (confirmed stable across 5 prior reuses) applies at 6th reuse without re-derivation.
6. **A6:** Triple-fire-mode post-promotion routing (confirmed stable across 4 prior triple-fires post-F#721 promotion) applies at 6th triple-fire without re-derivation.
7. **A7:** Same-parent-repeat-blocker promotion threshold = 4 instances (per F#727 canonical note "4th same-parent child → promote"). This is the 4th, trigger fires.
8. **A8:** Verdict classification: `killed` per CLI status enum. Kill reason encoded in evidence + MATH.md + PAPER.md.
9. **A9:** No `_impl` follow-up filed. Parent's existing `exp_jepa_adapter_residual_stream_impl` (P=1) is the sole external unblock gate; KC-augmentation is in-scope for re-claim, not `_impl`.
10. **A10:** All 4 required artifacts (MATH.md, run_experiment.py, results.json, PAPER.md) written per objective success-criteria. LEARNINGS.md + REVIEW-adversarial.md are downstream hats' responsibilities.
11. **A11:** F#702 hygiene-patch APPLIED for `platform`, `dir`, `success_criteria`, `evidence`. `references` array left empty per F#698/F#699/F#727 precedent for preempt-KILL.
12. **A12:** No new F#669 or §5 or F#666-pure sub-axes emitted. This is canonical post-promotion reuse for all three memories. The **new** memory-write milestone is `mem-promotion-same-parent-repeat-blocker` (analyst-owned) triggered at this 4th-instance threshold.
