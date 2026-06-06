# MATH.md — exp_g4_per_layer_cos_baseline (PREEMPT-KILL, F#666-pure)

## Verdict: PREEMPT-KILL (KC-structural, F#666-pure)

This experiment is preempt-killed before any code is run. The kill is **structural**: the pre-registered kill-criterion set violates guardrail 1007 (Finding #666, target-gated KILL discipline). No MLX code is attempted because the KC set cannot produce a valid verdict under F#666 regardless of empirical outcome.

This is a **new sub-case** in drain-window taxonomy, orthogonal to the F#669 family:
- F#669 family (F#669/F#687/F#698/F#699): preempt because **parent PROVISIONAL** ⇒ child KCs require non-existent artifact.
- **This case (F#666-pure standalone):** no parent dependency, but KC set itself is malformed (proxy-only, no target) ⇒ no verdict possible under F#666.

## §0 Platform / skills / model pins

Included for completeness per reviewer checklist item (m2), even though no platform code executes.
- Platform skills: `/mlx-dev` + `/fast-mlx` (per PLAN.md Part 2). **Not invoked** — no MLX code written; canonical preempt-form disclosure.
- Base model: `mlx-community/gemma-4-e4b-it-4bit` (per F#627). **Not loaded.**
- Adapter targets: N/A — this experiment is measurement-only (no LoRA).
- Parent dependency: **none** (`depends_on: []`). This is NOT an F#669 preempt.

## §1 Preempt-KILL theorem (F#666-pure)

**Theorem (KC-structural invalidity under target-gated KILL).** Let `E` denote experiment `exp_g4_per_layer_cos_baseline` with kill-criterion set K = {K1856}. Let `K1856` := "Per-layer cos-sim variance across 100 diverse prompts < 0.02 (routing is uniform, not layer-specific)".

**Classification of K1856:** K1856 is a **proxy metric** (geometric similarity of representations). It is not a target metric (behavioral task accuracy, oracle-gap, user-quality). The threshold 0.02 is a structural claim about embedding geometry, not an outcome that binds to downstream behavioral quality.

**F#666 gating (guardrail 1007):** KILL requires **both** a failing proxy KC and a failing target KC. SUPPORTED requires **both** to pass. Any verdict depending on a proxy-only KC set is tautological.

**Corollary (standalone F#666-pure preempt).** Let V(K) be the verdict derivable from K = {K1856}:
- If K1856 passes (variance < 0.02): V(K) = "routing is uniform" — a statement about embedding geometry, not about whether Hedgehog distillation *works behaviorally*. This is proxy-SUPPORTED without target pairing → tautological per F#666.
- If K1856 fails (variance ≥ 0.02): V(K) = "routing is layer-specific" — again, a geometric claim, not a behavioral one. This is proxy-FAIL without target pairing → per F#666 a proxy-FAIL alone does not KILL (Proxy-FAIL + target-PASS = finding about the proxy, not a kill).

∴ V(K) is unidentifiable under F#666 discipline. K is malformed. **QED.**

### §1.1 Secondary structural defects

Per `experiment get exp_g4_per_layer_cos_baseline`:

1. **`success_criteria: []`** — empty. No pass condition is declared. Independent of F#666, this means the experiment cannot be marked SUPPORTED under any outcome (there is no SUPPORTED-condition to reference).
2. **`references: []`** — violates guardrail 1002 ("Every new experiment MUST cite an arxiv paper or prior finding"). Purpose is stated as "baseline for Hedgehog distillation claims" but no Hedgehog-paper or prior-finding anchor is registered.
3. **`platform: ~`** — unset. Violates MATH.md §0 discipline (cannot declare base model / framework without a platform pin; researcher hat requires this explicit).

Each of (1)-(3) independently warrants a KC-augmentation pass; all three together plus the F#666 violation make the KC set structurally beyond repair without pre-registration modification.

## §2 Prior art

- **F#666** (target-gated KILL discipline, guardrail 1007): KILL requires proxy+target, SUPPORTED requires proxy+target. Proxy-alone is tautological.
- **F#698** (2026-04-24): first F#666-compound sub-case — JEPA child had proxy-only KC set AND parent-unverified. That KILL combined F#669 preempt + F#666 violation. This experiment isolates the F#666 arm (no parent dep).
- **F#669** (2026-04-19) / **F#687** (2026-04-23) / **F#699** (2026-04-24): preempt-KILL family for parent-unverified target. Not applicable here (no parent dependency).
- **F#683 / F#684 / F#696 / F#697** (2026-04-23/24): Hedgehog-axis PROVISIONAL set. This baseline experiment's purpose (per `notes`) is to "establish baseline for Hedgehog distillation claims" — but the Hedgehog experiments have already proceeded to design-lock PROVISIONAL without this baseline, so the downstream gating value is also minimal.

## §3 Predictions (registered, not measured)

| KC    | Claim                                                                                | Kind  | Measurement status                           |
| ----- | ------------------------------------------------------------------------------------ | ----- | -------------------------------------------- |
| K1856 | Per-layer cos-sim variance across 100 diverse prompts < 0.02 (routing uniform) | proxy | untested (preempt-blocked, F#666-pure structural) |

No target-metric KC exists. The KC set is structurally malformed per F#666.

## §4 Unblock condition

Re-claim requires **KC-augmentation** (pre-registration modification):

1. Add a target-metric KC pairing K1856 to a behavioral outcome. Candidate formulations:
   - "On Gemma 4 E4B distilled via Hedgehog cos-sim (per F#683/F#684/F#696/F#697 recipes), downstream task accuracy gain on domain-axis evaluation correlates with per-layer cos-sim variance at r ≥ 0.4 (Spearman across N=24 layers)."
   - Or: "Hedgehog distillation preserves ≥ 90% of base accuracy when applied only to layers with cos-sim variance > 0.05 (layer selection informed by this baseline)."
2. Add at least one reference (arxiv ID for Hedgehog method or prior finding).
3. Set `platform` field explicitly (e.g. `mlx`).
4. Populate `success_criteria` (mirror of KC pass condition).

These edits must happen **before** the experiment is re-claimed (KC-pre-registration rule). Post-claim KC mutation is antipattern-u per reviewer checklist.

**Note on downstream gating.** The Hedgehog-axis experiments (F#683/F#684/F#696/F#697) have already reached PROVISIONAL without this baseline. A re-claimed, augmented version would therefore be **best treated as a sibling of the Hedgehog family**, not a blocker. Consider re-scoping it as `exp_g4_per_layer_cos_variance_hedgehog_informed` with the target KC pairing baked in, rather than resurrecting the existing malformed pre-reg.

## §5 Follow-up

No `_impl` companion filed — preempt-structural KILL does NOT spawn `_impl` (per F#687/F#698/F#699 precedent + reviewer.md §5). Unblock is **pre-registration-external** (requires editing the DB entry, not writing an impl).

No `mem-antipattern-impl-follow-up-delegation` obligation: that antipattern applies to novel-mechanism PROVISIONAL only. Preempt-structural KILL is explicitly excluded (canonical per F#687/F#698).

## §6 Scope integrity

No silent objective swap (antipattern-t): this scaffold does NOT:
- Substitute a runnable proxy-only verdict (e.g. measure cos-sim variance and claim SUPPORTED/KILLED on K1856 alone — that would be F#666-tautological).
- Invent a target-metric KC post-claim (that would be antipattern-u, post-claim KC mutation).
- Swap to a simpler measurement (e.g. cos-sim between two random prompts, not across 100 diverse) — would preserve the F#666 violation.

Pre-registered KC (K1856) is preserved verbatim and marked `untested (preempt-blocked)`. KC text in DB matches MATH.md §3 verbatim — no post-claim KC mutation.

## §7 New sub-case classification

Drain-window taxonomy after this iteration:

| Sub-case                                      | Parent status       | KC-structure        | Finding |
| --------------------------------------------- | ------------------- | ------------------- | ------- |
| F#669 classic (parent-unverified, F#666-ok)   | PROVISIONAL         | target-gated        | F#669/F#687/F#699 |
| F#669 + F#666 compound                        | PROVISIONAL         | proxy-only          | F#698   |
| **F#666-pure (this)**                         | **none / ok**       | **proxy-only**      | **this** |
| (runnable, F#666-compliant)                   | none / SUPPORTED    | target-gated        | regular KILL/SUPPORT |

This iteration fills the third row: standalone F#666-pure KC-structural KILL. If a second instance occurs, promote to standalone antipattern memory (`mem-antipattern-f666-pure-standalone-kc-structural-preempt` or similar). For now: 1st instance → document, watch, don't promote.
