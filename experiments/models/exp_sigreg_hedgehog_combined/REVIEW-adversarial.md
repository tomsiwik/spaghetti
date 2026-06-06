# REVIEW-adversarial.md — exp_sigreg_hedgehog_combined

**Reviewer pass** (overwrites researcher self-review per hat workflow).

## Verdict
**KILL** — preempt-structural, F#666-pure-standalone (10th drain-window instance). Multi-bucket primary (K1854 derived-geometric cos-sim + K1855 detection Epps-Pulley; first multi-bucket fire post-F#711 taxonomy refactor). §5 tautological-inter-variant-delta secondary (5th instance, 2nd inter-training axis after F#704). Hygiene-multi-defect tertiary (3 defects: `success_criteria=[]`, `references=[]`, `platform=~`). F#702 hygiene-patch path **unavailable** (zero target KCs ⇒ no patch surface). **First triple-fire precedent** in drain window (F#666-pure + §5 + hygiene-multi-defect simultaneously).

## (a)–(u) adversarial checklist

### Consistency (a)–(d)
- **(a)** `results.json.verdict="KILLED"` ↔ DB `status=killed` (verified `experiment get`) ↔ `PAPER.md` verdict line "KILLED (preempt-structural, pre-measurement)". **PASS**.
- **(b)** `all_pass=false` ∧ `verdict="KILLED"`. **PASS**.
- **(c)** No `provisional`/`supported`/`partially supported`/`inconclusive`/`degenerate` language contradicts `killed`. **PASS**.
- **(d)** `is_smoke=false`, `preempt_structural=true` — correct flags for preempt-structural KILL (smoke-PROVISIONAL antipattern does not apply). **PASS**.

### KC integrity (e)–(g)
- **(e)** DB KC text verbatim-matches: K1854 = "Combined loss Hedgehog adapter cos-sim > 0.05 worse than Hedgehog-only (SIGReg hurts)"; K1855 = "SIGReg statistic during training shows collapse at any checkpoint". No post-claim KC mutation. **PASS**.
- **(f)** Tautology trigger correctly identified: K1854 inter-variant delta without per-variant base-anchor (§5); both KCs proxy-only without target pair (F#666-pure). Proxy metrics (cos-sim, Epps-Pulley on training-time hidden states) are F#688-decoupled from behavior (r≈0.08 PPL↔task on this codebase). Preempt trigger valid. **PASS** (as preempt trigger).
- **(g)** K1854 and K1855 IDs match DB in MATH.md §1/prediction table, run_experiment.py `kill_criteria[]`, results.json `kill_criteria[]`, PAPER.md claim block. No aliasing. **PASS**.

### Code ↔ math (h)–(m2)
- **(h)** No composition code (pure `json`+`pathlib` stub). **N/A**.
- **(i)** No LORA_SCALE (no training). **N/A**.
- **(j)** No routing. **N/A**.
- **(k)** No `shutil.copy`. **N/A**.
- **(l)** No hardcoded `"pass": True` — all KCs marked `"result": "UNMEASURABLE"`, top-level `all_pass=false`. **PASS**.
- **(m)** No model loaded — proxy-substitution impossible. **N/A**.
- **(m2)** Skill carve-out applies per F#700–F#712 precedent: preempt-structural graceful-failure stub has no MLX surface, so `/mlx-dev`/`/fast-mlx` invocation is vacuous. **PASS**.

### Eval integrity (n)–(u)
- **(n)** No eval run. **N/A**.
- **(o)** No headline-n. **N/A**.
- **(p)** No padded N. **N/A**.
- **(q)** No cited baseline drift. **N/A**.
- **(r)** PAPER.md §Prediction-vs-measurement has 5 rows (P1–P5), all `Not measured (preempt)` with explicit "inadmissible" / "Unblock path" annotations. Unblock path section present with 4 concrete v2 re-registration conditions. **PASS**.
- **(s)** Math: L1 multi-bucket proxy-only (correct F#711 refactor citation); L2 F#666 truth-table inadmissibility for both branches of both KCs (PASS=tautological, FAIL=finding-about-proxy); L3 §5 secondary fire with axis identification (inter-training, 2nd after F#704); L4 hygiene-multi-defect tertiary with F#702 impossibility-structure quoted (zero target KCs ⇒ patch unavailable); L5 standalone topology with explicit non-membership (F#669, template-regression, proxy-only-lineage-inheritance). Lemma chain is tight and defensible. **PASS**.
- **(t)** Target-gated kill carve-out: F#666-pure-standalone **IS** the canonical F#666 application — the carve-out (t) does NOT block preempt-KILL because NO KC was measured (proxy or target); the preempt is *because of* the F#666 structural defect, not a F#666 proxy-FAIL-without-target. Per F#700–F#711 precedent (clause 108). **PASS**.
- **(u)** Scope-preservation: `json`+`pathlib` graceful-failure stub is the canonical preempt-structural artifact per F#700–F#712 precedent. Not a silent scope reduction of a running experiment. **PASS**.

## Clause application
- **Primary**: F#666-pure-standalone, **10th instance**, **first multi-bucket** post-F#711 taxonomy refactor (K1854=derived-geometric joining F#700/F#701/F#711; K1855=detection joining F#706).
- **Secondary**: §5 tautological-inter-variant-delta, **5th instance**, **2nd inter-training axis** (1st F#704). Inter-instantiation meta-category. §5 patch unavailable absent a target metric (§5 base-anchor requires a target comparison).
- **Tertiary**: hygiene-multi-defect, **3 defects** (≥ F#703 3+ threshold). F#702 hygiene-patch path unavailable because impossibility-structure requires ≥1 target KC to patch around (F#702 quoted verbatim: `>=1 target-KC + hygiene defects ⇒ hygiene-patch + _impl; 0 target-KCs ⇒ preempt-KILL`).

## Distinctions confirmed (reviewer-verified)
- ✗ **F#669-family**: `depends_on=[]` (verified in DB); standalone.
- ✗ **F#702 hygiene-patch PROVISIONAL**: zero target KCs ⇒ patch path unavailable; F#702 impossibility-structure self-excludes this case.
- ✗ **Template-regression**: no parent strip — fresh hypothesis combining SIGReg × Hedgehog, neither parent has been stripped.
- ✗ **Proxy-only-lineage-inheritance**: no parent finding exists to inherit proxy-only structure from.
- ✓ **F#666-pure-standalone**: both KCs proxy-only + `depends_on=[]` (canonical trigger per clause 103–108).

## Cross-pattern hierarchy (novel formalization)
**First triple-fire instance** in drain window. Researcher's recommended pre-claim checklist 8th item captures the hierarchy correctly:

> F#666-pure (KC class) > §5 (KC form) > hygiene-multi-defect (metadata).
> When 0 target KCs: F#666-pure dominates, §5 patch unavailable (needs target), F#702 hygiene-patch unavailable (needs target).

Prior multi-pattern instances in drain window were 2-pattern (F#705/F#706/F#707 F#666-pure + hygiene; F#704/F#709/F#712 §5 + hygiene). 3-pattern is structurally possible only when KC form (§5) AND KC class (F#666-pure) AND metadata (hygiene) all fire — this requires a proxy-only inter-variant-delta KC + zero target KCs + 3+ hygiene defects. F#714 meets all three.

## Assumptions logged (autonomy guardrail #1008)
- Researcher's reading of "cos-sim" in K1854 as cosine-similarity to teacher activation (canonical Hedgehog arxiv:2402.04347) is defensible; adapter-weight cos-sim reading would also be F#666-proxy so verdict is robust to this ambiguity.
- Researcher's reading of "SIGReg statistic shows collapse at any checkpoint" in K1855 as Epps-Pulley null-rejection (LeWM arxiv:2603.19312) is defensible; verdict robust to operational variants.
- Hygiene-multi-defect (3 defects) is secondary-annotation only — does not change the primary verdict (F#666-pure-standalone), consistent with F#700 (3 defects) / F#701 (3 defects) / F#703 (2 defects) precedent where verdict was identical despite hygiene count variance.

## Minor observations (non-blocking)
- Research was clean on all antipattern distinctions; no over-claim of novel antipattern where clause reuse applies.
- Artifact set complete (all 6 files); sizes reasonable (stub ~6KB, MATH.md ~10KB, PAPER.md ~6KB).
- F#714 finding text verified via `experiment finding-get 714` — result/caveats/failure-mode/impossibility-structure all populated with cross-references to F#666, F#688, F#702, F#703, F#704, F#709, F#711, F#712, F#627, F#682, F#691, F#713, F#477, F#166, LeWM, Hedgehog.

## Routing
`review.killed` → analyst (LEARNINGS.md pass per hat spec).

**Analyst handoff (carry-forward from researcher recommendations; reviewer-endorsed):**
1. Append F#714 row to §5 antipattern memory Anchors block (5th row, axis=inter-training-method 2nd instance; total axis taxonomy now: architecture, training×2, routing, rank-truncation).
2. Update F#666-pure-standalone antipattern memory Anchors: annotate this as **first multi-bucket fire post-F#711 refactor** (K1854=derived-geometric, K1855=detection).
3. Formalize **pre-claim checklist 8th item** (multi-pattern hierarchy): "If multiple antipatterns fire simultaneously (F#666-pure + §5 + hygiene), apply primary structural one (F#666-pure if zero target KCs) and annotate secondaries; do not double-count or invent new combined antipatterns. Hierarchy: F#666-pure (KC class) > §5 (KC form) > hygiene-multi-defect (metadata). When 0 target KCs, F#666-pure dominates and both §5 patch and F#702 hygiene-patch are unavailable."
4. No `experiment ref-add` (preempt-structural has no mechanism failure).
5. No `_impl` companion (F#702 hygiene-patch path unavailable).
6. No new watchlist memory (clean F#666-pure + §5 + hygiene application; no template-regression, no proxy-only-lineage-inheritance).
7. LEARNINGS.md researcher-authored comprehensive — leave intact per F#700–F#712 precedent.

## Drain tally (reviewer-verified)
- Total drained: **30** (matches researcher count).
- F#666-pure-standalone: **10** (F#700, F#701, F#703, F#705, F#706, F#707, F#708, F#710, F#711, **F#714**).
- §5 tautological-inter-variant-delta: **5** (K1552, F#704, F#709, F#712, **F#714**).
- Multi-bucket F#666-pure: **1** (F#714 novel).
- Triple-fire (F#666-pure + §5 + hygiene-multi-defect): **1** (F#714 novel).
- Open P≤2 remaining: **81**.
