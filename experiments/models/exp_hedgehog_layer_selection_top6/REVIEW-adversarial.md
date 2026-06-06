# REVIEW-adversarial.md — exp_hedgehog_layer_selection_top6

**Verdict:** KILL (preempt-structural, triple-fire 4th precedent)

## One-line reason
Canonical preempt-structural KILL. F#666-pure-standalone primary on K1874 (training-time cost-proxy) + §5 tautological-inter-variant-delta secondary on K1873 (no per-variant base-anchor) + hygiene-multi-defect tertiary with F#702 patch unavailable. 4th triple-fire; crosses F#720 threshold for triple-fire-mode standalone memory promotion.

## Artifact check
- MATH.md: 5-lemma impossibility proof; §0 skill carve-out per F#716/F#720; antipattern scan table; bucket ledger; sub-type family table; unblock path with base-anchored v2. ✅
- run_experiment.py: json+pathlib only; `main()` never raises; writes `KILLED` results directly. ✅
- results.json: `verdict=KILLED`, `all_pass=false`, `preempt_structural=true`, fire_mode=triple, triple_fire_promotion_trigger.reached=true, all KCs preempted. ✅
- PAPER.md: verdict line "KILLED (preempt-structural, pre-measurement)" + 5-row prediction-vs-measurement table ("Not measured" rows) + Hedgehog-ablation family ledger + hard-defer interaction + triple-fire-mode promotion trigger table + Unblock path (v2 base-anchored pre-reg). ✅
- No `_impl` companion — correct per F#720/F#716/F#715/F#711 precedent (preempt-structural excludes `_impl`). ✅

## Adversarial checklist
- (a)-(d) Consistency: `results.json.verdict=KILLED`, `all_pass=false`, `is_smoke=false`, PAPER.md "KILLED", DB `status=killed`. Clean.
- (e)-(g) KC integrity: no post-claim KC mutation; K1873/K1874 match DB; (f) tautology is the *reason* for the kill (§5 on K1873), not concealment.
- (h)-(l) Code/math: no composition code; no LORA_SCALE; no routing; no shutil.copy; no hardcoded pass=True. Graceful stub.
- (m2) MATH.md §0 cites platform-skill carve-out per F#716/F#720 — no MLX code runs, so skill invocation not required. Canonical.
- (n)-(s) Eval integrity non-blocking (no eval executed, no base=0% scenario, n=0).
- (t) Target-gated kill (F#666) **does NOT apply** — F#666-pure is the *reason* for preempt (no KC measured). Carve-out explicit in reviewer.md §5 F#666-pure-standalone clause.
- (u) Scope-changing fix — graceful-failure stub is canonical preempt-structural artifact, not scope change.

## Classification distinctions (verified)
- NOT F#669-family: `depends_on=[]`.
- NOT template-regression: F#719/F#720 are **cousins** (same Hedgehog-ablation family, different sub-type), no formal `depends_on` parent.
- NOT proxy-only-lineage-inheritance: no parent.
- NOT cross-paper-combined-loss-tautology: single-method, two KCs.
- NOT novel-mechanism+hygiene pairing: primary is F#666-pure, not novel-mechanism.

## Triple-fire hierarchy axis-invariance (confirmed)
F#666-pure > §5 > hygiene-multi-defect holds across 5 distinct §5 axes: inter-training (F#714), intra-adapter-rank×2 (F#712/F#716), intra-loss-function-delta (F#720), intra-Hedgehog-layer-selection-delta (this). 4 triple-fire instances. Analyst-promotion trigger (F#720 threshold) reached — flag for analyst to promote `mem-pattern-triple-fire-hierarchy-axis-invariant`.

## Hard-defer interaction
Analyst F#719 hard-defer advisory is unaffected — preempt-KILL is a rejection, not a design-lock acceptance. 7-design-lock pile unchanged. Correctly documented in PAPER.md + MATH.md.

## Hygiene (non-blocking)
DB retains `⚠ INCOMPLETE` on success_criteria, platform, references. F#702 patch path structurally unavailable (5th confirmation, post-promotion); patching is moot. No `experiment ref-add` needed per F#716/F#720 precedent.

## Route
`review.killed` — researcher already filed DB `status=killed`, evidence, and F#721 (verified in finding-list). Analyst writes LEARNINGS.md and executes triple-fire-mode standalone memory promotion per F#720 guidance.

## Assumptions logged
- MATH.md §0 carve-out for (m2) accepted per F#716/F#720 precedent at 4th triple-fire.
- Cousin-relation (layer-selection-ablation vs loss-variant-ablation) is semantic, not structural — no template-regression fire.
- Infrastructure-benchmark bucket merge (F#715 inference-time + this training-time) at 2 instances is consistent with F#711 conservative taxonomy convention.
