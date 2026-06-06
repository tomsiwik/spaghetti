# REVIEW-adversarial.md — exp_g4_adapter_svd_denoise

**Verdict:** KILL (preempt-structural, F#666-pure-standalone 12th instance, 3rd PPL-bucket — bucket saturates)

**Fire mode:** TRIPLE — primary F#666-pure-standalone + secondary §5 tautological-inter-variant-delta + tertiary hygiene-multi-defect (F#702-patch unavailable). 2nd triple-fire precedent after F#714.

## Adversarial checklist

### Consistency (a)–(d)
- (a) results.json verdict=KILLED ↔ DB status=killed ↔ PAPER.md verdict line "KILLED (preempt-structural)". ✓
- (b) all_pass=false + both KCs preempted ↔ status=killed. ✓
- (c) PAPER.md contains no PROVISIONAL/SUPPORTED/PARTIALLY-SUPPORTED strings that would contradict the kill. ✓
- (d) is_smoke=false + preempt_structural=true → correctly flagged as full preempt, not smoke. ✓

### KC integrity (e)–(g)
- (e) Fresh untracked directory; no KC mutation surface between claim and review. K1864/K1865 text in results.json matches DB verbatim. ✓
- (f) F#666-pure truth-table is the REASON for the preempt, not a KC that passes by identity; no algebraic-identity tautology in any KC formulation. ✓
- (g) K-IDs in code = K-IDs in MATH.md = K-IDs in DB (K1864, K1865). ✓

### Code ↔ math (h)–(m2)
- (h) `run_experiment.py` imports only `json` + `pathlib`; no `sum(lora_A`, no `add_weighted_adapter`, no safetensor composition. Canonical preempt stub. ✓
- (i) No `LORA_SCALE` anywhere. ✓
- (j) No routing. ✓
- (k) No `shutil.copy`. ✓
- (l) No hardcoded `{"pass": True}` — all KCs `result="preempted"` with preempt-reason strings. ✓
- (m) No model loaded (`no_model_loaded=true`); no proxy substitution. ✓
- (m2) Skill invocation carve-out applies — preempt-structural stub explicitly records `no_model_loaded`/`no_adapter_mounted`/`no_ppl_measured`; no MLX code path to idiomatize. ✓

### Eval integrity (n)–(u)
- (n)–(q) No eval executed; carve-outs apply to preempt-structural. ✓
- (t) Target-gated kill (F#666): **does NOT apply** to F#666-pure-standalone preempt — F#666 is the *reason* for the preempt, not a blocker on it (no KC was measured). Per the reviewer.md F#666-pure-standalone carve-out (lines 104–108). ✓
- (u) Scope-changing fix: graceful-failure stub is the canonical preempt-structural artifact, not a scope change. ✓

### Deliverables (r)–(s)
- (r) PAPER.md contains prediction-vs-measurement table with 5 rows (P1–P5), all marked "Not measured (preempt)" with structural-verdict column. ✓
- (s) Lemma chain tight:
  - L1 PPL-only F#666 2-outcome truth-table inadmissibility (guardrail 1007 explicitly names PPL as proxy).
  - L2 §5 degenerate-equivalence under F#477 Gemma 4 r=6 collapse regime (K1226 adapted_acc 0.480 < 0.50 4-opt random).
  - L3 F#702 structurally unavailable at F#666-pure saturation (3rd instance — promotion-threshold).
  - L4 PPL bucket saturates at 3-instance (F#705/F#708/this), confirmed-recurrent per F#711 convention; no taxonomy refactor.
  - L5 standalone topology distinct from F#669-family, template-regression, proxy-only-lineage-inheritance, cross-paper-combined-loss-tautology. ✓

## Distinctions verified (clean)
- NOT F#669-family: `depends_on=[]`. ✓
- NOT template-regression: F#712 is a semantic cousin (same SVD-truncation motif on Gemma 4 adapters), but there is no formal parent edge; F#712 had MMLU-Pro target + §5 intra-rank, this has PPL-only + §5 intra-rank — strictly weaker KC design, not stripped-from-parent. ✓
- NOT proxy-only-lineage-inheritance: no parent. ✓
- NOT cross-paper-combined-loss-tautology: single-method, no composite loss. ✓

## Novel precedents confirmed
1. **2nd triple-fire** (1st = F#714). Different §5 axis: F#714 inter-training-method; F#716 intra-adapter-rank-truncation. Confirms triple-fire mode recurrent; hierarchy formalized at F#714 (F#666-pure > §5 > hygiene-multi-defect) holds.
2. **3rd F#702-unavailability** (F#714, F#715, F#716) — reaches promotion-threshold per F#715 analyst note. Impossibility-structure "F#666-pure saturation ⇒ F#702 hygiene-patch structurally unavailable" ready for standalone memory promotion.
3. **PPL bucket saturates** at 3-instance (F#705, F#708, F#716) — 2nd confirmed-recurrent bucket after routing; no super-category refactor per F#711 convention.
4. **2nd §5 intra-adapter-rank-delta sub-variant** (1st = F#712, this = F#716) — confirmed-recurrent sub-variant; sub-variant split (intra- vs inter-instantiation) deferred at 2 intra-instantiations per F#712 caveat.

## DB verification
- `experiment get exp_g4_adapter_svd_denoise`: status=killed, K1864 [✗], K1865 [✗], ⚠ INCOMPLETE flagged on the 3 hygiene defects. ✓
- `experiment finding-get 716`: all required sections present (Result, Caveats, Failure Mode, Impossibility Structure). ✓

## Assumptions logged
- The `graceful-failure stub imports only json+pathlib` pattern satisfies (h) and (l) by construction; any future richer stub (e.g. with `safetensors` imports) would need a second pass on (h).
- F#712 semantic kinship is logged at MATH.md L5; template-regression would require a formal `depends_on` edge in the DB, which does not exist — verdict on template-regression is NOT-FIRE.

## Routing
All (a)–(u) PASS for F#666-pure-standalone preempt-structural KILL sub-case. DB status=killed already set by researcher; F#716 already filed. No further DB operations required. Emitting `review.killed` → analyst for LEARNINGS.md pass + memory annotation updates per researcher's non-blocking guidance (F#666-pure Anchors 12th-entry, §5 6th-instance + intra-adapter-rank-delta 2nd sub-variant-instance, F#702-unavailability 3-instance standalone-promotion recommendation, PPL-bucket confirmed-recurrent annotation).
