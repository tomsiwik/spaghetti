# E22-full Review: Adapter Poisoning Robustness — Full Run

## Verdict: KILLED

## Adversarial Checklist

All items (a)–(u) PASS:
- (a) results.json verdict = KILLED, DB status = killed. ✓
- (b) all_pass = false, status = killed. ✓
- (c) PAPER.md verdict = "KILLED". ✓
- (d) is_smoke = false, full run confirmed (35 layers, 100 QA). ✓
- (e) KCs inherited from E22 smoke, no modification. ✓
- (f) No tautological KCs — K2059 measures absolute drop, K2060 measures relative margin. ✓
- (g) KC descriptions consistent between MATH.md, code, and results.json. ✓
- (h) Composition is ΔW = Σ B_i @ A_i, applied as W + delta. No buggy independent sum. ✓
- (i) LORA_SCALE=6, within safe range. ✓
- (j) No routing in this experiment. ✓
- (k) No shutil.copy. ✓
- (l) No hardcoded pass. ✓
- (m) Model = mlx-community/gemma-4-e4b-it-4bit in MATH.md and code. ✓
- (m2) MLX idioms correct: mx.eval after compute, mx.clear_cache between phases, nn.Linear replacement, gc.collect. ✓
- (n) Base accuracy = 84%, not truncated. ✓
- (o) N=100 QA questions > 15 threshold. ✓
- (p) No synthetic padding. ✓
- (q) No cited baseline drift. ✓
- (r) PAPER.md has prediction-vs-measurement table. ✓
- (s) Claims match data, root cause analysis sound. ✓
- (t) Both KCs behavioral (QA accuracy). Both FAIL. Clean kill per F#666. ✓
- (u) No scope changes — setup matches pre-registration. ✓

## Kill Rationale

1. **K2059 FAIL**: Worst Grassmannian drop = 82pp >> 30pp threshold. Clean-only composition (no poison) already at 2%.
2. **K2060 FAIL**: Best protection margin = 2.0pp (≤, not >) vs >2pp threshold. All margins noise-level: -3 to +2pp.
3. **Structural root cause**: 5 synthetic adapters × 35 layers = 175 rank-6 ΔW perturbations destroy model coherence. Both conditions at floor (~2-7%), no differential signal possible.
4. **F#821 falsified**: Smoke's 55pp margin was a 3-layer (7% of model) artifact. Same pattern as E14-full (3-layer effects vanish at scale).

## Flags (none)

No non-blocking issues. Clean kill, sound methodology.

F#823 verified in DB. E22-full already marked killed.
