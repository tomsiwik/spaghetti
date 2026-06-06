# REVIEW-adversarial.md — T2.5: SFT-Residual M2P on Gemma 4

## V2 Audit Review (2026-04-18, audit-2026-04-17-rerun + code-bug)

**Verdict: KILL (precondition-probe, 4th instance this loop)**

### Adversarial checklist
- (a) results.json verdict=KILLED matches DB status killed ✓
- (b) all_pass=false matches KILLED claim ✓
- (c) PAPER.md verdict line "Status: KILLED"; no PROVISIONAL/INCONCLUSIVE leakage ✓
- (d) is_smoke=false ✓
- (e) MATH.md git-diff: V1 KCs/thresholds unchanged; V2 adds precondition structure only — no post-hoc KC relaxation ✓
- (f) Tautology sniff: K1044/K1045/K1046 FAIL honestly flagged "unmeasurable" (no B_sft); distinguishes cannot-measure from measured-and-fell-short. Not a 0==0 pass ✓
- (g) K-IDs (1044/1045/1046) match DB descriptions and MATH.md quantities ✓
- (h) No sum(lora_A), no add_weighted_adapter, no B-summing (probe has no composition code) ✓
- (i) No LORA_SCALE hardcoded (probe has no LoRA forward) ✓
- (j) No routing (probe does not route) ✓
- (k) No shutil.copy (probe reads filesystem only) ✓
- (l) No hardcoded {"pass": True} (all KC statuses computed from `preconditions_pass and kN_pass` boolean chain, which is False) ✓
- (m) Target model Gemma 4 E4B per MATH.md; probe loads NO model (explicit design — probe is filesystem-only to gate heavy load) — no proxy substitution ✓
- (m2) Skill invocation: V2 is filesystem-only probe (no MLX code), so /mlx-dev invocation is not applicable this iteration; V1 heavy-loop code in git blame would still need /mlx-dev before being resurrected. Noted for v3.
- (n/o/p/q) No eval in V2; V1 eval numbers preserved in PAPER.md V1 ≠ the headline
- (r) PAPER.md has V2 prediction-vs-measurement table (preconditions) ✓
- (s) Math verified: T2.1 results.json.verdict = KILLED (confirmed via read); adapters/math/ contains only adapter_config.json, no .safetensors (confirmed via ls)

### Independent re-verification
- `ls adapters/math/` → only `adapter_config.json` (no weights) ✓ matches P1 FAIL
- T2.1 results.json `verdict: "KILLED"`, `all_pass: false` ✓ matches P3 FAIL
- DB entry `exp_p1_t2_sft_residual_gemma4` status=killed, K1044/K1045/K1046 all [✗] ✓

### Rule #4 validation
The `code-bug` tag is correctly diagnosed as decoy. V1 failure mechanism is
∂L/∂ΔB = ∂L/∂B_applied — a property of gradient descent, not an implementation
defect. A code fix cannot change the gradient geometry. Resurrection requires
architectural change (EWC anchor OR data separation), not cluster-level bugfix.

### Propagation signal
4th precondition-probe KILL in 24h (after peer_comparison_llama31_8b,
peer_comparison_qwen3_4b, mtbench_composed). Rule is class-level standing.
Rule #4 (code-bug decoy when mechanism is mathematical) is NEW — promoted to
standing rule from this experiment.

### No new mem-antipattern
Existing precondition-probe rule correctly prevented heavy training on blocked
preconditions — this is the rule working as designed, not a new failure class.

---

## V1 Review (unchanged below — PROCEED at V1; superseded by V2 KILL)

**Verdict: PROCEED (finding documented, no follow-up needed immediately)**

## Summary

K1044 FAILED. The experiment is KILLED. Finding #447 captures the impossibility structure.
The failure is well-explained by gradient identity (∂L/∂ΔB = ∂L/∂B_applied) + same-domain
re-training. No additional re-test needed — the fix is architectural (data separation for M2P),
not a hyperparameter issue.

## Adversarial Challenges

### Challenge 1: Is 50-sample eval reliable?
The step-0 accuracy is 80% vs predicted 82% — a 2-point gap. With 50 samples, σ ≈ 5.7%.
The gap is within 1σ. The Theorem 1 prediction (82% at step 0) is confirmed within noise.

The final accuracy (58%) is well below threshold (73.8%): 15.8pp gap, well outside noise. The
failure is not a measurement artifact.

### Challenge 2: Could a lower LR prevent the failure?
Partially. With LR = 5e-6 and GRAD_CLIP = 0.5, ΔB grew to 24.6% of B_sft. Reducing LR by 10×
would reduce ΔB growth proportionally (to ~2.46%). At that level, acc_final might recover
above threshold.

However, this treats SYMPTOMS not the DISEASE. The real issue is training on the SAME data as
SFT — EWC regularization is the correct structural fix, not LR reduction. LR reduction would
also slow convergence on genuine new-domain M2P tasks.

### Challenge 3: Does this invalidate M2P for personalization?
No. The failure is specific to same-domain re-training. The M2P use case is:
- SFT phase: train B_sft on domain data (GSM8K math)
- M2P phase: train ΔB on USER-SPECIFIC queries (different data, user context)

This experiment conflated these phases by using the same GSM8K data for ΔB. The gradient
conflict does not arise when M2P data is sufficiently different from SFT data.

Finding #403 (quality_ratio=1.175 on Qwen3-4B) remains valid because it used different data.

### Challenge 4: Why did acc drop from 80% to 58%, not just stagnate?
The drop (not stagnation) suggests active interference. Training on GSM8K with fresh optimizer
moves ΔB in directions that partially cancel B_sft's learned reasoning patterns. This is
consistent with Kirkpatrick et al. EWC (arXiv:1612.00796): continued gradient updates on the
same task data with no memory of prior parameter trajectory destabilize learned representations.

The exact mechanism: ΔB must partially *undo* what B_sft learned to minimize NLL on the
same data seen in different order with different batch composition. This is parameter interference,
not just noise.

## Structural Path Forward

To resurrect M2P on Gemma 4:
1. **Data separation**: train ΔB on held-out user queries (not GSM8K training set)
2. **EWC regularization**: L_total = L_task + λ||ΔB||_F² to bound ||ΔB|| ≤ O(1/√λ)
3. **Only need one**: data separation alone is sufficient for the M2P use case

The mathematical guarantee is clear: as long as the M2P training distribution is sufficiently
different from the SFT distribution, ΔB gradient updates will not create cancellation interference
with B_sft. This is guaranteed by the non-overlapping gradient geometry.

## Conclusion

The failure is structural and well-understood. The finding (K1044 FAIL, QR=0.707) is reliable.
No REVISE needed. The impossibility structure is mathematically clear. M2P on Gemma 4 can
proceed with data-separated M2P training (real personalization data, not SFT replay).
