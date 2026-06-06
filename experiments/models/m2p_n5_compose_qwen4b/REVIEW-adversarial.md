# REVIEW-adversarial.md — exp_m2p_n5_compose_qwen4b

**Verdict: PROCEED**

---

## Round 2 Review (REVISE resolved)

Both blocking fixes from Round 1 applied:
- Fix 1 (PAPER.md): Present and complete with prediction-vs-measurement table ✓
- Fix 2 (Full run): n_eval_math=200, synth_train_steps=100, all K PASS ✓
- Non-blocking concern (sort_m2p_loss): sort=0.798, reverse=0.898, count=1.160 — all > 0 ✓

---

## Kill Criteria Verification

| K | Threshold | Measured | Status |
|---|-----------|----------|--------|
| K978: max\|A_i^T A_j\|_F (10 pairs) | < 1e-4 | 1.38e-05 | **PASS** |
| K979: TF-IDF 5-class routing | ≥ 80% | 100% (500/500) | **PASS** |
| K980: math quality_ratio | ≥ 0.70 | 1.3125 | **PASS** |

All evidence in results.json consistent with PAPER.md claims.

---

## Adversarial Checks

**Theorem 4 is a structural tautology (acceptable):** The proof that N=5 routed quality equals N=2 holds because exclusive routing means only one adapter fires. The "theorem" is trivially true by construction. This is not a flaw — the real empirical content is K978 (isolation) + K979 (routing) together making the tautology load-bearing. If either failed, Theorem 4 would not apply.

**K980 only measures math quality:** Sort/reverse/count adapters are synthetic and their quality is not independently measured. The M2P losses (0.80–1.16) suggest they converge but task accuracy isn't verified. This is acceptable for this experiment scope — routing accuracy (K979: 100%) is the binding guarantee.

**peak_memory_gb = 0.0:** Reporting bug in run_experiment.py — memory tracking not implemented. Non-blocking; doesn't affect scientific validity.

**base_accuracy=0.65, sft_accuracy=0.73 inherited from Finding #404:** Not re-measured at n=200. The quality_ratio calculation depends on these values. Since these are stable properties of the model (not M2P artifacts), inheriting is acceptable — but a caveat in PAPER.md would improve rigor.

---

## Non-Blocking Observations

- Finding status "supported" is defensible; "conclusive" would also be justified given exact prediction match (qr=1.3125 = predicted ≈ 1.31). No change required.
- The progressive isolation pattern (math×code at 1e-05, synthetic pairs at sub-1e-09) confirms sequential Gram-Schmidt works as designed — this is a useful empirical characterization beyond the kill criteria.
- N_max=640 at d=2560, r=4 is correctly stated as structural (no empirical burden).

---

## Verdict

**PROCEED** — all REVISE fixes applied, all K criteria verified, PAPER.md complete with prediction-vs-measurement table. Math is sound, evidence matches claims. Finding #405 "supported" is appropriate.
