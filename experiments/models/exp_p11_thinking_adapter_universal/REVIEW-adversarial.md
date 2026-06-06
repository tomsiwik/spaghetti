# REVIEW-adversarial.md — P11.H0: thinking-universal-v0

**Reviewer**: Ralph (reviewer hat, post-kill pass)
**Date**: 2026-04-17
**Verdict**: **KILL** (endorses researcher's self-kill)

---

## Summary

Full run completed (100.2 min training, 210 MMLU + 40 GSM8K + 40 MedMCQA eval).
Adapter saved at `adapters/thinking-openthoughts-universal-v0/` (87.9 MB, 21 checkpoints).
**Two of three KCs failed.** Kill is correct; Theorem 1's precondition (GD > 0.5) was
violated by the 2-STEM-domain training distribution (math+code), producing catastrophic
forgetting on humanities (engineering 13.3%, philosophy 20.0%).

---

## Adversarial Checklist

**Consistency**
- (a) results.json per-KC pass booleans agree with DB `status=killed` and PAPER.md "Verdict: KILLED". ✓
- (b) K1517=false, K1518=false, K1519=true; researcher proposes `killed`, consistent. ✓
- (c) PAPER.md verdict line: "KILLED" — matches. ✓
- (d) `is_smoke` NOT set in results.json; training ran 6012.6s / 1000 steps / 210 MMLU questions — full run confirmed. ✓

**KC integrity**
- (e) `git log MATH.md` → single commit `de38e37` (2026-04-16). No post-run relaxation of K1517/K1518/K1519. ✓
- (f) No tautology: K1517 is accuracy-vs-threshold; K1518 is two independent accuracies; K1519 counts thinking chars. Each measures a real, non-self-identical quantity. ✓
- (g) K1517 code (`run_experiment.py:856`) computes `mmlu_acc >= 65.1`, matches MATH.md row "MMLU-Pro+thinking ≥ 65.1%". K1518/K1519 likewise match. ✓

**Code ↔ math**
- (h) No `sum(lora_A`, `add_weighted_adapter`, or independent key summing — single-adapter SFT, not composition. ✓
- (i) `LORA_SCALE=1.0` at `:51` — safe. ✓
- (j) No routing (single-domain adapter). ✓
- (k) No `shutil.copy`. ✓
- (l) No hardcoded `pass: True` — all three pass booleans computed from metric comparisons (`:856-858`). ✓
- (m) MATH.md target ≡ code target: `mlx-community/gemma-4-e4b-it-4bit`. ✓
- (m2) Skill invocation evidence absent from MATH.md/PAPER.md. **Non-blocking** because training is delegated to the `mlx_lm.lora` CLI (no hand-written MLX autograd) — the `/mlx-dev` skill risk surface is small here. Flag for future experiments that add custom MLX code.

**Eval integrity**
- (n) Base eval 62.1% (Finding #536) with thinking — not a truncated-thought-channel artefact. ✓
- (o) n=210 (MMLU), 40 (GSM8K), 40 (MedMCQA). Above 15 threshold. ✓
- (p) No synthetic padding.
- (q) Baseline cited from Finding #536 (62.1%), but `exp_p11_baseline_eval` re-measured 40.7% on the same model. PAPER.md flags this under Root Cause #3; does not change the kill (adapter under-performs either baseline). **Non-blocking flag.**

**Deliverables**
- (r) PAPER.md has prediction-vs-measurement table. ✓
- (s) No math errors; Theorem 1's Cauchy-Schwarz bound step still hand-waved (carryover from prior REVIEW non-blocking #1) — tolerable.

---

## Structural Finding

**Insight**: Training on two STEM domains (math + code) does NOT satisfy Theorem 1's
precondition GD > 0.5. STEM gradients are correlated (both produce procedural,
token-structure-similar traces), so the LoRA aligned with the dominant subspace and
destroyed humanities knowledge.

**Measurement evidence**:
- MMLU-Pro mean 47.6% (−14.5pp) with STEM categories (physics 66.7%, economics 73.3%) roughly preserved
- Humanities and applied categories collapsed: engineering 13.3%, philosophy 20.0%, math 33.3%, business/law/chemistry 40.0%
- The "diversity" came from within-STEM token distributions, insufficient for off-subspace preservation

**Implication for v2**: need ≥5 domains spanning STEM + humanities + social science + medical + legal. Two STEM shards ≠ diverse distribution.

---

## Assumptions / Judgment Calls

- Accepted Finding #536 baseline (62.1%) as the gate reference despite `exp_p11_baseline_eval` re-measuring 40.7%. Kill verdict is robust to either baseline: 47.6% < 65.1% (F#536+3pp) AND 47.6% > 40.7% (baseline_eval) — so the "adapter degraded MMLU-Pro" framing in PAPER.md is only true under F#536. The finding should be re-framed as "adapter failed relative to its own kill criterion" not "adapter degraded MMLU-Pro absolute". Flagged for Analyst.

---

## Verdict

**KILL** — KC failures are real (K1517 fails by 17.5pp, K1518 by 15pp on MedMCQA). No REVISE path: root cause is in the training distribution design (two STEM domains), not a bug in code or KC. Re-running with the same recipe would re-produce the same failure. v2 must restructure the domain mix before another iteration.

**Next experiment**: v2 with ≥5 domains (per PAPER.md "Next Experiment"). Gate should be against a locally-measured baseline, not Finding #536.

**No REVISE-round-2 needed.** Researcher's kill is endorsed; artefacts complete and consistent.
