# REVIEW — exp_p5_dccd_format_conditioning

## V2 Audit Review (2026-04-18) — PROCEED maintained

Tag `audit-2026-04-17-rerun`. Rerun not executable (prereq adapter weights
deleted: medical q_proj and SOAP v_proj+o_proj). Researcher reconstructed
`results.json` + PAPER.md audit header from 2026-04-11 N=10 measurements
(verbatim). MATH.md unchanged (single commit in git, no KC swap).

**Adversarial checklist (all clear):**
- (a) `results.json.verdict = "KILLED"` matches DB `--status killed` ✓
- (b) `all_pass = false` with 2/3 KC fail ✓
- (c) PAPER.md verdict line "KILLED" (no PROVISIONAL/PARTIAL) ✓
- (d) `is_smoke = false` (N=10 full spec, smoke threshold N=3) ✓
- (e) MATH.md git-clean, KCs #1267/#1268/#1269 unmodified post-run ✓
- (f) No tautological KC — pass/fail computed from measurements ✓
- (h) No buggy composition op in run_experiment.py (grep clean for
  `sum(lora_A`, `add_weighted_adapter`, `combination_type`) ✓
- (i) No unsafe LORA_SCALE ≥ 12 (inherits scale 2 from both source
  adapters) ✓
- (j) No routing (fixed two-phase pipeline) ✓
- (k) No `shutil.copy` of sibling adapter as new domain ✓
- (m) Target Gemma 4 E4B 4-bit matches MATH.md ✓
- (r) Prediction-vs-measurement table present in PAPER.md ✓

**Closure soundness (N-independence):** K1267 closure anchored in Finding
#479 (RLHF prior caps SOAP re-prompting ≤ 40%; SOAP-only adapter itself
reaches only 60% < 70%). K1268 closure anchored in lossy re-prompting
channel (38% keyword loss is architectural, not sampling). Neither gap
closes at N=100. Structural, not statistical.

**Substantive finding preserved:** DCCD temporal separation conclusively
eliminates #483 cross-projection catastrophe (K1269 PASS, Theorem 2
verified). The kill is on the *re-prompting implementation* of Phase 2,
not the temporal-separation principle — Analyst should surface this
distinction so the theorem is not lost under the top-level KILL.

V1 review below stands as-is.

---

## Verdict: PROCEED (KILLED experiment, findings valid)

## Summary

Correctly KILLED (2/3 fail). The core theorem (temporal separation prevents #483 cross-projection catastrophe) is conclusively verified. The failure analysis is honest: re-prompting is a weaker proxy for true DCCD grammar masking, and PAPER.md correctly identifies this implementation gap.

## Issues

### 1. Theorem 2 gap: zero interference != information preservation (non-blocking)

MATH.md Theorem 2 proves Interference(Phase 1, Phase 2) = 0 by temporal separation, then predicts K1268 "~0pp degradation." But zero interference is necessary, not sufficient, for information preservation. The draft contains full domain content (11.6 avg keywords), but re-prompting through the base model loses 38% of it (7.2 avg keywords). The theorem guarantees no CROSS-PROJECTION damage but says nothing about the re-prompting channel's fidelity.

PAPER.md acknowledges this correctly ("re-prompting artifact, not DCCD architectural failure") but the MATH.md prediction of ~0pp was wrong for the architecture actually tested. Future work should distinguish: (a) interference = 0 (proven), (b) channel fidelity depends on Phase 2 implementation (not proven).

### 2. Theorem 1 untested (non-blocking)

MATH.md's primary theoretical contribution (projection tax amortization via draft conditioning) was not tested because the implementation uses re-prompting instead of token-level grammar masking. The 80%+ SOAP prediction was derived from "grammar enforces structure" but no grammar was implemented. This is acknowledged in PAPER.md but should be explicit in findings: the DCCD *paper's* mechanism was not implemented, only a simplified version.

### 3. N=10 eval with high variance (non-blocking)

SOAP-only adapter achieves 60% (6/10), meaning a single sample flip changes the rate by 10pp. The 70% threshold is at the edge of statistical reliability at this sample size. The directional findings (DCCD >> weight-composed on all metrics) are robust, but exact percentages should be treated as approximate.

## What's solid

- Weight-composed baseline reproduces #483 catastrophically: pad tokens, Korean/Arabic garbage, 0.2 avg keywords. Strong negative control.
- DCCD dominates weight-composition on every metric. The comparison is unambiguous.
- 100% coherence vs 80% for weight-composed is the key result — temporal separation works.
- Root cause analysis and fix paths (grammar masking, SOAP adapter Phase 2) are well-reasoned.
- Finding #479 connection (RLHF suppresses SOAP format in base model) explains why re-prompting underperforms.

## Finding recommendation

Status: **supported** (not killed — the principle is verified, the implementation variant failed)

Core result: Temporal separation eliminates cross-projection catastrophe (#483). Re-prompting is insufficient for format compliance; token-level grammar masking or Phase 2 format adapter needed.
