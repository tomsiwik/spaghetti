# REVIEW-adversarial.md — P11.J0: Adapter Composition via Exclusive Routing

**Reviewer**: Adversarial Reviewer (post-run)
**Date**: 2026-04-18
**Verdict**: **KILL** (endorse researcher determination)

> Supersedes 2026-04-14 PROCEED. That review correctly anticipated K1527 FAIL
> via Finding #517 but missed two facts: (a) all 4 domain-adapter directories
> are weight-less stubs (no `adapters.safetensors`), and (b) the upstream H0
> "thinking" adapter would land regressed. Both materialised at runtime.

---

## Adversarial checklist

**Consistency (a–d):** all clean
- (a) `results.json["verdict"] = "KILLED"` ↔ DB `status=killed` ↔ PAPER.md §1 KILLED.
- (b) `all_pass=false`, status killed.
- (c) PAPER.md verdict line consistent.
- (d) `is_smoke=false`.

**KC integrity (e–g):** all clean
- (e) `git log MATH.md` → single commit `de38e37`. No post-data KC drift.
- (f) K1528 is direct measurement (187/280 = 0.668 vs 0.85 pre-reg). No tautology.
- (g) K-IDs 1526/1527/1528 in code (`run_experiment.py:659-661`) match DB and
  MATH.md descriptions.

**Code ↔ math (h–m2):** all clean
- (h) Exclusive routing partitions queries; no `add_weighted_adapter`, no
  summed `lora_A`/`lora_B`. Each phase loads exactly one adapter via
  `mlx_lm.load(..., adapter_path=...)` (L283).
- (i) No `LORA_SCALE` (inference-only experiment).
- (j) Routing is per-question: `route_query(row["question"], ...)` inside
  `for cat, row in eval_questions` (L553-556). No per-batch contamination.
- (k) No `shutil.copy`.
- (l) No hardcoded `{"pass": True}`. KC dict computed from measurements
  (L688-690).
- (m) `MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"` (L48); MATH.md predicts
  Gemma 4 baselines. Same model in calibration + eval.
- (m2) MLX skill invocation not explicitly cited in MATH.md/PAPER.md, but
  inference-only code uses `mx.eval` (L204, L226), `mx.clear_cache` via
  `cleanup()` helper, and the MLX-correct `embed.weight` access pattern.
  Researcher fixed a pre-run bf16→numpy bug at L207/L227 by inserting
  `.astype(mx.float32)` casts. Non-blocking.

**Eval integrity (n–q):** non-blocking
- (n) Phase 3 thinking_only `n_questions=80`, `avg_thinking_chars` not zeroed
  (researcher reports 2286s wallclock for 80 questions → ~28.6s/q, consistent
  with 1k+ token reasoning). Not the B0/H1 truncation pattern.
- (o) K1528 n=280 (router accuracy); Phase 3 n=80. Adequate.
- (p) No synthetic padding.
- (q) MATH.md L44 cites Finding #530 base=62.1%; F#560 reconciliation thread
  measures 40.7%. Non-blocking because all KCs are relative deltas, not
  absolute thresholds. Flagged for analyst.

**Deliverables (r–s):**
- (r) PAPER.md §"Theorem Predictions vs Measurements" present (L8-14).
- (s) Math claims hold; researcher correctly notes Theorem 1 premise
  inversion (acc_t reasoning 0.375 < knowledge 0.725, attributable to H0
  regression) is a structural failure, not a noise-driven near-miss.

---

## Independent verification of kill drivers

1. **K1528 direct measurement**: PAPER.md §"Phase 2: Router accuracy"
   reports 187/280 = 0.668. results.json corroborates. 18.2pp below the
   pre-registered 0.85 threshold from MATH.md L75. No path to K1528 PASS
   without redesigning the router (later-layer hidden states or learned
   classifier).
2. **K1526/K1527 untestable**: `ls` of all 4 referenced domain-adapter dirs
   confirms only `adapter_config.json` present (no `adapters.safetensors`,
   no `*.npz`). The crash at "Phase 4 domain_math" is genuine infra
   blocker, not a code bug. Even if fixed, Finding #517 predicts domain
   adapters degrade MCQ — likely vacuous K1526 PASS / K1527 FAIL with no
   composition insight.
3. **Theorem 1 premise inversion**: Phase 3 measured 0.375 (reasoning)
   vs 0.725 (knowledge) on the H0 thinking adapter. MATH.md L24 assumes
   `acc_t(P_r) ≥ acc_t(P_k)`. Inversion magnitude (35pp) puts this beyond
   noise — H0's same-day kill (47.6% MMLU-Pro, humanities collapse)
   provides a coherent mechanism.

Cross-experiment context: 8th P11 kill in this chain (F0/H0/B0/C0/D0/H1/I0/J0).
J0 is the **first measurement-based kill** of the chain (K1528 ran and
failed); the prior 7 were preemptive on protocol-bug recurrence
(antipattern-018) or upstream cascade. J0 also tests a different
mechanism (routing, not training) and so the kill is independent evidence
that the broader thinking-adapter line of work needs the P11.HARNESS
unblock + adapter retrains before composition is meaningfully measurable.

---

## Assumptions

- "Untestable counts as FAIL" for K1526/K1527: pre-registration says the
  KCs require a measured comparison; no measurement → cannot satisfy.
  Reviewer accepts researcher's judgment. Alternative would be `unknown`
  status, but DB schema treats KCs as binary and this experiment cannot
  produce the data within its claim scope.
- F#560 baseline drift (62.1% vs 40.7%) is non-blocking for this kill
  because all 3 KCs are relative; flagged separately.
- Researcher's pre-run fix (L207/L227 `.astype(mx.float32)`) is not in git
  diff against pre-claim state — accepting researcher's note that this
  was needed for `np.array(...)` to work on bf16 tensors. Non-blocking
  since it's a numerical/dtype fix that doesn't touch KC logic.

---

## Findings / DB writes

DB already `status=killed` from researcher with `--k 1526:fail --k 1527:fail
--k 1528:fail`. **No additional DB writes needed.**

**No new finding** added. Mechanism is composition of:
- Finding #517 (domain adapters degrade MCQ)
- Finding #527 (pre-merge killed → exclusive routing motivation)
- Today's H0 kill (regressed thinking adapter)
- antipattern-018 (cascade-kill driver in B0 → propagates as adapter
  unavailability / regression for all downstream composition work)

---

## Open threads for Analyst / successor

1. **P11.HARNESS** is now the atomic unblock for the entire thinking-adapter
   composition line (J0-v2, M0, K1527 retest with non-regressed adapter).
2. **Adapter-infra audit** — 4 of 4 referenced domain adapters are stubs.
   The training scripts in `exp_p1_t2_single_domain_training/` and
   `exp_p1_t2_multi_domain_5/` either never produced safetensors or had
   them deleted during cleanup. Worth a one-shot audit of all
   `adapter_config.json` files in `micro/models/**/adapters/` to flag
   weight-less stubs before they're cited as dependencies. Candidate for
   new antipattern if it recurs (currently 1 instance).
3. **Router redesign** — 66.8% centroid accuracy on token-mean embeddings
   is a meaningful negative result independent of the H0 cascade. Suggests
   raw `embed_tokens` mean is too lossy; future work should try later-layer
   hidden states (one forward pass to layer ~10) or a learned linear head.
4. **F#560 baseline reconciliation** still open — blocks any "≥ X%
   absolute" KC design across the P11 chain.
5. **First measurement-based kill in P11.X chain** — analyst should
   distinguish in LEARNINGS between protocol-bug preemptive kills (B0
   chain) and this measurement-fail kill, since the research lessons
   differ (former: shared training harness; latter: weak router +
   regressed dependency).
