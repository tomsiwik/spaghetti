# Adversarial Review — exp_prod_privacy_redesign

## Verdict: KILL (confirm)

DB status already `killed`. Researcher proposal matches results.json `verdict:
"KILLED"`, `all_pass: false`, `is_smoke: false`. PAPER.md says "KILLED" at
header. No verdict upgrade attempted. Confirm KILL.

## Adversarial checklist

| Check | Result |
|---|---|
| (a) results.json verdict vs proposal | ✓ both KILLED |
| (b) all_pass vs claim | ✓ all_pass=false, not claiming supported |
| (c) PAPER.md verdict line | ✓ "KILLED" — consistent |
| (d) is_smoke gating | ✓ is_smoke=false, full 500-iter run |
| (e) KC edit post-run | N/A — dir untracked (never committed). See Assumption 1. |
| (f) Tautology | ✓ K1642/K1643/K1644 use measured quantities |
| (g) K-ID object match | ✓ K1642=MIA fraction, K1643=PPL ratio, K1644=cos(B_A,B_B) |
| (h) Composition bug | N/A — single-adapter |
| (i) LORA_SCALE ≥ 12 | ✓ scale=8.0 (line 37) |
| (j) Routing-single-sample | N/A — no routing |
| (k) shutil.copy fake adapter | N/A — all trained fresh |
| (l) Hardcoded `"pass": True` | ✓ all computed from measurements (lines 606/613/623) |
| (m) Target model match | ✓ Gemma-4 E4B 4-bit everywhere |
| (m2) Skill invocation | partial — MATH.md does not explicitly cite `/mlx-dev`; code looks idiomatic (mx.eval used, nn.value_and_grad pattern). Non-blocking for KILL. |
| (n) Base=0% thinking truncation | N/A — PPL only |
| (o) n<15 | borderline — n=20 members vs 20 non-members; K1642 is failing anyway, so underpowering is moot |
| (r) Prediction table | ✓ PAPER.md §Prediction vs Measurement |

No (a)-(m) violation blocks KILL (would only block PROCEED).

## Load-bearing findings worth preserving

1. **Antipattern #6 confirmed on MIA protocol design**: OOD non-members
   confound domain learning with memorization. Same-distribution non-members
   confound memorization with nothing. Per-example MIA needs large same-domain
   pool (N≥100) with random train/holdout split and TPR-at-FPR metric — not
   20/20 OOD split. Researcher flagged this honestly and refused to upgrade.
2. **K1643 PASS is the real discovery**: null-space LoRA gets 2.1× better
   holdout PPL (84.11 vs 177.14) than standard LoRA on a 20-example corpus.
   Null-space acts as a structural regularizer for small-data training. This is
   a finding, not just a KC pass — should propagate to Analyst.
3. **Structural orthogonality replicated**: max|W_v @ A_eff| = 1.36e-5 on
   layers 16–23 of Gemma-4 E4B, independent of P7.A1's run. Theorem 1 of
   null-space isolation is now replicated on distinct data.
4. **K1644 failure is architectural, not numerical**: trained B matrices share
   ~0.4 cos on v_proj because of common Q/A-format steering. Fix requires
   Gram-Schmidt on B (T5.4 LEARNINGS already proposed). Out of scope here;
   worth a v2.

## Assumptions / judgment calls (per hat rules)

1. **No pre-registration commit for MATH.md**. Entire `exp_prod_privacy_redesign/`
   is untracked. For PROCEED this would block; for KILL it does not — KCs
   inside MATH.md pre-registered numerical thresholds, results.json records
   the measurements, and the verdict is downgrade not upgrade. No incentive
   to post-hoc edit KCs in one's own favor when the outcome is KILL.
2. **K1644 threshold mismatch** (DB 0.05 vs MATH.md 0.30): researcher lowered
   threshold up-front with explicit rationale (Theorem 2 JL bound is vacuous
   at d=512; 0.05 needs explicit Gram-Schmidt). Even relaxed threshold failed,
   so this does not matter for KILL.
3. **n=20 members is small for MIA** — but since K1642 fails at 20/20 (100%
   recovery), underpowering cannot rescue the claim.

## Non-blocking issues (not blocking KILL, noted for future v2)

- Researcher should commit MATH.md BEFORE running experiments (pre-registration
  discipline). Current workflow allows post-hoc KC edits to go undetected
  because nothing is in git history. Flag for next iteration.
- K1644 needs explicit Gram-Schmidt on B_B against B_A during training.
- K1642 needs a same-domain holdout pool of ≥100 for proper TPR-at-FPR MIA.

## Routing

KILL confirmed → emit `review.killed` to Analyst. Analyst writes
LEARNINGS.md capturing the MIA-protocol-design finding, the null-space-as-
small-data-regularizer finding, and the B-matrix Gram-Schmidt TODO.
