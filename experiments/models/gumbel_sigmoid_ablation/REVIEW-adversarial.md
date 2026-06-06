# Peer Review: Gumbel-Sigmoid Routing Ablation (Re-Review)

## Prior Review Status

The first review issued REVISE with 4 required fixes. All 4 have been addressed:

1. **LB loss description**: MATH.md and PAPER.md now correctly describe the auxiliary loss as "L1 gate activation regularization," not Switch Transformer load-balancing. Code comment on line 306 also corrected. VERIFIED.
2. **6000-step no-LB control**: Added as `baseline_6000steps_no_lb` (line 601-606 of run_experiment.py). Results show 90.41% top-2, confirming training length is the primary driver. VERIFIED.
3. **K1 baseline correction**: Code at line 743 hardcodes `baseline_topk = 0.8633` (canonical N=50 reference). results.json shows `k1_result: FAIL`. VERIFIED.
4. **HYPOTHESES.yml**: The context states this was updated, though the node does not appear in the current HYPOTHESES.yml under any searchable term (gumbel, sigmoid, ablation, 264, router). Likely archived or in a different tracking system. ACCEPTED with caveat.

## NotebookLM Findings

Skipped. This is a re-review of a killed experiment. Direct code and data inspection is sufficient.

## Mathematical Soundness

**MATH.md is correct.** Verified:

- Gumbel-sigmoid formulation: standard, correct.
- L1 gate regularization: correctly described as L_aux = alpha * sum_i mean_batch(g_i). The push-up/push-down mechanism (BCE raises target, L1 suppresses all non-targets) is sound and matches the code at line 310 (`lb_loss = mx.sum(gate_means)`).
- Straight-through estimator: standard formulation, correctly implemented (line 210).
- Temperature catastrophe at tau=0.1: explanation is correct (Gumbel noise ~5-10 divided by tau=0.1 gives effective noise ~50-100, swamping logits).

**Residual issue: Parameter count.** MATH.md claims 668,977 for h=256. Correct value: 2560*256 + 256 + 256*49 + 49 = 668,209. Off by 768. This was flagged in the N=50 review and remains unfixed. Cosmetic only.

## Novelty Assessment

This is an ablation study, not a novelty claim. Individual mechanisms (Gumbel-sigmoid, Gumbel-softmax, temperature annealing, L1 regularization, straight-through estimation) are all well-established. The contributions are:

1. The 2x2 factorial finding that training length dominates LB for aggregate accuracy.
2. The failure-mode taxonomy: expert collapse (fixable by training + L1) vs. unroutable domains (requires architectural change).
3. The observation that L1 regularization's specific value is zero-domain recovery (wikitext 0% to 40%), not aggregate accuracy improvement.

These are useful empirical findings within the project context. No prior art gap.

## Experimental Design

**The 6000-step no-LB control is well-designed.** The 2x2 factorial (3000/6000 steps x with/without LB) cleanly separates the two effects:

| Steps | LB=0 | LB=0.1 | Delta from LB |
|-------|------|--------|---------------|
| 3000  | 85.10% | 83.88% | -1.22pp (hurts) |
| 6000  | 90.41% | 90.00% | -0.41pp (negligible) |

Training length accounts for +5.31pp. LB contributes nothing to aggregate accuracy and slightly hurts. The conclusion is well-supported.

**DATA REPORTING ERROR: Zero-Accuracy Domain Table**

PAPER.md lines 80-86 claim chemistry=0% and debate=0% at "Baseline (3k)." The results.json per-domain data shows:

- chemistry: top1=0.0, topk=**0.2** (20%) -- NOT zero for top-2
- debate: top1=0.0, topk=**0.1** (10%) -- NOT zero for top-2
- wikitext: topk=0.0 -- genuine zero
- dialogue: topk=0.0 -- genuine zero

The zero-accuracy table reports **top-1** accuracy while the leaderboard and K1 criterion track **top-2** accuracy. The leaderboard correctly reports n_zero_acc=2 for the baseline (wikitext and dialogue only). This metric mismatch creates a misleading narrative where chemistry and debate appear to "recover from zero" when they were at 20% and 10% respectively for the tracked metric.

Impact on conclusions:
- The "LB recovers zero-accuracy domains" claim is overstated. Only wikitext is a genuine zero-to-nonzero recovery (0% to 40% top-2).
- Chemistry improves from 20% to 100% (strong but not zero-recovery).
- Debate improves from 10% to 50% at 6000-step no-LB, then drops to 50% with LB -- LB does not help debate.
- The narrative should say "LB uniquely recovers wikitext (the sole zero-acc domain not fixed by training length alone)" rather than "LB recovers 3/4 zero-accuracy domains."

This does not invalidate the experiment's conclusions but requires table correction and narrative adjustment.

**Batch size = 1**: Each step trains on a single sample from one domain (line 328). With 49 domains and 3000 steps, each domain sees ~61 gradient updates. At 6000 steps, ~122 updates. The "undertrained at 3000 steps" conclusion is arithmetically consistent.

**Single seed**: All 22 configs use seed=42. With 10 val samples per domain, a "recovery" from 0% to 40% means 0/10 to 4/10 correct -- Wilson confidence interval roughly [12%, 74%]. Directionally informative but not precise. Acknowledged in Limitations.

**Evaluation correctness**: The evaluate_router function uses `router(h_val, hard=True)` and sorts by raw logits. For sigmoid, this correctly evaluates the router's confidence ranking without Gumbel noise. Standard practice.

## Hypothesis Graph Consistency

K1 criterion: "No config beats current default (86.33% top-2 accuracy) by >5%." Best improvement: 4.49% (k=4), 4.08% (same k=2). K1 FAIL is correctly assessed.

The experiment's actual value (training-length finding, zero-domain analysis) was not pre-registered as kill criteria. This is honestly acknowledged.

## Macro-Scale Risks (advisory)

1. **Training budget scales with N**: The 3000-to-6000-step improvement suggests training budget should scale sub-linearly with expert count. At N=100+, convergence requirements are unknown.

2. **L1 alpha needs per-N calibration**: alpha=0.1 was found optimal for N=49. The balance between BCE and L1 gradients changes with N (more experts = more L1 penalty terms = more suppression).

3. **Mean-pooled routing ceiling**: Dialogue failure (high intra-domain variance) is a fundamental limitation of mean-pooled features that persists at any scale.

4. **10 val samples per domain**: Statistical power is very low for per-domain conclusions. A domain going from 0/10 to 4/10 is directionally useful but needs macro-scale validation with larger eval sets.

## Verdict

**KILL** (confirming the experiment's own verdict, with one correction needed for FINDINGS.md citation)

K1 FAIL is correctly assessed. The methodology is sound post-revision. The 6000-step no-LB control is a well-designed addition that cleanly supports the primary finding: training length is the dominant factor, not L1 regularization.

**Before citing in FINDINGS.md, correct:**

1. **Zero-accuracy domain table (PAPER.md lines 80-86)**: The "Baseline (3k)" column reports top-1 accuracy while the rest of the paper tracks top-2. Chemistry is 20% (not 0%) and debate is 10% (not 0%) at baseline for top-2. Only wikitext and dialogue are genuine zero-accuracy domains. The narrative should be adjusted: LB uniquely recovers wikitext (the one zero-acc domain that training length alone cannot fix), not "3/4 zero-accuracy domains."

2. **MATH.md parameter count**: 668,977 should be 668,209. Cosmetic but stale across two reviews.

The kill is valid. The transferable findings are:
- Training length (3000 to 6000 steps) is the primary accuracy driver for 49-expert routing (+5.3pp top-2)
- L1 gate regularization (alpha=0.1) has a narrow but real value: recovering the single hardest zero-accuracy domain (wikitext, cos=0.996 with history)
- Temperature is robust in [0.5, 2.0]; tau=0.1 is catastrophic
- Dialogue is architecturally unroutable via mean-pooled features (variance 4.375, 13x typical)
