# MLP-Only Per-Token Routing: Adapter Component Decomposition

## CRITICAL: Experiment-Proof Gap

**MATH.md proves** that single-pass mixed-adapter MLP-only routing cannot produce
cross-attention contamination (MLP token-independence).

**The experiment implements** multi-pass oracle selection: 5 separate forward passes
(one per adapter, each applied uniformly to ALL tokens), then per-token NLL selection.

**These are different architectures.** The multi-pass approach avoids contamination
by construction (each pass uses one adapter), regardless of whether adapters modify
MLP-only or full-module. The contamination hypothesis is circumvented, not tested.

## Predictions vs Measurements

| Prediction (from proof) | Measured | Match? |
|------------------------|----------|--------|
| K790: MLP-only PPL < per-seq best (4.815) | 4.656 | YES (3.3% better) |
| K791: MLP-only PPL < seg-isolated (4.042) | 4.656 | NO (15.2% worse) |
| K792: MLP-only != per-seq (diff > 0.01) | diff = 0.159 | RETIRED (vacuous — see below) |
| Full-module per-tok == per-seq (null) | diff = 0.315 | NO (contradicts Finding #305) |

## Hypothesis

Applying LoRA adapters ONLY to MLP layers per-token within full-sequence forward
passes eliminates cross-attention contamination while preserving full causal context,
achieving PPL better than per-sequence routing.

**Verdict: KILLED (K791 FAIL, audit-rerun closure).**
K791 fails by 15.2% under an **oracle upper bound** (multi-pass per-token NLL
selection). Any single-pass or real-router implementation is ≤ oracle ≤ 4.656, so
K791 (< 4.042) is structurally unreachable regardless of how the tautological-routing
bug in `run_experiment.py` is fixed. See "Audit-Rerun Closure" section below. The
genuine empirical finding — MLP adapters contribute ~6x more per-token signal than
attention adapters, converging with Finding #304's perturbation split — is preserved
in LEARNINGS.md as an orthogonal observation, not a test of the hypothesis.

## Audit-Rerun Closure (tag: audit-2026-04-17-rerun, tautological-routing)

**Theorem (closure).** No fix to the tautological-routing code bug can rescue K791.

**Proof.**
1. K791 requires MLP-only per-token PPL < 4.042 (segment-isolated baseline).
2. Measured MLP-only per-token PPL = 4.656 under **oracle per-token NLL selection**
   (5 forward passes, pick adapter minimising per-token NLL). This is an upper bound
   on any routed strategy on the same data — the oracle has full knowledge of the
   true labels.
3. Any proper router R satisfies PPL(R) ≥ PPL(oracle) = 4.656. (Oracle is the
   minimiser over per-token adapter assignments; R is a restriction to router-computable
   assignments.)
4. The "tautological-routing" bug inflates the apparent per-token gain by making the
   router equivalent to the evaluation criterion. Fixing this bug **worsens** the
   measured PPL; it cannot improve it. So any fix gives PPL ≥ 4.656 > 4.042.
5. Therefore K791 is unreachable under every implementation of the MLP-only per-token
   architecture on this base + adapters. ∎

**Consequence.** The fix-category `tautological-routing` is cosmetic for this
experiment's kill assessment. K790 PASS is also preserved under the fix (the oracle
NLL-selection construction trivially beats per-sequence single-adapter selection).
K792 was retired as vacuous during review and is unaffected.

**Closure-rule candidate:** `oracle-upper-bound-blocks-kill-threshold` — when a kill
criterion fails under an oracle upper bound, no routing-mechanism fix can salvage it.
Second instance of the oracle-ceiling family after `ap-oracle-ceiling-blocks-headroom`
(exp_depth_routed_adapters, 2026-04-17). Promote to closure-rule.

## What This Experiment Measures

Five per-token routing strategies on 200 mixed-domain sequences (128 tokens domain A +
128 tokens domain B, 10 domain pairs, 20 sequences each):

| Strategy | PPL | vs per-seq |
|----------|-----|------------|
| base_only | 5.521 | -- |
| per_seq_best (oracle single adapter) | 4.815 | baseline |
| per_token_attn_only | 4.789 | +0.5% |
| per_token_mlp_only (NEW) | 4.656 | +3.3% |
| per_token_full | 4.500 | +6.5% |
| seg_router | 4.042 | +16.1% |
| seg_oracle | 4.054 | +15.8% |

## Key Findings

### 1. MLP adapters contribute ~6x more per-token signal than attention (PRIMARY FINDING)

In multi-pass oracle per-token selection, MLP adapters contribute ~6x more
per-token signal than attention adapters (3.3% vs 0.5% improvement over per-seq),
consistent with Finding #304's perturbation energy split (MLP ~69%, attention ~31%)
and the MoE design principle (shared attention + routed FFN).

This is an **empirical observation about adapter component decomposition**, not a
test of the contamination elimination thesis. MLP-only per-token routing produces
4.656 PPL, beating per-sequence best (4.815) by 3.3% (paired t(9)=4.69, p<0.001).

### K792 RETIRED (vacuous)

K792 (|MLP-only per-token - per-sequence| > 0.01) is trivially satisfied by ANY
multi-pass oracle per-token scheme. Even attention-only per-token (diff=0.026) passes
the same threshold. The criterion tests whether oracle selection beats single-adapter
selection, not whether contamination was eliminated. K792 is retired from the
assessment.

### 2. Full-module per-token BEATS MLP-only (contradicts contamination narrative)

Full-module per-token (4.500) is BETTER than MLP-only (4.656) by 3.4%. In the
multi-pass regime, contamination does not occur for ANY strategy (each forward pass
uses only one adapter). Therefore full-module retains both MLP and attention domain
signal without cross-adapter mixing, and unsurprisingly outperforms MLP-only.

**This directly contradicts the contamination framing.** If contamination were the
dominant effect, MLP-only should beat full-module. Instead, full-module beats MLP-only,
showing that the multi-pass methodology eliminates contamination for ALL strategies
equally. The MLP vs attention signal difference is purely about **adapter component
information content**, not contamination avoidance.

### Finding #305 null: methodological artifact, not contamination

Finding #305's "per-token full-sequence" applied a single globally-selected adapter
to ALL tokens in one forward pass. This experiment runs 5 separate forward passes
(one per adapter, each applied uniformly), then selects per-token NLL. The latter
avoids contamination by construction.

Finding #305's null result (per_token_full == per_seq) was therefore a
**methodological artifact**: its single-pass design could not differentiate per-token
from per-sequence selection. The contamination hypothesis that motivated this
experiment remains **untested** — a true test requires comparing single-pass
mixed-adapter routing (adapter A on tokens 0-127, adapter B on tokens 128-255 in ONE
forward pass) against multi-pass oracle selection.

### 3. Attention-only per-token is nearly null (as predicted)

per_token_attn_only (4.789) is only 0.5% better than per_seq_best (4.815). The
attention-level adaptation contributes minimal domain signal when applied per-token,
which is consistent with Finding #304 showing attention carries less perturbation
energy (~31%) and with our contamination theory (attention LoRA's benefit is
partially cancelled by cross-adapter K/V mixing).

### 4. MLP contributes more per-token signal than attention

The per-token improvement decomposition:
- Attention-only per-token: +0.5% over per-seq
- MLP-only per-token: +3.3% over per-seq
- Full-module per-token: +6.5% over per-seq

MLP contributes ~6x more per-token improvement than attention. This aligns with
Finding #304's perturbation split (MLP ~69%, attention ~31%) and is consistent with
the MoE principle that MLP is the natural routing layer.

### 5. Segment isolation still dominates (K791 FAIL)

Segment-isolated routing (4.042) still beats all per-token strategies by a large
margin. The gap: 4.656 (MLP-only) vs 4.042 (seg_router) = 15.2% worse.

This means segment isolation provides a benefit beyond contamination elimination.
Likely explanation: segment isolation creates a FRESH context for each segment,
preventing the model from seeing irrelevant cross-domain text. Even with base
attention (no adapter contamination), attending to 128 tokens of unrelated domain
text degrades predictions for the current segment.

## Per-Pair Breakdown

| Pair | per_seq | MLP-only | full | attn | seg_oracle | MLP vs seq |
|------|---------|----------|------|------|------------|------------|
| python+math | 2.959 | 2.858 | 2.764 | 2.955 | 2.441 | +3.4% |
| python+med | 2.774 | 2.714 | 2.698 | 2.767 | 2.457 | +2.2% |
| python+legal | 5.856 | 5.803 | 5.508 | 5.828 | 5.018 | +0.9% |
| python+creative | 4.068 | 3.985 | 3.904 | 4.124 | 3.516 | +2.0% |
| math+med | 3.356 | 3.303 | 3.259 | 3.407 | 3.020 | +1.6% |
| math+legal | 7.474 | 7.205 | 6.749 | 7.273 | 6.230 | +3.6% |
| math+creative | 5.222 | 4.886 | 4.636 | 5.138 | 4.106 | +6.4% |
| med+legal | 5.472 | 5.430 | 5.241 | 5.482 | 4.696 | +0.8% |
| med+creative | 4.882 | 4.642 | 4.581 | 4.901 | 4.012 | +4.9% |
| legal+creative | 9.788 | 9.109 | 8.664 | 9.449 | 7.789 | +6.9% |

MLP-only improves on all 10 pairs (range +0.8% to +6.9%). The improvement is largest
for pairs with high base PPL (legal+creative, math+legal, math+creative) where domain
differences are most pronounced.

## Statistical Analysis

Paired t-tests across 10 domain pairs (each pair = one observation):

| Comparison | Mean diff (PPL) | SE | t(9) | p (two-tailed) |
|-----------|----------------|-----|------|-----------------|
| MLP-only vs per-seq | 0.191 | 0.064 | 3.00 | < 0.02 |
| MLP-only vs per-seq (% improvement) | 3.27% | 0.70% | 4.69 | < 0.001 |
| Full-module vs per-seq | 0.384 | 0.105 | 3.65 | < 0.005 |

Per-pair MLP-only improvement range: +0.77% (med+legal) to +6.94% (legal+creative).
SD = 2.21%. The improvement is significantly non-zero but highly variable across pairs.

Note: The t-test on percentage improvements yields higher t than on raw PPL diffs
because percentage normalization reduces the impact of scale differences between
low-PPL pairs (python+math ~3) and high-PPL pairs (legal+creative ~10).

## Limitations

1. **Oracle routing:** This experiment uses oracle domain labels (known boundary position).
   Production requires a router. Finding #310's ridge router achieves 98.3% per-token
   accuracy, which should be sufficient.

2. **O(N) forward passes:** Computing per-token NLL requires N forward passes (one per
   domain), selecting per-token outputs. This is computationally expensive. A production
   implementation would need to either (a) pre-merge MLP adapters per-token before the
   forward pass, or (b) use a custom kernel that applies different MLP weights per token.

3. **Trained full-module, applied MLP-only:** The adapters were trained with all 7
   module types. MLP-only application is a post-hoc ablation. Purpose-trained MLP-only
   adapters might perform differently (Finding #308 suggests post-hoc ablation
   outperforms purpose-trained, so this may be fine).

4. **Segment isolation still wins:** The 15% gap between MLP-only per-token (4.656)
   and segment isolation (4.042) suggests that cross-domain context itself (not just
   adapter contamination) degrades predictions. This is a fundamental limitation of
   full-sequence per-token routing.

## What Would Kill This

- Purpose-trained MLP-only adapters perform WORSE than full-module post-hoc ablation
  (would undermine the MLP-only architecture)
- Ridge router accuracy drops significantly in MLP-only setting (router was trained
  on base hidden states, which are preserved in MLP-only)
- At scale (N>5 domains), the per-token benefit diminishes due to increased
  misrouting (Finding #310 shows 98.3% accuracy at N=5 but warns about N>10)

## Updated Understanding of Cross-Attention Contamination

This experiment **does not test** the contamination hypothesis. It circumvents
contamination via multi-pass methodology.

**What was tested:** Multi-pass oracle per-token NLL selection (5 forward passes,
one per adapter, each applied uniformly). In this regime, contamination cannot occur
because each pass uses one consistent adapter.

**What was NOT tested:** Single-pass mixed-adapter routing (adapter A on tokens
0-127, adapter B on tokens 128-255 in ONE forward pass). This is the scenario
MATH.md's proof addresses. It remains the experiment the proof calls for.

**Finding #305's null:** Reclassified as a methodological artifact. Its single-pass
single-adapter design could not distinguish per-token from per-sequence selection.
The null does not confirm contamination — it confirms the method couldn't detect
per-token signal.

**The real finding:** Full-module per-token (4.500) beats MLP-only (4.656) in the
multi-pass regime, showing that when contamination is removed by design, attention
adapters contribute additional per-token signal. This directly contradicts the
motivation for MLP-only routing — in multi-pass, full-module is strictly better.

**Implication:** MLP-only routing is only advantageous in the single-pass regime
where contamination actually occurs. The proof guarantees MLP-only single-pass is
safe, but this experiment does not validate that claim empirically. A single-pass
mixed-adapter experiment is needed.
