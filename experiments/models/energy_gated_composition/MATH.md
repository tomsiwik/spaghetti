# Energy-Gated Composition: Mathematical Framework

## Type: Guided Exploration

**Proven framework:** Energy gap discriminator (Finding #182, AUC=0.851)
**Unknown:** (1) Optimal energy gap threshold for gating. (2) Whether per-query gating
recovers prose domain quality. (3) Whether selective composition preserves structured wins.

## Step A: Diagnose the Disease

The disease is **indiscriminate composition**. When ALL adapters are composed uniformly
for every query, adapters that *hurt* on a given domain are included alongside adapters
that help. The generation_quality_test showed this clearly:

- Code: routed +14.4% (adapter helps)
- Math: routed +142.1% (adapter helps)
- Medical: routed -6.9% (adapter hurts)
- Legal: routed -8.6% (adapter hurts, repetition collapse)
- Finance: routed -11.9% (adapter hurts)

The problem is NOT that adapters are universally bad. The problem is that the same adapter
that helps on code HURTS on legal. Uniform composition includes everything, so the hurting
adapters drag down the helping ones on prose domains, and on structured domains the helping
adapter cannot compensate for the interference from unhelpful adapters.

**Root cause:** No per-query quality gate exists to decide which adapters to include.

## Step B: The Right Question

**Wrong:** "How do we improve prose generation with adapters?"
**Right:** "What decision rule optimally excludes harmful adapters per query, given
that we have a proven quality discriminator (energy gap, AUC=0.851)?"

The answer exists in classical statistics: the Neyman-Pearson lemma.

## Step C: Derive From Existing Math

### Neyman-Pearson Lemma (Neyman & Pearson, 1933)

The most powerful test of a simple null hypothesis H0 against a simple alternative H1
at significance level alpha uses the likelihood ratio:

  Lambda(x) = P(x|H1) / P(x|H0)

and rejects H0 when Lambda(x) > tau for some threshold tau.

In our setting:
- H0: adapter i hurts on query x (should exclude)
- H1: adapter i helps on query x (should include)
- P(x|H1) = P_adapted(x): probability under adapted model
- P(x|H0) = P_base(x): probability under base model

The log-likelihood ratio is:

  log Lambda(x) = log P_adapted(x) - log P_base(x)
               = -NLL_adapted(x) - (-NLL_base(x))
               = NLL_base(x) - NLL_adapted(x)
               = -Delta_E(x)

where Delta_E = NLL_adapted - NLL_base is the energy gap.

**Therefore: the energy gap IS the Neyman-Pearson optimal test statistic (up to sign).**

### Decision Rule

Include adapter i for query x if and only if:

  Delta_E_i(x) < tau

where tau <= 0 means "adapter must lower NLL" (strict gating), and tau = 0 is the
natural threshold (include if and only if the adapter reduces NLL on this specific query).

### ROC Analysis for Threshold Selection

Finding #182 established AUC=0.851 using task accuracy as ground truth. The optimal
threshold tau* is the one that maximizes Youden's J-statistic:

  J(tau) = sensitivity(tau) + specificity(tau) - 1

This is a data-driven threshold that the experiment will discover (the "unknown" in
this guided exploration).

## Step D: Predictions

### Behavioral Predictions

1. **Energy-gated composition will beat base on >= 4/5 domains** because it excludes
   harmful adapters (the ones causing prose degradation) while preserving helpful ones.

2. **On structured domains (code, math):** gated composition will match or exceed
   oracle top-1 routing because the energy gap correctly identifies the domain adapter
   as helpful (AUC=0.942 for math).

3. **On prose domains (medical, legal, finance):** gated composition will match or
   beat base because the energy gate will EXCLUDE the adapters causing mode collapse
   (legal "hoa" repetition, finance vocabulary narrowing), effectively falling back
   to base model on these queries.

4. **Gated will beat uniform composition on ALL 5 domains** because uniform includes
   harmful adapters; gated excludes them.

### Quantitative Predictions

| Metric | Prediction | Basis |
|--------|-----------|-------|
| Domains where gated > base | >= 4/5 | Energy gate excludes harmful adapters |
| Domains where gated > uniform | 5/5 | Uniform includes all; gated excludes harmful |
| Code domain: gated vs base | >= +10% | Energy gap correctly selects code adapter (AUC high) |
| Math domain: gated vs base | >= +50% | Energy gap AUC=0.942, strong signal |
| Prose domains: gated vs base | >= -2% (no worse) | Energy gate falls back to base when no adapter helps |
| Energy gap overhead | < 20% of gen time | Single forward pass per adapter on prompt only |
| Energy gap AUC for gating | > 0.80 | Consistent with Finding #182 |

### Key Insight: Base Fallback

When the energy gate determines NO adapter helps a given query, the composition
produces zero adapter contribution -- effectively running the base model. This is
the "selective activation" approach from LEARNINGS.md (Alternative #5). The energy
gap provides a principled, Neyman-Pearson-optimal way to implement it.

## Step E: Assumptions & Breaking Conditions

1. **Energy gap computed on prompt tokens generalizes to generated tokens.**
   If violated: energy gap is only useful for classification, not generation gating.
   Consequence: gate makes wrong decisions, gated performance could be worse than base.

2. **Single-adapter energy gap transfers to multi-adapter composition.**
   Finding #182 tested single adapters. In composition, adapter interactions might
   change the energy landscape. If violated: need per-composition energy measurement
   (exponential cost).

3. **The 5 real-data adapters have sufficient quality variance.**
   Code and math must be "helps" on their domain; legal/finance must be "hurts" on
   prose. If all adapters hurt on all domains, gating cannot help.

4. **Threshold tau=0 is a reasonable initial gate.**
   If the optimal threshold is far from 0, the experiment will find it via ROC analysis.

## Step F: Worked Example (d=2560, BitNet-2B)

Consider a medical query:
- Base NLL on prompt: 2.50 nats/token
- Medical adapter NLL: 2.35 nats/token -> Delta_E = -0.15 (helps, include)
- Code adapter NLL: 2.65 nats/token -> Delta_E = +0.15 (hurts, exclude)
- Math adapter NLL: 2.70 nats/token -> Delta_E = +0.20 (hurts, exclude)
- Legal adapter NLL: 2.80 nats/token -> Delta_E = +0.30 (hurts, exclude)
- Finance adapter NLL: 2.55 nats/token -> Delta_E = +0.05 (marginal, exclude at tau=0)

With tau=0 gate: only medical adapter is composed. This avoids the mode collapse from
legal adapter while preserving medical domain benefit.

For a code query:
- Base NLL: 1.80
- Code adapter NLL: 1.50 -> Delta_E = -0.30 (helps, include)
- Others: all Delta_E > 0 (exclude)

Result: code adapter only, matching oracle top-1 behavior.

For a general query where no adapter helps:
- All Delta_E > 0 -> compose nothing, use base model directly.

## Step G: Complexity & Architecture Connection

**Energy gap computation:** For each query with T prompt tokens and N adapters:
- 1 base forward pass: O(T * d^2 * L) where L = num layers
- N adapter forward passes: O(N * T * d^2 * L)
- Total: (1 + N) forward passes on prompt only

**Optimization:** The base forward pass is shared. Each adapter forward pass adds
only the LoRA delta, so the cost is O(T * d * r * L) per adapter (r << d).

**At serving time with N=5 adapters, r=16, d=2560, L=32:**
- Base pass: standard cost
- Per-adapter energy: 5 * T * 2560 * 16 * 32 = 5 * T * 1.3M ops (negligible vs base)
- The energy computation is dominated by the base forward pass, which is already
  needed for generation. Net overhead is ~5 cheap LoRA forward passes.

**Memory:** Only one model in memory. Adapters loaded one at a time for energy check.
After energy computation, only selected adapters remain for generation.

## Self-Test

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   The Neyman-Pearson energy gate excludes adapters that increase NLL, making it
   impossible for harmful adapters to participate in composition.

2. **Which existing theorem(s) does the proof build on?**
   Neyman-Pearson lemma (1933): the log-likelihood ratio is the uniformly most
   powerful test statistic for simple hypotheses.

3. **What specific numbers does the proof predict?**
   >= 4/5 domains gated > base; 5/5 domains gated > uniform; code >= +10%, math >= +50%;
   prose domains no worse than -2% vs base; overhead < 20%.

4. **What would FALSIFY the proof?**
   If energy gap on prompt tokens does NOT predict which adapters help during generation
   (Assumption 1 fails), the gate makes wrong decisions.

5. **How many hyperparameters does this approach add?**
   1: threshold tau. The experiment explores tau in {-0.1, -0.05, 0, 0.05, 0.1} and
   also tests the ROC-optimal threshold. tau=0 has a natural interpretation (include
   if adapter reduces NLL). This is a Type 2 unknown.

6. **Hack check:** This is NOT a fix on top of fixes. It replaces uniform composition
   with a single principled gate (Neyman-Pearson optimal). No auxiliary losses, no
   regularizers.
