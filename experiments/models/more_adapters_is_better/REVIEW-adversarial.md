# Peer Review: More Adapters Is Better

## NotebookLM Findings

Skipped -- direct code and data inspection is sufficient for this experiment. The central issue is structural and visible in the raw JSON.

## Mathematical Soundness

### MATH.md is a tautology proof, not a derivation

MATH.md Section 2 ("Why It Works") states:

> Under perfect routing for domain d tokens: S(x) = {d} (only the correct adapter fires).
> Then: PPL_k(d) = PPL_{k+1}(d) = exp(L_base - delta_d)

This is not a mathematical finding. It is a restatement of the definition of oracle routing. If you perfectly route each domain to exactly one adapter, then by construction adding a new adapter for a new domain cannot affect existing domains. The entire "proof sketch" is definitional.

The only non-trivial content in MATH.md is the interference bound under imperfect routing (Section 2, last paragraph), which estimates ~2.4% interference from false positives. But this bound is never tested because the experiment's routed PPL matches oracle PPL exactly (see below).

### The independence argument is unfalsifiable as designed

K2 ("zero regression") tests whether existing domains regress when new domains are added. But the routing mechanism evaluates each domain by computing the mean hidden state over all validation samples for that domain, then applying each routing head to this single mean vector to produce a static binary mask. This is **per-domain routing**, not per-sequence or per-token routing.

Under per-domain routing with threshold 0.5, the routing decision for domain X is deterministic and depends only on domain X's hidden states and the routing heads. It does not depend on how many other adapters exist in the pool (N). The only way K2 can fail is if the routing head for a newly-added domain fires a false positive on domain X's mean hidden state. But routing heads are retrained at each N (acknowledged in Limitations item 2), so the head for domain X is re-optimized at each N to correctly classify X.

This makes K2 trivially true by construction. It is not evidence that "more adapters don't interfere."

## Novelty Assessment

### The experiment proves routing = oracle, not that more adapters = better

The smoking gun is in `results.json`. Compare `avg_routed_ppl` and `avg_oracle_ppl` at every N:

| N  | avg_oracle_ppl | avg_routed_ppl | Difference |
|----|---------------|---------------|------------|
| 5  | 7.5302        | 7.5302        | 0.0000     |
| 10 | 7.8766        | 7.8766        | 0.0000     |
| 15 | 7.3890        | 7.3897        | 0.0007     |
| 20 | 7.0152        | 7.0157        | 0.0005     |
| 24 | 6.2932        | 6.2941        | 0.0009     |

**Routed PPL equals oracle PPL to 4 decimal places at every N.** This means the routing heads are achieving effectively perfect oracle selection. The experiment is not testing "more adapters is better" -- it is confirming that the routing heads can correctly identify which domain each validation batch belongs to.

This is expected: each validation batch is 100% single-domain text. The routing heads are trained on those same domains' hidden states (40 train, 50 val samples per domain). Classifying "is this medical text?" given the mean hidden state of 25-50 medical validation samples is a trivial binary classification problem, especially when the positive class is a single narrow domain and the negative class is a mixture of 23 others.

### The gamma improvement is a composition effect, not a quality improvement

gamma_routed goes from 0.668 (N=5) to 0.625 (N=24). But gamma is defined as avg_routed_ppl / avg_base_ppl, where the average is over all N domains in the pool. As N increases, different domains enter the average. The gamma "improvement" is entirely driven by the changing denominator population.

To see this clearly: at N=5, avg_base_ppl = 11.274. At N=24, avg_base_ppl = 10.076. The domains added at N=20-24 (cybersecurity 3.83, marketing 3.83, sports 3.59, music 3.57) have very low base PPL, pulling down the average. Meanwhile, the routed PPL for these domains is also low (2.3-3.1). The ratio gamma is dominated by which domains are in the pool, not by any change in per-domain quality.

**No existing domain gets better as N increases.** Every domain's routed PPL is frozen at its oracle value across all N. The paper's own Table (K2 evidence) proves this: 9/10 domains show 0.0% change, 1 shows +0.6%.

### What "more adapters is better" actually means here

The claim reduces to: "adding a new adapter that helps its domain, with oracle routing that ensures zero interference, improves the system average." This is arithmetically obvious. If adapter N+1 reduces domain N+1's PPL below its base PPL, and oracle routing ensures no other domain is affected, then the system average must improve (assuming the new domain's improvement ratio is better than the current average, or at minimum doesn't worsen it).

## Experimental Design

### Critical flaw: routing heads are retrained at each N

Lines 519-522 of `run_experiment.py`:
```python
heads = train_routing_heads_for_n(all_hidden, domains_subset)
```

New routing heads are trained from scratch at each N value. This is explicitly acknowledged in Limitations item 2 but severely undercuts the claim. In production, you cannot retrain routing heads every time an adapter is added. The experiment does not test whether routing degrades as N grows with frozen heads, which is the actual concern for the "more adapters is better" thesis.

### Positive-class recall is catastrophically low for many routing heads

The results show many routing heads have near-zero positive recall while maintaining high negative recall:

| Domain | N=24 pos recall | neg recall |
|--------|----------------|------------|
| science | 0.12 | 0.94 |
| history | 0.14 | 0.94 |
| philosophy | 0.10 | 0.94 |
| creative_writing | 0.10 | 0.97 |
| environmental | 0.10 | 0.95 |
| politics | 0.12 | 0.93 |
| economics | 0.14 | 0.95 |

These heads almost never fire (10-14% recall). The "overall accuracy" of ~90% is inflated by the 23:1 class imbalance (always predicting "no" gives 95.8% accuracy). Yet the routed PPL still matches oracle. How?

The answer is in the routing logic (lines 700-714). For each domain's evaluation, the code checks ALL routing heads:

```python
for di, d in enumerate(domains_subset):
    if d in heads:
        score = mx.sigmoid(heads[d](h_mean).squeeze()).item()
        mask[di] = score > 0.5
```

When evaluating "science" text: the science head rarely fires (12% recall), but it doesn't matter because other heads also rarely fire on science text (high negative recall ~94%). The likely outcome is that exactly 1 head fires -- the correct one -- or 0 heads fire (triggering the oracle fallback at line 710):

```python
if not any(mask):
    mask[eval_di] = True
```

**The oracle fallback silently converts routing failures into oracle selections.** When no head fires, the code defaults to activating the correct adapter. This is not routing -- it is oracle with extra steps.

### No variance estimates

Single seed (42). 25 validation samples per domain. No confidence intervals on PPL. No multi-seed runs. The code even re-seeds numpy at each N (`np.random.seed(SEED)` at line 521), ensuring routing head training is deterministic but also ensuring no variance information.

### No ablation to distinguish routing from oracle

The experiment should include a "random routing" or "worst-case routing" baseline to show the routing mechanism adds value beyond oracle fallback. Without this, the claim that "routing works" is unfalsifiable.

## Macro-Scale Risks (advisory)

1. **Routing head retraining at each N is O(N^2).** Each head trains against all other domains as negatives. At N=1000, this is prohibitive. The experiment provides no evidence about routing quality with frozen heads.

2. **Oracle fallback mask.** Any production system that silently falls back to oracle when routing fails is not actually routing. The fallback must be removed or replaced with a "base-only" fallback to get honest routing measurements.

3. **Per-domain mean hidden state routing will not work for mixed-domain inputs.** Real text is not 100% single-domain. The routing evaluation here (mean over all domain validation samples) is maximally favorable.

4. **The positive-recall problem compounds at scale.** At N=24, many heads already have <15% positive recall. At N=100+, the negative class becomes even more diverse, positive recall will likely approach 0% for most heads, and the system will rely almost entirely on oracle fallback.

## Verdict

**KILL**

The experiment is a tautology. It proves that oracle routing + domain-specialized adapters produces oracle-quality results at every N. This is true by construction and provides no evidence for the thesis that "more adapters improve the system."

Specific grounds:

1. **Routed PPL = Oracle PPL at every N** (within 0.01%), meaning the routing mechanism is not being tested -- it is achieving perfect oracle selection via a combination of near-perfect negative recall and an oracle fallback for routing failures.

2. **K2 is trivially true.** Per-domain routing with retrained heads and oracle fallback makes it impossible for existing domains to regress. The kill criterion is unfalsifiable.

3. **gamma improvement is an averaging artifact.** No existing domain improves as N grows. The system average changes only because new domains (with their own fixed improvement ratios) enter the pool.

4. **MATH.md derives nothing.** The "proof sketch" restates the definition of oracle routing. The interference bound is never tested because routing never introduces interference (it matches oracle exactly).

5. **Oracle fallback in the routing code** (lines 709-710) silently converts routing failures into correct answers, making the routing evaluation uninterpretable.

If this experiment is to be salvaged, it requires:

1. Remove the oracle fallback. When no head fires, apply base-only (no adapter). This gives honest routing measurements.
2. Freeze routing heads trained at N=10 and evaluate at N=15, 20, 24 without retraining. This tests whether routing degrades with more adapters.
3. Report per-domain routed PPL separately from oracle PPL with the difference highlighted. The current presentation obscures that they are identical.
4. Track how many domains hit the oracle fallback at each N. If >20% of evaluations use fallback, the routing is not functioning.
5. Include a control: random routing mask (selecting k random adapters) to establish a lower bound.
6. Replace gamma (which conflates pool composition with quality) with a fixed-domain metric: average routed PPL over the first 10 domains only, measured at every N. This isolates interference from composition changes.
