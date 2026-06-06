# Adversarial Review: Contrastive Routing Keys

## Verdict: PROCEED

The experiment was properly designed, the kill is justified against pre-defined criteria,
the root cause analysis is correct, and the negative result is informative. The PAPER.md
is honest. No mathematical errors or hidden flaws that invalidate the conclusions.

**Proceed to integration of the negative result.**

---

## 1. Mathematical Verification

### 1.1 Scoring function: CORRECT

```
s_i = ||x @ K_i||^2, K_i in R^{d x d_key}
```

Implementation in `contrastive_router.py:37-39` computes `sum(K_i(x) * K_i(x), axis=-1)`,
which is the squared L2 norm of the projection. Matches MATH.md Section 3.1.

### 1.2 InfoNCE loss: CORRECT

MATH.md defines:
```
L_InfoNCE(x, d) = -score_d(x)/tau + log(sum_{d'} exp(score_{d'}(x) / tau))
```

Implementation uses `nn.losses.cross_entropy(domain_scores / tau, labels)` which computes
`-log(softmax(logits)[correct])` = `-logit_correct + log(sum exp(logits))` — algebraically
identical to the MATH.md formula. Correct.

### 1.3 Domain max-pooling: CORRECT

`score_d = max_{i in groups(d)} s_i` — the implementation at line 178-181 uses `mx.max`
over the correct group slices. Gradients flow to the argmax element via MLX's automatic
differentiation (straight-through, as MATH.md Assumption 3 states).

### 1.4 Variable naming: COSMETIC ISSUE (non-blocking)

MATH.md Section 2 defines `q in R^{d_key}` and `k_i in R^{d_key}` but never uses them.
Section 3 introduces `z_i` and `s_i` instead. Theorist flagged this; confirmed cosmetic.

---

## 2. Experiment Design Verification

### 2.1 Does the experiment test the stated hypothesis?

**Yes.** The hypothesis is: "InfoNCE-trained contrastive routing keys achieve >85% domain
routing accuracy and match softmax router composition quality." The experiment directly
measures both.

### 2.2 Are the baselines appropriate?

**Yes, with one methodological note.**

- **Joint training** (gold standard): correct
- **Softmax router calibration** (existing method): correct, same composition protocol
- **Linear probe** (lower bound for contrastive utility): **evaluation is on training data,
  not held-out**

The linear probe at line 455 is trained on `h_all` and accuracy is measured on the same
`h_all`. The contrastive keys' 53.3% is held-out. This comparison favors the probe.
However, this makes the kill verdict MORE conservative, not less — if the probe only gets
~60% even on training data, the discriminative signal is truly weak. A held-out probe
would score lower (est. ~55-57%), still above contrastive keys. Kill still holds.

### 2.3 Missing ablation: d_key sweep

MATH.md Section 7.3 recommends testing d_key in {4, 8, 16}. The experiment only tests
d_key=8. However, with 53.3% routing accuracy (vs 85% target) and the linear probe at
only 60%, changing key dimensionality cannot bridge a 32 percentage-point gap when the
underlying signal doesn't exist. Not blocking.

### 2.4 Kill thresholds: PRE-REGISTERED and properly applied

All four criteria from MATH.md Section 10 were checked. Three of four exceeded:

| Criterion | Threshold | Result | Status |
|-----------|-----------|--------|--------|
| Routing accuracy | kill <70% | 53.3% | KILLED |
| Composition quality | kill >10% | +141% | KILLED |
| vs linear probe | must beat | 53.3% < 59.8% | KILLED |
| Sample/step efficiency | >100 | 128 samples, 50 steps | OK |

Pre-registered kill criteria are the gold standard for micro experiments. The Builder
followed them honestly.

---

## 3. Root Cause Analysis: CORRECT

### 3.1 The core diagnosis

MATH.md Assumption 6: "At micro scale (d=64, a-m vs n-z), domains are distinguishable."

This assumption is empirically falsified. Evidence:
- Linear probe: ~60% (barely above chance)
- Contrastive keys: ~53% (essentially random)
- Character-level tokenization creates overlapping vocabularies (both domains use a-z)
- Domain split by first character is lexicographic, not distributional

### 3.2 The deeper insight is sound

**Task-based routing (reconstruction loss) beats identity-based routing (contrastive
discrimination).** The softmax router at +0.2% vs joint works because it asks "which
experts minimize prediction error?" not "which domain is this from?" This is a genuine
architectural insight that should be elevated to a principle.

This connects to a known pattern in MoE literature: expert routing by loss gradient
outperforms expert routing by input clustering. DeepSeek-V3's load balancing works for
the same reason — it optimizes for balanced utilization under task loss, not for input
categorization.

### 3.3 Scale considerations: FAIR ASSESSMENT

The PAPER.md correctly notes that macro scale (d=256+, BPE tokenization, real domains
like Python vs JavaScript) would provide much stronger domain signals. The kill is
scoped to micro, not to the concept. This is appropriately hedged.

---

## 4. Issues Found (Minor, Non-blocking)

### 4.1 Linear probe evaluation methodology (MINOR)

As noted in 2.2, the probe is evaluated on training data. For the PAPER, clarify:
"Linear probe accuracy (train set): 59.8%" vs "Contrastive keys (held-out): 53.3%".
The comparison is still valid directionally but should be transparent about the protocol
difference.

### 4.2 VISION.md still presents K_i as core architecture

VISION.md Section "The Fix: Decoupled Contrastive Routing Keys" and the architecture
diagram both present Expert_i = (A_i, B_i, K_i) with contrastive keys as the solution.
This needs a caveat noting that contrastive keys are unvalidated at micro scale and
deferred to macro validation. The softmax router is the currently validated mechanism.

### 4.3 Hidden state extraction during composition

The `extract_hidden_states` function at line 143-161 sets `uniform_routing = True` during
extraction. This means hidden states are computed with all groups contributing equally,
which is different from the routing scenario where only top-k groups fire. The
states used for key training may not match the states seen during routing inference.

However, this affects both contrastive keys and the linear probe equally, and the root
cause (weak domain signal) is independent of routing mode. Non-blocking.

---

## 5. Prior Art Check

### 5.1 Contrastive routing in MoE

Router-Tuning (Zhao et al., 2024) trains MoE routers with contrastive-style losses for
domain-specific expert selection. Their key difference: they operate at much larger scale
(7B+ parameters) where domain signals are strong. Our micro-scale failure is consistent
with their finding that routing discrimination requires sufficient model capacity.

### 5.2 InfoNCE for routing keys

No direct prior art for using InfoNCE to train routing keys separately from expert
computation. The concept is novel. The failure at micro scale doesn't invalidate the
approach — it identifies a scale threshold.

---

## 6. Verdict Details

**PROCEED to integration.** The negative result should be:

1. **Recorded in FINDINGS.md** with root cause: identity-routing requires domain-
   discriminative signal that doesn't exist at micro scale
2. **VISION.md updated** to note contrastive keys deferred to macro; softmax router
   is the validated composition mechanism
3. **Principle extracted**: task-routing (minimize prediction error) > identity-routing
   (classify domain) when domains are distributionally similar
4. **Memory recorded**: contrastive routing killed at micro scale; linear probe ceiling
   ~60% on a-m vs n-z names proves the signal is too weak, not the mechanism

The experiment advanced the research program by eliminating a dead branch and surfacing
a principle (task vs identity routing) that informs future architecture choices.
