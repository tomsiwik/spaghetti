# Generation Quality Test v2: Mathematical Foundations

## Problem Statement

Given a base model $f_\theta$ and $N=5$ domain-specialized LoRA adapters
$\{\Delta W_i = B_i A_i^T\}_{i=1}^N$, does routed composition produce
measurably better generated text than the base model alone?

v2 revision addresses 6 fixes from adversarial review. Key changes:
top-1 routing, domain-appropriate metrics, multiple seeds, no XPPL in primary scoring.

## Notation

| Symbol | Shape/Range | Description |
|--------|-------------|-------------|
| $f_\theta$ | - | Base model (BitNet-2B-4T) |
| $A_i$ | $(d, r)$ | Frozen Grassmannian projection for expert $i$ |
| $B_i$ | $(r, d_{out})$ | Trained ternary B-matrix for expert $i$ |
| $d$ | $2560$ | Hidden dimension |
| $r$ | $16$ | LoRA rank |
| $N$ | $5$ | Number of domain experts |
| $w_i$ | $[0, 1]$ | Routing weight for expert $i$, $\sum_i w_i = 1$ |
| $K$ | $10$ | Prompts per domain |
| $S$ | $\{42, 137, 2024\}$ | Random seeds |
| $\text{DKD}(x, d)$ | $[0, 1]$ | Domain keyword density |
| $\text{NGD}(x)$ | $[0, 1]$ | N-gram diversity (trigram) |
| $\text{COH}(x)$ | $[0, 1]$ | Coherence score |
| $\text{REP}(x)$ | $[0, 1]$ | Repetition score (unique/total words) |
| $\text{SYN}(x)$ | $\{0, 1\}$ | Python syntax validity via ast.parse |
| $\text{ANS}(x, a^*)$ | $\{0, 1\}$ | Numeric answer correctness |
| $\text{XPPL}_d(x)$ | $[1, \infty)$ | Cross-perplexity under domain $d$ adapter |

## Three Configurations

### 1. Base Only
$$y = f_\theta(x)$$

### 2. Uniform 1/N Composition
$$y = f_\theta(x) + \frac{\alpha}{N} \sum_{i=1}^N x A_i Q(B_i)$$

where $Q(\cdot)$ is STE ternary quantization and $\alpha = 20$ is the LoRA scale.

### 3. Oracle-Routed Top-1 (Fix 1: changed from top-2)

$$y = f_\theta(x) + \alpha \cdot w_d \cdot x A_d Q(B_d)$$

where $w_d = 1.0$ for the correct domain $d$, and $w_j = 0$ for all $j \neq d$.

**Change from v1:** v1 used top-2 with $w_d = 0.7$ and $w_{(d+1) \bmod N} = 0.3$.
The secondary expert was selected by list order, not semantic relevance. This caused
interference (legal + finance pairing produced off-topic first-person narratives).
v2 isolates the single-expert effect cleanly.

**Justification:** Oracle routing is valid because prior experiments showed 99.9%
routing head accuracy on these 5 domains. Top-1 with weight=1.0 tests whether a
single domain adapter improves generation quality -- the most fundamental question.

## Primary Scoring Functions (Fix 3: Domain-Appropriate)

### Code Domain
$$\text{Score}_{\text{code}}(x) = 0.5 \cdot \text{SYN}(x) + 0.5 \cdot \text{DKD}(x, \text{code})$$

$\text{SYN}(x) = 1$ if `ast.parse` succeeds on the full text, any extracted code block
(`\`\`\`python ... \`\`\``), or any contiguous block of Python-like lines. This directly
measures whether the adapter steers generation toward syntactically valid Python.

**Rationale:** v1's composite penalized actual code because code has lower coherence
scores (no prose sentences) and different n-gram patterns than English text.

### Math Domain
$$\text{Score}_{\text{math}}(x) = 0.5 \cdot \text{ANS}(x, a^*) + 0.5 \cdot \text{DKD}(x, \text{math})$$

$\text{ANS}(x, a^*) = 1$ if the extracted numeric answer matches ground truth $a^*$
within 1% relative tolerance ($|a - a^*| / |a^*| < 0.01$, or $|a| < 0.01$ when $a^* = 0$).

Ground truth $a^*$ is extracted from training data's `<<calc=result>>` format (GSM8K style).
Answer extraction from generated text tries patterns in priority order:
1. "the answer is X" / "answer: X"
2. "= X" (final equation)
3. "$X" (dollar amount)
4. Last number in text

**Rationale:** v1's composite penalized concise correct answers (routed gave "$322"
while base gave incomplete calculation steps at 128-token cutoff).

### Medical, Legal, Finance Domains
$$\text{Score}_{\text{other}}(x) = 0.45 \cdot \text{DKD}(x, d) + 0.25 \cdot \text{NGD}(x) + 0.10 \cdot \text{COH}(x) + 0.20 \cdot \text{REP}(x)$$

**Change from v1:** Coherence weight reduced from 20% to 10%. Keyword density increased
from 40% to 45%. Repetition (20%) replaces the removed XPPL component.

The coherence metric $\text{COH}(x) = \max(0, 1 - |\bar{l}_s - 15| / 30)$ where
$\bar{l}_s$ is mean sentence length peaks at 15 words/sentence. This is calibrated
for English prose, not for domain-specific outputs (legal clauses with different
natural sentence lengths). Reducing its weight limits this bias.

## Sub-Metric Definitions

### Domain Keyword Density (DKD)
$$\text{DKD}(x, d) = \frac{|\{w \in x : w \in K_d\}|}{|x|}$$

where $x$ is the bag of words (lowercased) and $K_d$ is the domain keyword set.

### N-gram Diversity (NGD)
$$\text{NGD}_3(x) = \frac{|\text{unique trigrams in } x|}{|\text{total trigrams in } x|}$$

### Coherence Score
$$\text{COH}(x) = \max\left(0, 1 - \frac{|\bar{l}_s - 15|}{30}\right)$$

### Repetition Score
$$\text{REP}(x) = \frac{|\text{unique words in } x|}{|\text{total words in } x|}$$

### Cross-Perplexity (Diagnostic Only -- Fix 2)
$$\text{XPPL}_d(x) = \exp\left(-\frac{1}{|x|}\sum_{j=1}^{|x|} \log p_{f_\theta + \Delta W_d}(x_j | x_{<j})\right)$$

**Fix 2:** XPPL is NOT used in the primary scoring. It is computed and reported
as a diagnostic metric only. See the XPPL Normalization Asymmetry section below
for why it was removed.

## Aggregation Across Seeds (Fix 4)

For each domain $d$ and configuration $c \in \{\text{base}, \text{uniform}, \text{routed}\}$:

$$\mu_{d,c} = \frac{1}{|S|} \sum_{s \in S} \left( \frac{1}{K} \sum_{k=1}^{K} \text{Score}_d(x_{d,c,s,k}) \right)$$

$$\sigma_{d,c} = \sqrt{\frac{1}{|S|-1} \sum_{s \in S} \left( \frac{1}{K} \sum_{k=1}^{K} \text{Score}_d(x_{d,c,s,k}) - \mu_{d,c} \right)^2}$$

With $|S| = 3$ seeds, the standard error of the mean is $\sigma / \sqrt{3}$.

**Routed wins on domain $d$** iff $\mu_{d,\text{routed}} > \mu_{d,\text{base}}$.

## Kill Criteria (Fix 5: Pre-Registered, Unchanged)

**K1 (id=272):** Let $W = |\{d : \mu_{d,\text{routed}} \leq \mu_{d,\text{base}}\}|$. **KILL if $W \geq 3$.**

**K2 (id=273):** Let $M = |\{d : |(\mu_{d,\text{routed}} - \mu_{d,\text{base}}) / \mu_{d,\text{base}}| > 0.05\}|$. **KILL if $M = 0$.**

**K3 (id=274):** Let $I_d = 1$ if more than 50% of routed samples across all seeds are incoherent. **KILL if $\sum_d I_d = 5$.**

These are identical to the v1 criteria. The only change is the scoring function used
(domain-appropriate instead of uniform composite) and the aggregation (mean across 3
seeds instead of single seed).

## Fix 6: XPPL Normalization Asymmetry (v1 Bug Documentation)

The v1 composite score included an XPPL component with weight 0.2:

$$\text{xppl\_norm}(c, d) = \max\left(0, \; 1 - \frac{\text{XPPL}_d(x_c)}{\max(\text{XPPL}_d(x_{\text{base}}), 1)}\right)$$

**The asymmetry arises as follows:**

For the base configuration ($c = \text{base}$):
$$\text{xppl\_norm}(\text{base}, d) = \max\left(0, \; 1 - \frac{\text{XPPL}_d(x_{\text{base}})}{\text{XPPL}_d(x_{\text{base}})}\right) = \max(0, 0) = 0$$

The base configuration ALWAYS gets exactly 0 from this component. This is not an
approximation -- it is algebraically guaranteed.

For the routed configuration ($c = \text{routed}$):
- If $\text{XPPL}_d(x_\text{routed}) < \text{XPPL}_d(x_\text{base})$: positive contribution (helps routed)
- If $\text{XPPL}_d(x_\text{routed}) > \text{XPPL}_d(x_\text{base})$: $\max(0, \text{negative}) = 0$ (no penalty)

**Summary:** The XPPL component is a one-sided bonus for routed. It can only increase
routed's composite score relative to base. It can never decrease it. Base gets 0 regardless.

**Impact on v1 results:** Despite this systematic bias in routed's favor (worth up to
0.2 * 1.0 = 0.2 points in composite score), routed still lost 3/5 domains. This means
the v1 kill was actually STRONGER than it appeared -- routed lost even with a built-in
scoring advantage.

**Additional tautology:** Cross-PPL itself is tautological for the routed configuration.
Text generated WITH adapter $X$ naturally has lower perplexity UNDER adapter $X$ because
the adapter shaped the token distribution during generation. Measuring self-consistency
is not the same as measuring quality. A fair test requires evaluating under a metric that
is independent of the generation process.

**v2 resolution:** XPPL is completely removed from the primary scoring function. It is
computed as a diagnostic metric and reported in a separate section of the results, clearly
labeled "(DIAGNOSTIC ONLY)". This eliminates both the normalization asymmetry and the
tautological self-consistency bias.

## Worked Example

Domain: math, Seed: 42, Prompt: "Mr. Grey bought 3 shirts at $26 each..."

Ground truth (from training data `<<...=322>>`): $a^* = 322$

**Base output:** "The total cost is 3 * 26 + 2 * 83 + 90 = 78 + 166 + 90 = 334..."
- $\text{ANS}$: extract 334, $|334 - 322|/322 = 3.7\% > 1\%$ -> 0
- $\text{DKD}$: "cost", "total" -> ~0.10
- $\text{Score} = 0.5 \times 0 + 0.5 \times 0.10 = 0.050$

**Routed output:** "Total = (3 * 26) + (2 * 83) + 90 - 12 = $322"
- $\text{ANS}$: extract 322, exact match -> 1
- $\text{DKD}$: "total" -> ~0.05
- $\text{Score} = 0.5 \times 1 + 0.5 \times 0.05 = 0.525$

The domain-appropriate metric correctly rewards the correct answer rather than
penalizing conciseness.

## Assumptions

1. **Oracle routing is the upper bound.** If top-1 oracle routing does not improve
   generation, no routing scheme will (for independently-trained adapters).

2. **Ground truth extraction is reliable.** The `<<calc=result>>` format in the
   math data is unambiguous.

3. **ast.parse is a valid code quality proxy.** Syntactically valid Python is necessary
   (not sufficient) for useful code. It is a conservative lower bound.

4. **10 prompts x 3 seeds provides directional signal.** 30 total samples per domain
   per configuration. Standard error $\approx \sigma / \sqrt{3}$ across seed means.
   Powered for effects >10%, not for small effects <5%.

5. **Temperature=0.7 with fixed seeds is reproducible.** MLX's RNG is deterministic
   given the same seed. Cross-seed variance captures stochastic sampling effects.

## Computational Cost

Per seed, per configuration:
- Model loading + BitNet unpack: ~15s
- Generation: $K \times N = 50$ prompts x ~1s/prompt = ~50s
- Total per config per seed: ~65s

Full experiment:
- 3 configs x 3 seeds x 65s = ~585s for generation
- Cross-PPL: 5 domain models x 3 configs x 10 texts = ~150s (seed=42 only)
- Total: ~735s (~12 min) + overhead

## Memory Budget

| Component | Memory |
|-----------|--------|
| Base model (unpacked bf16) | ~4.8 GB |
| 5 adapter A matrices | ~0.15 GB |
| 5 adapter B matrices | ~0.15 GB |
| KV cache | ~0.1 GB |
| **Total** | **~5.2 GB** |

Well within M5 Pro 48GB budget.
