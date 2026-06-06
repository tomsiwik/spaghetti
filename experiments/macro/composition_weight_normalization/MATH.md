# Weight-Normalized LoRA Composition: Mathematical Foundations

## Notation

| Symbol | Definition | Dimensions |
|--------|-----------|------------|
| $W_0$ | Frozen base model weight matrix | $d_{out} \times d_{in}$ |
| $B_i, A_i$ | LoRA adapter $i$ low-rank factors | $B_i: d_{out} \times r$, $A_i: r \times d_{in}$ |
| $\Delta_i = B_i A_i$ | LoRA delta for adapter $i$ | $d_{out} \times d_{in}$ |
| $N$ | Number of composed adapters | scalar |
| $r$ | LoRA rank (uniform across adapters) | scalar |
| $d$ | Model hidden dimension ($d_{in} = d_{out}$ for square layers) | scalar |
| $\alpha$ | Per-adapter composition weight | scalar |
| $\sigma_j(\cdot)$ | $j$-th singular value of a matrix | scalar |
| $\|\cdot\|_F$ | Frobenius norm | scalar |
| $\|\cdot\|_2$ | Spectral norm (largest singular value) | scalar |

## 1. Why Unit-Weight Composition Explodes

### 1.1 The Composed Model

Under SOLE composition with uniform weight $\alpha$, the effective weight is:

$$W_{composed} = W_0 + \alpha \sum_{i=1}^{N} B_i A_i = W_0 + \alpha \sum_{i=1}^{N} \Delta_i$$

The model's output logits are $z = W_{composed} \cdot x$ for input activation $x$.
The LoRA contribution to logits is:

$$\delta z = \alpha \left(\sum_{i=1}^{N} \Delta_i\right) x$$

### 1.2 Spectral Analysis of the Summed Delta

Each $\Delta_i$ has rank at most $r$. The sum $S = \sum_{i=1}^N \Delta_i$ has
rank at most $\min(Nr, d)$.

**Case 1: Perfectly orthogonal adapters.** If the column spaces of all $B_i$ are
mutually orthogonal and the row spaces of all $A_i$ are mutually orthogonal, then
the singular values of $S$ are exactly the union of the singular values of each
$\Delta_i$. In this case:

$$\|S\|_2 = \max_i \|\Delta_i\|_2, \quad \|S\|_F^2 = \sum_{i=1}^N \|\Delta_i\|_F^2$$

The spectral norm does NOT grow with $N$. Composition is stable with unit weight.

**Case 2: Partially aligned adapters (realistic).** When adapters share subspace
overlap (measured by principal angles between column spans of $B_i$), constructive
interference occurs. Let $\cos\theta_{ij}$ denote the largest principal angle cosine
between the column spans of $B_i$ and $B_j$. For two adapters:

$$\|B_1 A_1 + B_2 A_2\|_2 \leq \|\Delta_1\|_2 + \|\Delta_2\|_2$$

with equality when the top singular vectors align. For $N$ adapters with pairwise
alignment $\rho$ (mean $|\cos\theta_{ij}|$):

$$\|S\|_2 \leq N \cdot \max_i \|\Delta_i\|_2$$

The spectral norm can grow **linearly** with $N$ in the worst case.

**Case 3: Random adapters (semi-realistic).** Under a random matrix model where
each $\Delta_i$ is drawn i.i.d. with $\mathbb{E}[\Delta_i] = 0$ and
$\text{Var}[\Delta_i] = \sigma^2_\Delta$, the Frobenius norm grows as:

$$\mathbb{E}[\|S\|_F^2] = N \cdot \mathbb{E}[\|\Delta_i\|_F^2]$$

and by the Marchenko-Pastur law for sums of rank-$r$ matrices, the spectral
norm grows as:

$$\|S\|_2 \approx O(\sqrt{N}) \cdot \|\Delta_i\|_2$$

when $Nr \ll d$ (the typical regime for LoRA composition).

### 1.3 Logit Perturbation and PPL Explosion

The cross-entropy loss is $\mathcal{L} = -\log \text{softmax}(z)_{y}$ where $y$ is
the target token. The softmax is exponentially sensitive to logit magnitudes:

$$\text{PPL} = \exp(\mathcal{L}) \propto \exp\left(\|\delta z\|_\infty\right)$$

If the LoRA perturbation $\delta z$ has magnitude $M$, the worst-case PPL
amplification is:

$$\text{PPL}_{composed} \approx \text{PPL}_{base} \cdot \exp(M)$$

With unit weight ($\alpha = 1$) and $N$ adapters producing spectral norm
$\|S\|_2 \sim c \cdot N$ (partially aligned case):

$$\text{PPL}_{unit} \approx \text{PPL}_{base} \cdot \exp(c \cdot N \cdot \|x\|)$$

This is **exponential in N**, explaining the trillions-PPL observation at N=5.

### 1.4 Empirical Calibration

From sole_critical_path (N=5):
- Unit weight: PPL in trillions ($\sim 10^{12}$)
- 1/N weight: PPL = 2.36
- Base: PPL = 5.70

The ratio $\text{PPL}_{unit} / \text{PPL}_{1/N} \approx 10^{12}$ implies:

$$\exp(c \cdot N \cdot \|x\|) / \exp(c \cdot \|x\|) = \exp(c \cdot (N-1) \cdot \|x\|) \approx 10^{12}$$

$$c \cdot 4 \cdot \|x\| \approx 27.6$$

So $c \cdot \|x\| \approx 6.9$ per adapter -- each unscaled adapter shifts logits
by roughly 7 nats. This is catastrophically large (typical logit values are O(10)).

## 2. Optimal Scaling Factor Derivation

### 2.1 The Scaling Problem

We seek $\alpha(N)$ such that the composed model's logit perturbation remains
bounded as $N$ grows:

$$\|\delta z\| = \alpha(N) \cdot \left\|\sum_{i=1}^N \Delta_i \cdot x\right\| \leq C$$

for some constant $C$ independent of $N$.

### 2.2 Three Orthogonality Regimes

The optimal $\alpha(N)$ depends on the subspace geometry of the adapters.

**Regime A: Perfect orthogonality** ($|\cos\theta_{ij}| = 0$ for all $i \neq j$).
The spectral norm of $S$ does not grow with $N$, so:

$$\alpha^*(N) = 1 \quad \text{(unit weight is fine)}$$

This is the theoretical ideal of SOLE. In practice, at macro scale (Qwen2.5-7B,
d=4096), trained adapter cosines are $\sim 0.142$ (not zero), so we are NOT in
this regime.

**Regime B: i.i.d. random subspaces** (no systematic alignment, but not zero overlap).
By random matrix theory, $\|S\|_2 \sim \sqrt{N}$, so:

$$\alpha^*(N) = \frac{1}{\sqrt{N}}$$

This is the "CLT-like" scaling: each adapter's contribution is like an i.i.d.
random variable, and the variance of the sum grows linearly, so the standard
deviation grows as $\sqrt{N}$.

**Regime C: Systematically aligned subspaces** (correlated adapters, e.g., similar
domains). The spectral norm grows linearly: $\|S\|_2 \sim N$, so:

$$\alpha^*(N) = \frac{1}{N}$$

This is the "averaging" regime. The N=5 success of 1/N suggests our current adapters
fall in or near this regime.

### 2.3 Unified Power-Law Model

We model the optimal scaling as a power law:

$$\alpha^*(N) = N^{-\beta}$$

where $\beta \in [0, 1]$ depends on adapter geometry:
- $\beta = 0$: perfect orthogonality (Regime A)
- $\beta = 0.5$: random subspaces (Regime B)
- $\beta = 1.0$: fully correlated (Regime C)

**Predicting $\beta$ from adapter statistics.** Let $\rho$ be the mean pairwise
absolute cosine similarity between adapter subspaces. Under a linear interpolation
model:

$$\beta \approx \frac{1}{2} + \frac{1}{2} \cdot \rho \cdot \sqrt{Nr/d}$$

At the measured $\rho = 0.142$ (d=896 macro), $N=5$, $r=16$:

$$\beta \approx 0.5 + 0.5 \cdot 0.142 \cdot \sqrt{80/896} \approx 0.5 + 0.021 \approx 0.52$$

This predicts $\alpha^* \approx N^{-0.52}$, close to $1/\sqrt{N}$ but slightly
steeper. At N=50: $\alpha^*(50) \approx 50^{-0.52} \approx 0.135$ vs $1/\sqrt{50} = 0.141$.

### 2.4 Why 1/N Worked at N=5 But May Be Too Aggressive

At $N=5$: $1/N = 0.200$ and $1/\sqrt{N} = 0.447$.

If the true optimal is $N^{-0.52}$, then $\alpha^*(5) \approx 5^{-0.52} \approx 0.431$.

The fact that 1/N "worked" (PPL=2.36) does not mean it is optimal -- it means it
was sufficient to prevent explosion. The question is whether 1/N **over-dilutes**
each adapter's contribution. Evidence for this:

- Top-1 (medical only, PPL=2.96) beats all 1/N compositions (PPL=3.50 in Exp 2)
- SOLE vs monolithic gap is 32.7% -- some of this may be over-dilution

At $N=50$: $1/N = 0.02$ but $1/\sqrt{N} = 0.141$. This is a **7x difference**.
If $\beta \approx 0.5$, then 1/N would over-dilute by 7x, each adapter
contributing only $0.02/0.141 = 14\%$ of its optimal weight. This would explain
poor PPL at high N with 1/N scaling.

### 2.5 Expected Behavior at Scale

| N | $1/N$ | $1/\sqrt{N}$ | $N^{-0.52}$ | Regime |
|---|-------|--------------|-------------|--------|
| 5 | 0.200 | 0.447 | 0.431 | All strategies viable |
| 10 | 0.100 | 0.316 | 0.295 | 1/N starts to over-dilute |
| 25 | 0.040 | 0.200 | 0.178 | 1/N is 4.5x below optimal |
| 50 | 0.020 | 0.141 | 0.123 | 1/N is 6.2x below optimal |

**Prediction:** At N=50, $1/\sqrt{N}$ will produce lower PPL than 1/N because
1/N over-dilutes each adapter's contribution. The grid search should find an
optimal weight near $0.12-0.14$ (consistent with $1/\sqrt{N}$ or $N^{-0.52}$).

## 3. Connection to Existing Methods

### 3.1 TIES-Merging (Yadav et al., 2023)

TIES resolves sign conflicts via trim-elect-merge. This addresses a **different**
failure mode than our PPL explosion:
- **TIES problem:** Sign disagreement between deltas cancels useful signal
- **Our problem:** Magnitude explosion from constructive interference

TIES uses a uniform scaling parameter $\lambda$ (their notation) that serves the
same purpose as our $\alpha$. Their recommended range is $\lambda \in [0.3, 0.7]$
for 2-3 task vectors -- they do not study N > 10 or derive N-dependent scaling.

### 3.2 DARE (Yu et al., 2023)

DARE drops parameters with probability $p$ and rescales by $1/(1-p)$. The
effective scaling when composing $N$ DARE-processed deltas is:

$$\alpha_{DARE} = \frac{1}{N} \cdot \frac{1}{1-p}$$

This is equivalent to our $1/N$ scaling with a sparsity correction. DARE's
analysis does not address the orthogonality-dependent scaling regime.

### 3.3 LoRA-Flow (Wang et al., 2024)

LoRA-Flow learns per-layer per-token fusion gates. This subsumes our static
scaling, but requires training data and gate parameters that scale as $O(NdL)$.
For N=50, d=4096, L=32, this is 6.6B gate params -- clearly impractical. Our
experiment tests whether static scaling is sufficient.

### 3.4 Task Arithmetic (Ilharco et al., 2023)

Task arithmetic composes task vectors $\tau_i = \theta_i - \theta_{pre}$ via:

$$\theta_{composed} = \theta_{pre} + \lambda \sum_i \tau_i$$

with a single scalar $\lambda$ analogous to our $\alpha$. They find $\lambda$
must be tuned per composition, typically $\lambda \in [0.3, 1.0]$ for 2-8 tasks.
They do not derive a theoretical scaling law.

## 4. Falsification Criteria

### What Would Kill This

**K1 (1/sqrt(N) does not help):** If at N=50, $1/\sqrt{N}$ does not reduce PPL
by >50% vs unit weight, then the constructive interference is more severe than
the random-subspace model predicts. This would imply adapters are in Regime C
(fully correlated) and only $1/N$ works. Implications: each adapter's contribution
is capped at $1/N$, and SOLE composition has an inherent $O(1/N)$ dilution penalty.

**K2 (best PPL still bad):** If the best scaling factor (from grid search) produces
PPL > 2x the average single-expert PPL, then scaling alone cannot fix composition.
This would suggest that interference is not just a magnitude problem but a
**destructive** one (sign conflicts, cancellation of useful features). Would
motivate TIES-like sign-aware composition or per-layer routing.

**K3 (no transfer):** If the optimal weight at N=50 does not work at N=25 (within
2x), then $\beta$ is not a stable property of the adapter set. This would imply
the scaling law is not a power law in N (or $\beta$ depends on which specific
adapters are composed, not just on the count).

## 5. Computational Complexity

The experiment is inference-only (no training).

**Per composition:**
- CPU: compose $N$ safetensors files = $O(N \cdot P)$ where $P$ is total adapter params (~40M for rank-16 all-modules)
- GPU: load PeftModel + eval on $S$ samples at max length $L$ = $O(S \cdot L \cdot d^2)$

**Total experiment:**
- $|N_{values}| = 4$ (N=5,10,25,50)
- $|strategies| = 3$ (unit, mean, sqrt) + 6 grid points + 1 transfer
- Total compositions: $4 \times 3 + 6 + 1 = 19$
- Each composition: ~2 min (load + eval)
- Phase 1 (single-expert PPL): up to 50 adapters x ~1 min = ~50 min
- **Total estimated: ~90 min**

## 6. Worked Example (Micro Scale)

**Setup:** $d=64$, $r=4$, $N=4$ adapters.

Each adapter has $\|\Delta_i\|_F \approx 0.5$ (typical for small LoRA at this scale).

**Unit weight:** $\|S\|_F^2 = \sum_i \|\Delta_i\|_F^2 + 2\sum_{i<j} \text{tr}(\Delta_i^T \Delta_j)$

With pairwise cosine $\rho = 0.15$:
$\text{tr}(\Delta_i^T \Delta_j) \approx \rho \cdot \|\Delta_i\|_F \cdot \|\Delta_j\|_F = 0.15 \cdot 0.25 = 0.0375$

$\|S\|_F^2 \approx 4 \cdot 0.25 + 2 \cdot 6 \cdot 0.0375 = 1.0 + 0.45 = 1.45$

$\|S\|_F \approx 1.20$

**With $\alpha = 1/\sqrt{4} = 0.5$:** $\|\alpha S\|_F = 0.60$

This is close to a single adapter's Frobenius norm (0.5), keeping the perturbation
bounded. Compare with $1/N = 0.25$: $\|\alpha S\|_F = 0.30$ (over-diluted to 60%
of a single adapter).

**With $\alpha = 1/4 = 0.25$:** $\|\alpha S\|_F = 0.30$

Each adapter contributes only 12.5% of a single adapter's effect. At $N=50$ with
$1/N$, each adapter contributes only 2% of its effect -- effectively invisible.

## 7. Assumptions

1. **Uniform scaling.** All adapters receive the same weight $\alpha$. In practice,
   per-adapter or per-layer weights could improve quality (cf. LoRA-Flow, PPL-probe).
   This experiment tests whether uniform scaling alone is sufficient.

2. **Adapter independence.** Adapters are trained independently on different domains.
   If adapters were jointly trained (as in MoE), the interference structure would
   differ.

3. **QLoRA quantization does not interact with scaling.** The base model is loaded
   in NF4 4-bit. We assume quantization noise is additive and independent of the
   LoRA scaling factor. This may not hold if quantization errors are correlated
   with the adapter subspace.

4. **Static scaling is sufficient.** We assume a single scalar $\alpha$ per
   composition (not per-layer, per-token, or per-adapter). If this fails, the
   experiment motivates more complex routing.

5. **Adapter quality is uniform.** The power-law analysis assumes adapters are
   drawn from a similar distribution. Outliers (e.g., a poorly-trained adapter)
   would shift the optimal $\alpha$ unpredictably.
