# Cosine Convergence Trajectory: Mathematical Analysis

## 1. Setup and Notation

Let $W_b \in \{-1, 0, 1\}^{d_{out} \times d_{in}}$ be the frozen ternary base weight matrix (BitNet-2B-4T, $d = 2560$).

For domain $i$, the LoRA adapter consists of:
$$\Delta W_i = B_i A_i, \quad A_i \in \mathbb{R}^{r \times d_{in}}, \quad B_i \in \mathbb{R}^{d_{out} \times r}$$
with $r = 16$.

The "adapter vector" for cosine computation is:
$$v_i = \text{vec}(\Delta W_i) \in \mathbb{R}^{d_{out} \cdot d_{in}}$$

flattened across ALL layers and projection types (q, k, v, o, gate, up, down across 30 layers = 210 projections).

## 2. Cosine Similarity Between Adapters

$$\cos(v_i, v_j) = \frac{\langle v_i, v_j \rangle}{\|v_i\| \|v_j\|}$$

We measure $|\cos(v_i, v_j)|$ because sign is arbitrary (a negated adapter produces the same function up to output sign).

**Dimension of $v_i$**: Each projection has $d_{out} \times d_{in}$ parameters in the weight matrix, but LoRA stores only $A_i$ ($r \times d_{in}$) and $B_i$ ($d_{out} \times r$). The cosine is computed over the concatenation of all LoRA A and B matrices:
$$v_i = [A_i^{(1)}, B_i^{(1)}, A_i^{(2)}, B_i^{(2)}, \ldots] \in \mathbb{R}^{D}$$
where $D = 210 \times (r \cdot d_{in} + d_{out} \cdot r)$ summed over all projection dimensions.

For BitNet-2B-4T: $D \approx 20.5M$ parameters per adapter.

## 3. Random Baseline

For random vectors $v_i, v_j \in \mathbb{R}^D$, the expected $|\cos|$ is:
$$\mathbb{E}[|\cos|] = \sqrt{\frac{2}{\pi D}} \approx \sqrt{\frac{2}{\pi \cdot 20.5 \times 10^6}} \approx 1.76 \times 10^{-4}$$

Our measured $|\cos| = 0.001$ at 400 steps is $\sim 5.7\times$ the random baseline. This excess is expected: all adapters share the same base model and optimizer dynamics.

## 4. Why Cosine Might Inflate During Training

### Hypothesis A: Shared Feature Space Convergence
As training progresses, all adapters learn to use the same "important" subspace of the weight space. The base model's pre-trained representations create common directions that all domains exploit.

$$v_i(t) = v_i^{\text{domain}} + v_i^{\text{shared}}(t)$$

If $\|v_i^{\text{shared}}(t)\|$ grows relative to $\|v_i^{\text{domain}}\|$, then:
$$|\cos(v_i(t), v_j(t))| \to \frac{\|v^{\text{shared}}(t)\|^2}{\|v^{\text{shared}}(t)\|^2 + \|v^{\text{domain}}\|^2} \to 1$$

### Hypothesis B: Ternary Constraint Prevents Inflation
BitNet's ternary base weights create a discrete signal pathway. The adapter perturbation interacts with base weights via:
$$y = (W_b + \sum_i \alpha_i B_i A_i) x$$

Because $W_b \in \{-1, 0, 1\}$, the Jacobian $\partial y / \partial (B_i A_i)$ is piecewise constant, not smooth. This discreteness may prevent the smooth gradient flow that normally causes adapters to converge to a shared subspace.

**Key insight**: In FP16 bases, the gradient $\nabla_{\theta_i} \mathcal{L}$ has a component from $W_b$'s learned directions that is shared across domains. In ternary bases, $W_b$ contributes only discrete routing (sign/zero), so the shared component is weaker.

## 5. Convergence Detection

We detect convergence via loss plateau:
$$\text{converged}(t) = \frac{\bar{\mathcal{L}}[t-2w, t-w] - \bar{\mathcal{L}}[t-w, t]}{\bar{\mathcal{L}}[t-2w, t-w]} < \epsilon$$

with $w = 200$ steps, $\epsilon = 0.01$ (1% improvement threshold).

## 6. Kill Criteria Formalization

**K1**: $\bar{|\cos|}(t^*) \geq 0.05$ where $t^*$ is the convergence step (or $t^* = 2000$ if no convergence).

**K2**: Let $c(t) = \bar{|\cos|}(t)$ for checkpoint steps $t_1 < t_2 < \ldots < t_n$. The trajectory is "monotonically increasing" if:
$$\frac{|\{k : c(t_{k+1}) > c(t_k)\}|}{n-1} > 0.8$$
AND the coefficient of variation in the second half exceeds 0.3 (i.e., no plateau has formed).

## 7. Worked Example

At micro scale ($d = 2560$, $r = 16$, $N = 5$):
- 10 adapter pairs to compare
- $D \approx 20.5M$ parameter dimensions per adapter
- Random baseline $|\cos| \approx 1.76 \times 10^{-4}$
- Measured at 400 steps: $|\cos| \approx 0.001$ ($5.7\times$ random)
- If trajectory plateaus at $0.003$ by step 800: K1 PASS ($0.003 < 0.05$)
- If trajectory reaches $0.06$ by step 2000: K1 KILL ($0.06 > 0.05$)

## 8. Computational Cost

- Training: $5 \times 2000$ steps at $\sim 0.5$s/step = $\sim 5000$s = $\sim 83$ min
- Cosine checkpoints: $20 \times 10$ pairs, each requiring $2 \times 20.5M$ flattens + dot product = negligible
- PPL evaluation: $6$ checkpoints $\times 5$ domains $\times 25$ batches $= 750$ forward passes $\approx 15$ min
- **Total: $\sim 100$ minutes**

## 9. Assumptions

1. **Adapter parametrization**: We measure cosine over raw LoRA parameters $(A_i, B_i)$, not over the effective delta $\Delta W_i = B_i A_i$. This is consistent with all prior SOLE experiments.
2. **Single seed**: Justified by multiseed CV=0.5% at N=5 (bitnet_multiseed_validation).
3. **seq_len=128**: Shorter than production but sufficient for convergence dynamics.
4. **Loss-based convergence**: We use training loss plateau, not validation loss, for convergence detection. Overfitting would INCREASE cosine (adapters fitting noise in the same way), so this is conservative.
5. **FP16 LoRA on ternary base**: We do NOT use ternary QAT+STE for the adapters in this experiment. We use standard FP16 LoRA to isolate the base model's effect on orthogonality convergence.
