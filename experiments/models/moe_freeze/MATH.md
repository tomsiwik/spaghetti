# MoE-Freeze: Mathematical Foundations

## Expert Specialization Metric

For an expert `i` with parameter set `{theta_i^(l)}` (all weight matrices across layers of
the MLP), specialization is measured by the sum of squared Frobenius norms:

```
S(i) = sum_l || theta_i^(l) ||_F^2
     = sum_l sum_{jk} (theta_i^(l)_{jk})^2
```

In code this is computed as:

```python
total += mx.sum(v * v).item()   # for each parameter tensor v
```

### Why Weight Norm Is a Valid Proxy

A freshly initialized expert has small weights (Xavier init scales as `1/sqrt(fan_in)`).
As an expert fits a domain, gradient descent pulls weights away from zero to encode structure.
The key intuition is:

- **Initialization**: `E[|| theta_init ||^2] = d_in * d_out * sigma_init^2` — small and fixed.
- **After learning**: weights grow to represent the learned function. A flat, undertrained
  expert retains small norms; a specialized expert accumulates larger weights.
- **Noise vs. structure**: random noise in weight space sums to near zero in expectation due
  to cancellation. Structured weights that encode a function resist this cancellation, giving
  larger `S(i)`.

This is an implicit assumption, not a theorem. It holds when:
1. Initialization is near zero (satisfied by Xavier/Kaiming).
2. The expert has received sufficient gradient signal.
3. The learning rate does not cause weight explosion unrelated to task fit.

Weight norm is a cheaper proxy than task-specific metrics (e.g., held-out loss on domain `d`)
because it requires no held-out data and no additional forward pass.

---

## Freeze Operation

Freezing sets `requires_grad = False` for all parameters of expert `i`:

```
theta_i  ->  theta_i  (no update)
```

Formally, frozen parameters are excluded from the gradient computation graph. If the optimizer
update rule is:

```
theta <- theta - lr * grad_theta L
```

then for frozen parameters `grad_theta L` is not computed (or is set to zero before the
optimizer step). The learned function `f_i(x; theta_i)` is thus **preserved exactly** for all
future training steps.

In MLX, `expert.freeze()` marks the module's subtree as non-trainable. The forward pass still
runs through frozen experts (their outputs contribute to the MoE weighted sum), but no gradient
flows back through their parameters.

---

## Recycle Operation

Recycling replaces expert `j`'s parameters with a fresh random initialization:

```
theta_j  ->  theta_j^(init)   where theta_j^(init) ~ Xavier(fan_in, fan_out)
```

In code:

```python
moe.experts[worst] = ExpertMLP(n_embd)
mx.eval(moe.experts[worst].parameters())
```

The new `ExpertMLP` uses MLX's default initialization (equivalent to Kaiming uniform for
`nn.Linear` without bias). This resets the expert to a blank slate, giving it full plasticity
for the next domain.

**What is lost**: all accumulated weight structure. The router may still route tokens to this
expert at the same rate; the output quality drops until the recycled expert re-trains.

**What is gained**: the parameter budget is freed from a low-quality expert and reallocated to
a fresh one capable of learning a new domain.

---

## The Freeze-Recycle Lifecycle

On each domain switch signal, for every MoE layer `l`:

```
best  = argmax_i  S(i)
worst = argmin_{i not in Frozen_l}  S(i)

if best not in Frozen_l:
    Frozen_l <- Frozen_l + {best}
    freeze(expert_best)

if worst is defined and worst != best:
    recycle(expert_worst)
```

The guard `worst != best` prevents recycling the expert we just froze in the same step. The
`default=None` guard handles the edge case where all experts are already frozen; in that case
no recycling occurs.

---

## Capacity Analysis

Let `N` be the number of experts per layer and `M` be the number of domain switches observed.

After `M` switches, the maximum number of frozen experts per layer is:

```
|Frozen_l| <= min(N, M)
```

Because at most one expert is frozen per switch, and there are only `N` slots. Once `|Frozen_l| = N`,
the lifecycle silently stops freezing (the `if best not in Frozen_l` guard fires false) and no
recycling occurs (the `worst` computation yields `None` since all unfrozen candidates are
exhausted).

The number of **active (trainable) experts** at switch `m` is:

```
A(m) = N - |Frozen_l(m)|  >= max(0, N - m)
```

Active experts provide plasticity; frozen experts provide stability.

---

## Continual Learning Connection

Continual learning seeks to learn a sequence of tasks `T_1, T_2, ..., T_M` without forgetting
earlier tasks. The central tension is the **stability-plasticity dilemma**:

- **Stability**: preserve knowledge of past tasks.
- **Plasticity**: adapt to new tasks.

The MoE-Freeze lifecycle maps directly onto this dilemma:

| Concept          | Lifecycle implementation                  |
|------------------|-------------------------------------------|
| Episodic memory  | Frozen experts (immutable, task-specific) |
| Plasticity       | Recycled and unfrozen experts (trainable) |
| Memory capacity  | `N` expert slots per layer                |
| Forgetting       | Overwriting an active expert's weights    |

The lifecycle acts as **homeostatic cleanup**: it does not improve peak accuracy but reduces
destructive interference by protecting highly-specialized experts from overwriting.

---

## The Stability-Plasticity Trade-off

Define the frozen ratio at switch `m` as:

```
rho(m) = |Frozen_l(m)| / N
```

- `rho = 0`: full plasticity, maximum forgetting risk.
- `rho = 1`: full stability, no capacity for new learning.

The lifecycle drives `rho` upward monotonically. Under a fixed expert budget `N` and domain
sequence of length `M`, the system reaches full stability (`rho = 1`) after `N` switches and
then provides no further benefit.

**When lifecycle helps**: `N >= M` (slack capacity). There are enough experts that each domain
can claim at least one frozen expert without competing for slots.

**When lifecycle is zero-sum**: `N < M` (constrained capacity). Freezing experts for early
domains steals capacity from late domains. The preservation-coverage trade-off: early tasks
are preserved at the cost of coverage for later tasks. Net forgetting reduction can be near
zero or even negative.
