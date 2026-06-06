# Mixed-Domain Sequences: Mathematical Foundations

## Setup

Let a mixed-domain sequence $\mathbf{x} = [x_1, \ldots, x_T]$ consist of two
contiguous segments from domains $A$ and $B$:

$$\mathbf{x} = [\underbrace{x_1, \ldots, x_b}_{\text{domain } A}, \underbrace{x_{b+1}, \ldots, x_T}_{\text{domain } B}]$$

where $b$ is the boundary position.

We have $N = 5$ domain-specialized LoRA adapters $\{\Delta W_1, \ldots, \Delta W_N\}$,
each of rank $r = 16$ on a $d = 2560$ dimensional model (BitNet-2B-4T).

## Routing Conditions

### Per-sequence routing (baseline)

The router sees the mean-pooled hidden state $\bar{h} = \frac{1}{T}\sum_{t=1}^T h_t$
and selects top-$k$ experts globally:

$$\hat{y}_t = (W + \sum_{i \in \text{top-}k(\bar{h})} w_i \Delta W_i) \cdot h_t \quad \forall t$$

The weights $w_i = \frac{\sigma(f(\bar{h}))_i}{\sum_{j \in \text{top-}k} \sigma(f(\bar{h}))_j}$
are constant across all positions.

**Problem**: For mixed sequences, $\bar{h}$ is the average of domain-$A$ and domain-$B$
representations. If domains are well-separated in embedding space, this mean
falls in neither domain's region, leading to suboptimal expert selection.

### Per-token routing (experiment)

Each position gets its own expert selection:

$$\hat{y}_t = (W + \sum_{i \in \text{top-}k(h_t)} w_{t,i} \Delta W_i) \cdot h_t$$

where $w_{t,i}$ varies per position. Tokens $t \leq b$ should predominantly
select expert $A$; tokens $t > b$ should select expert $B$.

### Oracle routing (upper bound)

Uses perfect domain knowledge:

$$\hat{y}_t = \begin{cases} (W + \Delta W_A) \cdot h_t & \text{if } t \leq b \\ (W + \Delta W_B) \cdot h_t & \text{if } t > b \end{cases}$$

## Expected Advantage

The per-token advantage over per-sequence scales with **domain separability** in
hidden space. Let $\mu_A, \mu_B$ be the mean hidden states for each segment:

$$\text{separability} = \frac{\|\mu_A - \mu_B\|_2}{\frac{1}{2}(\sigma_A + \sigma_B)}$$

When separability is high (distinct domains like code vs. medical), the router
can detect the boundary and route correctly. When low (similar domains), per-token
reduces to noisy per-sequence (as observed in exp_molora_per_token_mlx).

## Kill Criteria Formalization

**K1**: Let $\text{PPL}_{\text{pt}}$ and $\text{PPL}_{\text{ps}}$ be the per-token
and per-sequence perplexities on mixed sequences:

$$\frac{\text{PPL}_{\text{ps}} - \text{PPL}_{\text{pt}}}{\text{PPL}_{\text{ps}}} < 0.05 \implies \text{KILL}$$

**K2**: Let $a_t$ be the primary expert selected at position $t$. Define boundary
accuracy as:

$$\text{BA} = \frac{1}{2}\left(\frac{\sum_{t \leq b} \mathbb{1}[a_t = A]}{b} + \frac{\sum_{t > b} \mathbb{1}[a_t = B]}{T - b}\right)$$

Random baseline: $\text{BA}_{\text{random}} = 1/N = 0.2$. Threshold: $\text{BA} \leq 0.4$
(less than $2\times$ random) $\implies$ KILL.

## Computational Cost

- Router: $2560 \times 64 + 64 \times 5 = 164,160$ params (0.58% overhead proven)
- Per-token routing requires $|\text{unique expert sets}|$ forward passes per sequence
  (vs. 1 for per-sequence). In practice, 2-4 unique sets observed.
- Oracle requires 2 forward passes per sequence (one per segment)

## Worked Example (micro scale)

$d = 2560$, $N = 5$, $T = 256$, boundary at $b = 128$.

Sequence: [128 Python tokens | 128 Math tokens]

- Per-sequence: mean-pool $\to$ selects python+math (if lucky) or python+creative (if unlucky)
  - Both halves get same adapter mixture
- Per-token: positions 0-127 select python+{something}, positions 128-255 select math+{something}
  - Each half gets appropriate adapter
- Oracle: positions 0-127 get pure python, 128-255 get pure math

If python adapter reduces PPL on code by 30% and math adapter reduces PPL on math
by 25%, but per-sequence router splits the budget: per-token should capture ~55%
of oracle advantage while per-sequence captures ~30%.

## Assumptions

1. Hidden states at position $t$ primarily reflect the local domain (not corrupted
   by cross-attention to tokens from the other domain)
2. Domain boundaries in token sequences are sharp (not gradual transitions)
3. The LoRA adapters are sufficiently domain-specialized that wrong-domain composition
   degrades performance
4. The router has enough capacity (64-dim hidden) to learn domain boundaries from
   hidden states
