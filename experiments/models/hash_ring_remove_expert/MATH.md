# Hash Ring Expert Removal: Mathematical Foundations

## 1. Consistent Hashing Removal Guarantee

### 1.1 Setup

Let $R = [0, 2^{32})$ be the ring space. We place $N$ experts on the ring,
each with $V$ virtual nodes. Expert $i$ occupies positions:

$$p_{i,v} = \text{FNV1a}(\text{pack}(i, v)), \quad v \in \{0, \ldots, V-1\}$$

The full ring contains $NV$ sorted positions. For a token with hash $h$,
its primary expert is:

$$\text{primary}(h) = e_j \text{ where } j = \min\{k : p_k \geq h\} \pmod{NV}$$

(walk clockwise to the first virtual node, return its expert ID).

### 1.2 Removal Displacement Bound

**Theorem (Karger et al. 1997).** Removing expert $i$ from a ring with $N$
experts displaces exactly the tokens assigned to expert $i$. All other
assignments are unchanged.

**Proof sketch.** Expert $i$'s virtual nodes are removed. For any token $h$
NOT assigned to expert $i$: its primary virtual node $p_j$ (with $e_j \neq i$)
is still on the ring, so its assignment is unchanged. For any token $h$
assigned to expert $i$: its primary virtual node $p_j$ (with $e_j = i$) is
removed, so the token walks clockwise to the next remaining virtual node. $\square$

**Corollary.** The displacement rate when removing expert $i$ equals
the load fraction of expert $i$:

$$\text{displacement}_i = \frac{|\{h : \text{primary}(h) = i\}|}{T}$$

With perfect balance, $\text{displacement}_i = 1/N$. With virtual nodes
and FNV1a, the actual load varies.

### 1.3 Neighbor Redistribution

**Theorem.** When expert $i$ is removed, 100% of its displaced tokens
go to clockwise neighbors of expert $i$'s virtual nodes.

**Proof.** Each of expert $i$'s virtual nodes $p_{i,v}$ has a clockwise
neighbor $p_{j,w}$ (the next virtual node on the ring with $j \neq i$).
All tokens in the arc $(p_{\text{prev}}, p_{i,v}]$ were assigned to expert $i$
via virtual node $p_{i,v}$. After removal, these tokens walk clockwise to
$p_{j,w}$, which belongs to expert $j$. Expert $j$ is the clockwise neighbor
of virtual node $p_{i,v}$. Therefore ALL displaced tokens go to clockwise
neighbors. $\square$

**Note:** Expert $i$ has $V$ virtual nodes, each potentially with a
different clockwise neighbor. The set of receiving experts is:

$$\mathcal{N}_i = \{e_j : (p_{i,v}, e_j) \text{ is a clockwise neighbor pair}\}$$

With $V = 150$ and $N = 8$, typically $|\mathcal{N}_i| = 2\text{-}4$ distinct
experts receive the displaced tokens.

### 1.4 Load Balance

The expected load fraction for expert $i$ with $V$ virtual nodes on a ring
of $NV$ total nodes:

$$\mathbb{E}[\text{load}_i] = \frac{1}{N}$$

The variance depends on $V$. For large $V$ and uniform hash:

$$\text{Var}[\text{load}_i] \approx \frac{1}{NV}$$

The standard deviation of load fraction is $\sigma \approx 1/\sqrt{NV}$.
At $N=8, V=150$: $\sigma \approx 0.029$, so 95% CI is $\pm 0.057$ around
$1/8 = 0.125$. This predicts load fractions in $[0.068, 0.182]$.

**Empirical observation:** FNV1a with sequential expert IDs produces
worse-than-random load balance. At $N=8, V=150$:

| Expert | Load | Ratio to 1/N |
|--------|------|-------------|
| 0 | 22.5% | 1.80x |
| 1 | 18.0% | 1.44x |
| 4 | 8.9% | 0.71x |
| 7 | 8.1% | 0.65x |

This is outside the 95% CI for a truly uniform hash, indicating FNV1a
has correlations for packed integer inputs. For production use, a better
hash function (e.g., xxHash, MurmurHash3) would improve balance.

## 2. Quality Degradation Model

### 2.1 Quality Matrix

Define expert quality matrix $Q \in \mathbb{R}^{N \times N}$ where
$Q_{ij}$ = quality of expert $i$ on domain $j$:

$$Q_{ij} = \begin{cases}
1 & \text{if } i = j \text{ (own domain)} \\
1 - s \cdot d(i,j) + \epsilon & \text{otherwise}
\end{cases}$$

where $s \in [0, 1]$ is specialization strength, $d(i,j) = \min(|i-j|, N-|i-j|)/(N/2)$
is normalized circular distance, and $\epsilon \sim \mathcal{N}(0, 0.02)$.

### 2.2 Aggregate Quality

Before removal, each token $h$ is served by expert $\text{primary}(h)$
on domain $\text{primary}(h)$ (assuming tokens match their routing expert's domain):

$$Q_{\text{before}} = \frac{1}{T} \sum_{h} Q_{\text{primary}(h), \text{primary}(h)} = 1$$

After removing expert $i$, tokens formerly on expert $i$ move to their
neighbor expert $n(h)$:

$$Q_{\text{after}} = \frac{1}{T} \left[\sum_{h \notin E_i} Q_{\text{primary}(h), \text{primary}(h)} + \sum_{h \in E_i} Q_{n(h), i}\right]$$

where $E_i = \{h : \text{primary}(h) = i\}$. The degradation is:

$$\Delta Q = Q_{\text{after}} - Q_{\text{before}} = \frac{|E_i|}{T} \cdot (\bar{Q}_{n,i} - 1)$$

$$\text{degradation} = \frac{|E_i|}{T} \cdot (1 - \bar{Q}_{n,i})$$

where $\bar{Q}_{n,i}$ is the average quality of neighbor experts on domain $i$.

### 2.3 Degradation Bounds

At specialization $s$:
- **Best case** (neighbor is adjacent domain): degradation $\approx \frac{1}{N} \cdot s \cdot \frac{2}{N}$
- **Worst case** (neighbor is maximally distant): degradation $\approx \frac{1}{N} \cdot s$
- **Expected** (random neighbor): degradation $\approx \frac{1}{N} \cdot s/2$

At $N=8, s=0.3$: expected degradation $\approx 1.875\%$, worst case $\approx 3.75\%$.
Both well under the 5% kill threshold.

### 2.4 Scaling

Degradation scales as $O(s/N)$: larger expert count reduces impact.

| N | Predicted degradation ($s=0.3$) | Measured |
|---|--------------------------------|----------|
| 4 | ~3.75% | ~3.07% (mean, mid) |
| 8 | ~1.88% | ~1.46% (mean, mid) |
| 16 | ~0.94% | ~1.51% (mean, mid) |
| 32 | ~0.47% | ~0.32% (mean, mid) |

## 3. Add-Remove Symmetry

**Theorem.** Add and remove are exact inverses on the hash ring.

Let $R_N$ be the ring with experts $\{0, \ldots, N-1\}$.
Let $R_{N-1}$ be the ring with expert $i$ removed.
Let $R_N'$ be $R_{N-1}$ with expert $i$ re-added.

Then $R_N = R_N'$ (identical ring positions), and for all tokens $h$:
$\text{primary}_{R_N}(h) = \text{primary}_{R_N'}(h)$.

**Empirically verified:** 100,000/100,000 roundtrip identity.

## 4. Computational Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Build ring | $O(NV \log NV)$ | $O(NV)$ |
| Remove expert | $O(NV)$ (filter) | $O((N-1)V)$ |
| Route one token | $O(\log NV + k)$ | $O(1)$ |
| Route $T$ tokens | $O(T(\log NV + k))$ | $O(T)$ |

At production scale ($N=500, V=150$): ring has 75,000 entries.
Remove is a single O(75K) filter. Routing is O(log 75K) = O(16) per token.

## 5. Assumptions

1. **Token hashes are approximately uniform** on $[0, 2^{32})$.
   Justified by FNV1a applied to float projections.
2. **Expert quality decreases with domain distance.** This is the
   standard MoE assumption. If all experts are equally good at all
   domains, removal has zero quality impact.
3. **Virtual node count is sufficient for balance.** V=150 provides
   adequate balance at N=8-32. Larger N may need V=300+.
4. **Single removal.** Results are for removing one expert at a time.
   Sequential removal of K experts is bounded by $\sum_{n=N}^{N-K+1} 1/n$.

## 6. Worked Example

$N=8$ experts, $V=150$ virtual nodes, remove expert 4:

- Ring has $8 \times 150 = 1200$ virtual nodes
- Expert 4 has 150 virtual nodes scattered across the ring
- Each virtual node has a clockwise neighbor (from 3 distinct experts: 2, 5, 7)
- Removal: filter out expert 4's 150 nodes, ring now has 1050 nodes
- Tokens formerly routed to expert 4 (8.87% of total) walk clockwise:
  - 44.5% go to expert 2
  - 14.6% go to expert 5
  - 40.9% go to expert 7
- Quality degradation: -1.43% (at specialization $s=0.3$)
- Zero false moves: no token that was NOT on expert 4 changes assignment
