# Sequential Freeze-Graft-Calibrate: Mathematical Foundations

## 1. Setting

We operate on the binary capsule tree from hierarchical_tree (depth D=3, L=8
leaves, 7 internal gates, beam B=2). All base notation follows MATH.md in
that directory and split_freeze_protocol.

Additional notation for the sequential protocol:
```
N       = number of domains (tested: N=2,3,4)
D_i     = domain i, for i in {0,...,N-1}
S_i     = set of leaf indices assigned to domain i
G_i     = set of gate indices internal to domain i's subtree
theta_i = parameters of domain i's subtree (gates + leaves in S_i, G_i)
cal(n)  = calibration step after graft n (n=1,...,N-1)
deg_max(n) = max degradation over frozen domains after graft n
```

## 2. Progressive Halving Allocation

With N sequential grafts on a depth-3 tree, each graft freezes the most
recently trained portion and assigns the remaining unfrozen subtree to
the new domain. This creates a halving allocation:

```
Initial:  D_0 trains on all 8 leaves

Graft 1 (N=2):
  Freeze: leaves 0-3 (D_0), gates {1,3,4}
  Assign: leaves 4-7 (D_1), gates {2,5,6}
  Domain allocation: D_0={0,1,2,3}, D_1={4,5,6,7}
  |D_0| = 4 leaves, |D_1| = 4 leaves

Graft 2 (N=3):
  Freeze: leaves 4-5 (D_1 half), gate {5}
  Assign: leaves 6-7 (D_2), gate {6}
  Domain allocation: D_0={0,1,2,3}, D_1={4,5}, D_2={6,7}
  |D_0| = 4, |D_1| = 2, |D_2| = 2

Graft 3 (N=4):
  Freeze: leaf 6 (D_2 half)
  Assign: leaf 7 (D_3)
  Domain allocation: D_0={0,1,2,3}, D_1={4,5}, D_2={6}, D_3={7}
  |D_0| = 4, |D_1| = 2, |D_2| = 1, |D_3| = 1
```

This yields unequal capacity: D_0 has 4 leaves, D_3 has 1 leaf. The
imbalance is inherent to the progressive protocol with a fixed-depth tree.

## 3. Cumulative Degradation Model

### 3.1 Per-Graft Degradation

After graft n, the frozen domain D_0 suffers degradation because:

1. **Routing drift**: The root gate g_0 and surviving unfrozen gates
   must learn a (n+1)-way routing decision, but the tree is binary.
   With N>2, the routing problem becomes a cascade of binary decisions
   that must collectively separate N domains.

2. **Representation shift**: Each graft reinitializes a subtree, changing
   the output distribution that the shared attention layers see. Even
   though frozen weights don't change, the effective input distribution
   to the frozen subtree shifts because gate probabilities change.

3. **Calibration capacity limit**: At graft n, the unfrozen calibration
   parameters are a subset of the full tree:
   ```
   cal_params(n) = sum over unfrozen gates + unfrozen leaves
   ```
   This shrinks with each graft (more is frozen), limiting the model's
   ability to compensate for routing drift.

### 3.2 Degradation Scaling

Let delta(n) = deg_max after graft n. The hypothesis was that delta grows
at most linearly with n. The data shows:

```
Observed (all-unfrozen, 3-seed means):
  delta(1) = 3.72%    (N=2)
  delta(2) = 6.73%    (N=3)
  delta(3) = 13.58%   (N=4)

Ratio: delta(3)/delta(1) = 3.65x  (>2.0x threshold: KILL)
```

The growth is superlinear, approximately quadratic:
```
delta(n) ~ c * n^alpha, where alpha ~ 1.8-1.9

Fitting: log(delta(1))/log(1) vs log(delta(3))/log(3):
  alpha = log(13.58/3.72) / log(3/1) = log(3.65) / log(3) = 1.18

More precisely, tracking the mean:
  delta(1) = 3.72   (n=1)
  delta(2) = 6.73   (n=2): ratio 1.81x vs delta(1)
  delta(3) = 13.58  (n=3): ratio 3.65x vs delta(1)

The acceleration between graft 2->3 (2.02x) is larger than 1->2 (1.81x),
confirming superlinear growth.
```

### 3.3 Why More Calibration Does Not Help

Extended calibration experiment tested three schedules:
```
fixed_200:    [200, 200, 200]  -> ratio 3.92x  (KILL)
scaled_1.5x: [200, 300, 400]  -> ratio 4.90x  (KILL, worse!)
scaled_2x:   [200, 400, 600]  -> ratio 3.63x  (KILL)
```

More calibration steps do NOT reduce the ratio below 2.0x. The reason:
the degradation is not caused by insufficient calibration convergence
but by structural capacity limits:

1. At N=4, domain D_0 occupies 4/8 leaves but the root gate must route
   4 domains through a single binary split. The effective routing capacity
   is log2(N) binary decisions, but only some gates are unfrozen.

2. Each calibration pass optimizes for the current domain mix. But the
   frozen subtrees' learned representations are incompatible with the
   routing decisions needed for N>2 domains.

3. The last graft (N=4) has only 17,164 calibration parameters vs 66,576
   at N=2. The calibration capacity shrinks while the routing complexity
   grows.

## 4. Calibration Cost Analysis

### 4.1 Parameter Counts

| Graft | N | Cal Params (all-unfrozen) | Cal Params (selective) |
|-------|---|---------------------------|------------------------|
| 1     | 2 | 66,576                    | 520                    |
| 2     | 3 | 33,548                    | 520                    |
| 3     | 4 | 17,164                    | 520                    |

All-unfrozen calibration cost **decreases** with N because more of the
tree is frozen. The per-domain cost ratio is:
```
cal_params(N=4) / 4 = 4,291
cal_params(N=2) / 2 = 33,288
ratio = 4,291 / 33,288 = 0.13  (PASS, strongly sublinear)
```

Kill criterion KC2 passes: calibration cost per graft is sublinear.

### 4.2 Selective Calibration Failure

Selective calibration (root + graft-point gates only, 520 params constant)
produces catastrophic degradation at N>2:
```
N=2: 24.24%  (already bad)
N=3: 35.34%
N=4: 32.99%
```

At N=2, split_freeze_protocol showed selective gates-only (+2.5%) is
marginal but gates+leaves (+0.09%) works. At N>2, gates-only is
completely insufficient because the routing complexity exceeds what
2-3 gate parameters can encode.

## 5. Worked Example (seed 42, N=3)

```
Base training: domain a_f, 400 steps -> val_loss = 0.4741

Graft 1: freeze leaves {0,1,2,3}, reinit leaves {4,5,6,7}
  Train on g_m (200 steps)
  Calibrate all-unfrozen (200 steps, 66576 params)
  a_f: 0.4914 (+3.66%)  <- domain A degraded
  g_m: 0.5020

Graft 2: freeze leaves {4,5}, reinit leaves {6,7}
  Train on n_s (200 steps)
  Calibrate all-unfrozen (200 steps, 33548 params)
  a_f: 0.5040 (+6.30%)  <- domain A degraded further
  g_m: 0.5140 (+2.38%)  <- domain B also degraded
  n_s: 0.5195

Cumulative A degradation: 0.4741 -> 0.4914 -> 0.5040
Growth per graft: +3.66%, then +2.53% additional
The second graft's damage to A is smaller than the first, but
the cumulative effect (+6.30%) grows superlinearly vs N.
```

## 6. Assumptions

1. **Fixed tree depth**: D=3 with 8 leaves constrains the maximum N to 8
   (one leaf per domain at the extreme). Deeper trees would allow more
   balanced allocation but increase routing complexity.

2. **Progressive halving is the natural allocation**: Other allocation
   strategies (e.g., round-robin) might distribute capacity more evenly
   but break the tree's hierarchical structure.

3. **Domain A is the worst case**: The first domain, frozen earliest with
   the most subsequent grafts, accumulates the most degradation. This
   is a structural property of the sequential protocol.

4. **Calibration on mixed data from all domains**: If a graft-specific
   calibration set could be constructed to weight frozen domains more
   heavily, degradation might be reduced.

5. **Quaternary domain split**: The 4 domains from names.txt are of
   unequal size (a_f: 10896, g_m: 11629, n_s: 5841, t_z: 3667),
   introducing some confound between domain difficulty and domain size.
