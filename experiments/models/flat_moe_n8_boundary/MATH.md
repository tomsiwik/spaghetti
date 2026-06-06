# Flat MoE N=8 Boundary: Mathematical Foundations

## 1. Problem Statement

Prior experiments established two facts about N-domain composition:
1. **Identity preservation** (n8_identity_boundary): Combined Jaccard = 0.800
   at N=8 pre-calibration, with sublinear degradation (~0.014/domain).
2. **Flat MoE is the only viable N>2 strategy** (sequential_freeze_graft):
   Sequential grafting degrades 3.65x at N>2.

What is missing: the **composition gap** (quality loss vs joint training) at
N=8 using the full flat MoE protocol (concatenation + calibration). The N=5
combined experiment showed +1.6% gap (seq_hybrid) and +3.32% (par_pure_linear).
Scaling to N=8 could reveal a quality phase transition.

Additionally, measuring Jaccard **post-calibration** (not just post-concatenation)
tests whether calibration preserves or destroys the identity signal that enables
pre-composition profiling.

---

## 2. Notation

All notation follows n5_identity_scaling/MATH.md and n8_identity_boundary/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
L         -- number of transformer layers (4 at micro scale)
N         -- number of composed domains (8 in this experiment)
P_comp    -- total capsules in composed model = N * P = 1024 per layer

a_i in R^d   -- detector vector for capsule i (row i of matrix A)
b_i in R^d   -- output vector for capsule i (column i of matrix B)

MLP_comp(h)  = B_comp * ReLU(A_comp * h)
             = sum_{k=0}^{N-1} B_k * ReLU(A_k * h)

joint_loss   -- val loss of model trained jointly on all domains
comp_loss    -- val loss of composed model (post-calibration)
gap          -- (comp_loss - joint_loss) / joint_loss * 100%
```

---

## 3. Composition Gap Scaling

### 3.1 Expected Gap Behavior

The composition gap arises from the function-space discrepancy:

```
f_composed(x) = f_base(x) + sum_{k=0}^{N-1} MLP_k(h)
f_joint(x)    = f_base(x) + MLP_joint(h)

gap = ||f_composed - f_joint|| / ||f_joint||
```

With more domains, the residual sum `sum_k MLP_k(h)` has more terms. Without
calibration, this produces interference. Calibration (fine-tuning on mixed
data) reduces the gap but cannot eliminate it in finite steps.

### 3.2 Calibration Budget Scaling

At N=2, 100 calibration steps suffice for -0.3% gap. At N=5, 200 steps give
+1.6%. The relationship between calibration steps and N is not established.

For this experiment, we use 300 calibration steps at N=8 (linear scaling:
100 * 8/2 = 400 would be conservative; we use 300 as a moderate choice).

### 3.3 Gap Bound

**No useful analytical bound is available.** An earlier version stated
`gap <= O(N * P / P_joint) * residual_noise`, but since the composed model
has P_comp = N * P capsules and the joint model also has P_joint = N * P
capsules by construction, the capacity ratio N * P / P_joint = 1 and the
bound reduces to O(1) * residual_noise, which provides no constraint.

A tighter bound would need to depend on the calibration budget (number of
gradient steps) and the function-space discrepancy between independently
fine-tuned domains and the joint configuration. Deriving such a bound
requires characterizing the optimization landscape of the calibration
objective, which is beyond the scope of this micro-scale experiment.

Empirically, with a consistent protocol across N, the calibrated gap is
~+6% at N=2,5 and ~+7.4% at N=8 (see PAPER.md for the full trajectory).
The earlier apparent linear trend (+1%/domain) was an artifact of comparing
results from different experiments with different protocols.

---

## 4. Post-Calibration Identity Tracking

### 4.1 Key Distinction

Prior Jaccard measurements (Exp 16, n5_identity_scaling, n8_identity_boundary)
measured identity **immediately after concatenation** (zero-shot composition).
This experiment measures identity **after calibration** (300 steps of mixed
training).

Calibration modifies all weights, including the A and B matrices of each
capsule pool. This can:
1. Kill additional capsules (calibration pushes borderline-alive capsules
   past the death boundary)
2. Revive dead capsules (calibration shifts inputs to activate previously
   dead detectors)
3. Fundamentally reshuffle which capsules are dead vs alive

### 4.2 Expected Jaccard Under Calibration

Let D^{single}_k be the dead set from single-domain profiling, and
D^{cal-comp} be the dead set after calibrated composition.

```
D^{cal-comp} = D^{zero-comp} UNION D^{cal-killed} MINUS D^{cal-revived}
```

where D^{cal-killed} and D^{cal-revived} are the sets of capsules killed
and revived during calibration.

If calibration is aggressive (many gradient steps, high learning rate),
the Jaccard J(D^{single}_k, D^{cal-comp}_k) can be much lower than the
zero-shot J(D^{single}_k, D^{zero-comp}_k).

### 4.3 Death Rate Amplification

At N=8, the composed model has P_comp = 1024 capsules per layer. Each
input token produces a d=64 hidden state that must activate a subset
of these 1024 capsules. With P/d = 1024/64 = 16, the representation is
severely overcomplete. Under ReLU, most capsules will be dead for most
inputs.

Expected death rate from overcomplete representation:

```
death_rate ~ 1 - d / P_comp = 1 - 64/1024 = 93.75%
```

This is much higher than the ~50% death rate in single-domain models
(where P/d = 128/64 = 2, a mild overcomplete ratio).

---

## 5. Worked Numerical Example (Illustrative, Not From Experimental Data)

*Note: This example uses toy numbers (d=4, P=4, N=4) to illustrate the
mechanism. The values are constructed to show the qualitative behavior of
calibration-induced identity loss. They do not correspond to actual
experimental measurements. See PAPER.md and results.json for empirical data.*

At d=4, P=4, L=1, N=4 (4 capsules per domain, 16 total composed):

### Single-domain dead sets
```
Domain 0: D^{single}_0 = {0, 1}    (50% dead)
Domain 1: D^{single}_1 = {2, 3}    (50% dead)
Domain 2: D^{single}_2 = {0, 3}    (50% dead)
Domain 3: D^{single}_3 = {1, 2}    (50% dead)
```

### Zero-shot composition (pre-calibration)
```
Composed (16 capsules): D^{zero} = {0, 1, 6, 7, 8, 11, 13, 14}
Death rate: 8/16 = 50%
Combined Jaccard (zero): 8/9 = 0.889
```

### Post-calibration composition
After 300 calibration steps, additional capsules die:
```
D^{cal} = {0, 1, 3, 5, 6, 7, 8, 10, 11, 13, 14, 15}
Death rate: 12/16 = 75%
Combined Jaccard (cal): |D^{single-union} INTERSECT D^{cal}| / |D^{single-union} UNION D^{cal}|
                      = 8/12 = 0.667
```

Calibration pushes Jaccard down by creating additional kills not predicted
by single-domain profiling.

---

## 6. Kill Criteria

```
Criterion 1: Composition gap > 10% at N=8
Criterion 2: Combined Jaccard < 0.60
```

If criterion 1 kills: Flat MoE composition produces unacceptable quality
loss at N=8. The calibration budget must be increased or the protocol
redesigned.

If criterion 2 kills: Pre-composition profiling is unreliable at N=8.
Post-composition profiling is mandatory. The pre-composition pipeline
validated at N=2 and N=5 does not extend to N=8.

---

## 7. Assumptions

1. **Same composition protocol as prior experiments.** Pretrain shared base
   (300 steps), fine-tune MLP per domain (200 steps, attention frozen),
   compose by concatenation, calibrate on mixed data (300 steps).

2. **Joint training uses same total compute.** Joint model trains for
   N * STEPS_FINETUNE = 1600 steps on round-robin mixed data. Same total
   gradient updates.

3. **Calibration budget is reasonable.** 300 steps is linearly scaled from
   100 at N=2. Insufficient calibration could artificially inflate the gap.
   However, overfitting to calibration data is also a concern.

4. **Octonary split produces adequate domain diversity.** 8 character-level
   domains are more similar to each other than real macro domains.

5. **Post-calibration Jaccard is the relevant metric.** Pre-composition
   profiling is only useful if the pruning decisions remain valid after
   calibration. If calibration reshuffles the dead set, pre-composition
   profiling becomes unreliable.
