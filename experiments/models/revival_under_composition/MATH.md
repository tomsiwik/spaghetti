# Revival Dynamics Under Composition: Mathematical Foundations

## 1. Problem Statement

Experiment 18 established that 28.1% of capsules dead at S=100 revive
by S=3200 under single-domain fine-tuning. Experiment 20 showed that
inter-layer coupling (upstream weight updates shifting downstream input
distributions) drives 79-94% of this revival. Experiment 16 showed that
the SAME capsules die in single-domain and composed models (Jaccard=0.895).

These findings leave an open question: when capsules live in a COMPOSED
model (concatenated pools from multiple domains), does the revival rate
change? Two competing mechanisms:

- **Suppression hypothesis**: Cross-domain gradient signals partially
  cancel, reducing the net magnitude of weight updates in upstream layers.
  This weakens inter-layer coupling and suppresses revival.

- **Amplification hypothesis**: Cross-domain inputs provide more diverse
  activation patterns, creating more pathways for dead capsules to see
  positive pre-activations. This amplifies revival.

---

## 2. Notation

All notation follows capsule_revival/MATH.md and training_duration/MATH.md.

```
d         -- embedding dimension (64 at micro scale)
P         -- capsules per domain pool (128 at micro scale)
D         -- number of domains (2 at micro scale)
L         -- number of transformer layers (4 at micro scale)
S_a       -- anchor step count (200: fine-tuning steps before composition)
S_post    -- additional training steps after anchor

a_i in R^d  -- detector vector for capsule i (row of A matrix)
f_i(S)      -- activation frequency of capsule i after S steps

D_S = {i : f_i(S) = 0}  -- dead capsule set at step S
A_S = {i : f_i(S) > 0}  -- alive capsule set at step S
```

### Composed model notation

In a composed model with D=2 domains:

```
P_composed = D * P  -- total capsules per layer in composed model (256)

A_composed in R^{P_composed x d} = [A_domain1; A_domain2]
B_composed in R^{d x P_composed} = [B_domain1, B_domain2]

-- Domain-half indexing:
  capsules [0, P-1]:    domain 1's capsules
  capsules [P, 2P-1]:   domain 2's capsules
```

---

## 3. Revival Under Composition

### 3.1 Gradient Competition

In single-domain fine-tuning, the gradient on capsule pool weights is:

```
dL/dA_i = sum_x dL/dy * dy/dh_i * dh_i/dA_i
```

where the sum is over inputs from ONE domain. In composed training on
joint data, the gradient becomes:

```
dL/dA_i = sum_{x in domain_1} (...) + sum_{x in domain_2} (...)
```

For domain 1's capsules, the domain 2 inputs contribute gradients that
are NOT aligned with domain 1's specialization. If these cross-domain
gradients partially cancel the in-domain gradients, the net weight
update magnitude decreases:

```
||Delta W_composed|| <= ||Delta W_single||    (potential cancellation)
```

Weaker weight updates -> smaller input distribution shifts downstream ->
less inter-layer coupling -> less revival.

### 3.2 Input Distribution Width

Alternatively, the composed model sees inputs from BOTH domains. At
layer l, the input distribution is:

```
x_composed^l = f_{l-1}(x^{l-1}_mixed)    where x^{l-1}_mixed includes both domains
x_single^l = f_{l-1}(x^{l-1}_single)     where x^{l-1}_single is one domain only
```

The composed input distribution has wider support (union of domain
distributions). A dead capsule i with detector a_i needs:

```
a_i^T x > 0   for some x in the profiling set
```

Wider support means more diverse x vectors, potentially more chances
for a_i^T x > 0. But the profiling is done on joint data for composed
models, so this effect is already captured.

### 3.3 Higher Death Plateau

A key interaction: composed models have HIGHER equilibrium death rates
(63% vs 53% at S=+3200 in our data). More dead capsules means more
"candidates" for revival, but the revival RATE (fraction of anchor
dead that revive) is lower. This suggests the suppression mechanism
dominates: the same inter-layer coupling that drives revival is
weakened by gradient competition.

---

## 4. Experimental Design

### 4.1 Conditions

**Condition A (SINGLE-DOMAIN)**: For each domain d in {a_m, n_z}:
1. Pretrain base model on all data (300 steps)
2. Fine-tune on domain d for S_a=200 steps (MLP only, attention frozen)
3. Profile dead/alive mask at anchor
4. Continue training on domain d for S_post in {100, 400, 800, 1600, 3200}
5. Profile dead/alive mask at each S_post
6. Compute revival: fraction of anchor dead cohort that revived

**Condition B (COMPOSED + JOINT)**:
1-2. Same pretraining and per-domain fine-tuning
3. Compose by concatenating A, B matrices from both domain models
4. Profile anchor (just-composed, no further training)
5. Continue training on JOINT data for same S_post values
6. Profile at each S_post, split masks into domain halves
7. Compute per-domain revival rates

**Condition C (COMPOSED + OWN-DOMAIN)**:
1-4. Same as Condition B
5. Continue training on domain a_m data ONLY (not joint)
6-7. Same profiling and revival computation

Condition C isolates the STRUCTURAL effect of composition (having 2x
capsules, different weight initialization) from the DATA effect (seeing
cross-domain inputs). If C shows similar revival to B, the structural
change matters more than the data distribution.

### 4.2 Anchor Point

We use S_a = 200 as anchor (STEPS_FINETUNE in the codebase). This
matches the standard composition protocol: fine-tune for 200 steps,
then compose. The anchor is the moment of composition.

For single-domain, the anchor is the same 200-step checkpoint.

### 4.3 Profiling Protocol

Standard: 20 batches x 32 samples on validation data, seed-matched.
- Single-domain: profile on own-domain val data
- Composed: profile on joint val data

### 4.4 Domain-Half Splitting

For composed models, the flat dead mask has P_composed = 2P capsules
per layer. We split by index:
```
domain_A_mask[l] = composed_mask[l * 2P : l * 2P + P]
domain_B_mask[l] = composed_mask[l * 2P + P : l * 2P + 2P]
```

This preserves capsule identity: capsule j in domain A's single-domain
model corresponds to capsule j in the composed model's A-half.

---

## 5. Revival Rate Computation

For single-domain:
```
revival_rate(S_post) = |D_anchor & A_{S_post}| / |D_anchor|
```

For composed (per domain half):
```
revival_rate_A(S_post) = |D_anchor_A & A_{S_post}_A| / |D_anchor_A|
revival_rate_B(S_post) = |D_anchor_B & A_{S_post}_B| / |D_anchor_B|
```

For composed (full model):
```
revival_rate_full(S_post) = |D_anchor_full & A_{S_post}_full| / |D_anchor_full|
```

---

## 6. Kill Criterion

```
|revival_composed - revival_single| < 5 pp at S_post = 3200
```

If the difference is less than 5 percentage points, composition does not
meaningfully change revival dynamics. The "prune after training"
recommendation from Exp 18 applies equally to composed models.

If the difference exceeds 5 pp:
- Positive diff (amplification): composed models have MORE revival,
  meaning pruning timing is even MORE critical in composed setting
- Negative diff (suppression): composed models have LESS revival,
  meaning the dead set is MORE stable and pruning timing is LESS critical

---

## 7. Worked Numerical Example

At d=4, P=4, L=1, D=2 (4 capsules per domain, 8 total in composed):

### Single-domain (domain A):
```
Anchor (S=200): capsules {0,1,2,3}
  Dead: {0, 1}  (f_0=0, f_1=0, f_2=0.3, f_3=0.5)
  |D_anchor| = 2

After S_post=3200:
  Dead: {1}  (capsule 0 revived, capsule 1 still dead)
  revival_rate = |{0,1} & {0,2,3}| / |{0,1}| = |{0}| / 2 = 50%
```

### Composed model:
```
Composed capsules: {0,1,2,3} (domain A) + {4,5,6,7} (domain B)

Anchor (just-composed):
  Dead: {0,1,5,6}  (A-half dead: {0,1}, B-half dead: {5,6})
  |D_anchor| = 4

After S_post=3200 on joint data:
  Dead: {0,5,6}  (capsule 1 revived, capsule 5,6 still dead)
  A-half: revival = |{0,1} & {1,2,3}| / |{0,1}| = 1/2 = 50%
  B-half: revival = |{5,6} & {4,7}| / |{5,6}| = 0/2 = 0%
  Full: revival = |{0,1,5,6} & {1,2,3,4,7}| / |{0,1,5,6}| = 1/4 = 25%

Difference from single: 25% - 50% = -25 pp (composition suppresses)
```

This example illustrates the suppression case: cross-domain capsules
(B's capsules) see their "wrong" domain inputs but don't revive,
dragging down the composed revival rate.

---

## 8. Assumptions

1. **Capsule identity preserved in composition.** Capsule j in domain A's
   single-domain model has the same weights as capsule j in the composed
   model's A-half (by construction: concatenation preserves indices).

2. **Profiling on joint data is appropriate for composed models.** The
   composed model will be used on mixed-domain data in deployment; profiling
   on joint data reflects deployment conditions.

3. **Same training seed across conditions.** All conditions use the same
   seed for the base pretraining and fine-tuning trajectory. The only
   difference is whether post-anchor training happens in single-domain
   or composed context.

4. **Anchor step S=200 is representative.** This matches the standard
   fine-tuning protocol. Different anchor steps could yield different
   revival dynamics (Exp 18 showed revival rate depends on training phase).

5. **Binary dead/alive at f=0.** Same threshold as all prior experiments.

6. **Revival rate is the correct metric.** We measure the FRACTION of
   anchor-dead capsules that revive, not absolute counts. This controls
   for the different number of dead capsules at anchor across conditions.

---

## 9. Computational Cost

Per seed, per condition:
- Base pretraining: 300 steps (shared, done once)
- Per-domain fine-tuning: 200 steps x 2 domains (shared across conditions)
- Post-anchor training: sum(100 + 400 + 800 + 1600 + 3200) = 6100 steps
- Each step in composed model: ~2x FLOPs (2P capsules vs P)

Total per seed: ~300 + 400 + 6100*2 (single) + 6100*2 (composed) = ~25K steps
Total experiment: 3 seeds * ~25K = ~75K equivalent steps
Estimated wall time: ~5-8 minutes (micro scale, M-series Apple Silicon)
