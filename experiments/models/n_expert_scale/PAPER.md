# Exp 4: Scaling Composition to N=5 Experts

## Hypothesis

The shared-base composition protocol scales from N=2 to N=5 domains:
capsule group subspaces remain approximately orthogonal, composition quality
stays within 5% of joint training, and router calibration scales linearly.

**Falsifiable**: If composition+calibrated vs joint exceeds +5%, the protocol
fails at N=5 and composition is limited to small N.

---

## What This Experiment Tests

Experiments 1-3 validated composition with N=2 domains (a-m vs n-z).
VISION.md asks: does this scale to N=5+? Three specific concerns:

1. **Subspace crowding**: Do 5 sets of capsule deltas interfere in d=64 space?
2. **Router complexity**: Can a softmax router handle G=20 groups effectively?
3. **Calibration scaling**: Does router calibration cost grow super-linearly?

---

## Setup

- **5 domains** by first letter: a-e (10.5K), f-j (5.0K), k-o (8.6K), p-t (5.6K), u-z (2.4K)
- **Architecture**: CapsuleMoEGPT, d=64, 4 layers, 4 heads
- **Per domain**: G=4 groups, 64 capsules/group, k=2 active
- **Composed**: G=20, k=10 (maintains 50% active fraction)
- **Protocol**: pretrain shared base (300 steps) → fine-tune capsule groups/domain
  (300 steps × 5) → compose → calibrate router (200 steps)
- **Joint baseline**: CapsuleMoEGPT(G=20, k=10), 1500 steps on all 5 domains
- **3 seeds** (42, 123, 7), ~732K params composed, ~203K per-domain

---

## Results (3-seed aggregate)

### Composition Quality

| Method | Avg Val Loss | vs Joint |
|--------|-------------|----------|
| Joint training (G=20, k=10) | 0.4951 | baseline |
| Composed + calibrated | 0.5032 | **+1.6%** |
| Composed + uniform | 0.6910 | +39.6% |
| Task arithmetic | 0.5228 | +5.6% |

### Per-Domain Breakdown (vs Joint)

| Domain | Size | Composed+cal vs Joint |
|--------|------|----------------------|
| a_e | 10,479 | -0.1% |
| f_j | 4,973 | +3.0% |
| k_o | 8,613 | -0.2% |
| p_t | 5,609 | +2.6% |
| u_z | 2,359 | +3.0% |

Smaller domains (f_j, u_z) show slightly more degradation — consistent
with less training data producing less-specialized capsule groups.

### Subspace Orthogonality

Pairwise cosine similarity of capsule weight deltas (Δ_i vs Δ_j):

| Metric | N=5 (this exp) | N=2 (prior) |
|--------|---------------|-------------|
| Mean | 0.112 | ~0.000 |
| Max | 0.167 | ~0.000 |
| Min | 0.054 | ~0.000 |

Orthogonality degrades from near-perfect at N=2 to modest correlation at N=5,
but remains well below the 0.5 concern threshold. The 5 domain subspaces are
approximately orthogonal in the d=64 parameter space.

Per-layer breakdown (mean pairwise cosine, seed-averaged):
- Layer 0: 0.119
- Layer 1: 0.119
- Layer 2: 0.114
- Layer 3: 0.096

Earlier layers show slightly more correlation than later layers.

### Router Analysis

| Metric | Value |
|--------|-------|
| H/H_max | 0.995-0.999 |
| C_1 (top group) | 0.055-0.063 |
| Theoretical uniform | 1/20 = 0.050 |

The router converges to near-uniform distribution across 20 groups.
This is consistent with prior findings: at micro scale with similar domains,
the router's value is in preventing bad routing (uniform without calibration:
+39.6%), not in achieving sharp domain specialization.

### Calibration Scaling

| N | Cal Steps | vs Joint |
|---|-----------|----------|
| 2 | 100 | -0.2% |
| 5 | 200 | +1.6% |

200 steps is sufficient for N=5. The +1.6% degradation vs N=2's -0.2%
suggests calibration quality degrades slightly with more groups, but remains
well within threshold. Linear scaling (2× steps for 2.5× domains) appears adequate.

---

## Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Composition+cal vs joint | +1.6% | <5% | **PASS** |
| Max pairwise cosine sim | 0.167 | <0.5 | **PASS** |
| Any single domain >10% | max +3.0% | <10% | **PASS** |
| Calibration scaling | 200 steps OK | <400 | **PASS** |

**All thresholds passed. The composition protocol scales to N=5.**

---

## Key Findings

1. **Composition scales sub-linearly**: +1.6% at N=5 vs -0.2% at N=2. Degradation
   exists but is modest. The protocol remains viable.

2. **Orthogonality degrades gracefully**: Mean cosine similarity goes from ~0.000
   (N=2) to 0.112 (N=5). At d=64 with character-level tokenization, 5 domains
   share some representation structure but remain largely independent.

3. **Calibration is essential and sufficient**: Without calibration, uniform routing
   at G=20 is catastrophic (+39.6%). 200 calibration steps restore near-joint quality.
   The calibration cost is low and appears to scale linearly with N.

4. **Task arithmetic still fails**: +5.6% at N=5 (vs +44.3% at N=2 with raw TA).
   The smaller per-domain delta (1/5 vs 1/2 scale) makes TA less catastrophic,
   but it still exceeds the 5% threshold.

5. **Router near-uniform at G=20**: Same finding as G=4/8 — the softmax router
   prevents bad routing but doesn't learn sharp specialization at micro scale.
   C_1 ≈ 0.056, essentially 1/G.

6. **Smaller domains degrade more**: u-z (2.4K names) and f-j (5.0K names) show
   +3.0% vs joint, while a-e (10.5K) and k-o (8.6K) are within 0.2%.
   Data quantity per domain affects composition quality.

---

## Honest Limitations

1. **Character-level domains are similar**: All 5 domains are character-level name
   generation. With truly distinct domains (code, prose, math), orthogonality
   and composition dynamics could differ substantially.

2. **Cosine similarity increase at N=5 is notable**: Going from 0.000 to 0.112
   suggests that at higher N (10, 20), subspace interference could become
   problematic. The d=64 space may saturate.

3. **Router provides no specialization**: H/H_max ≈ 0.999 means the router treats
   all 20 groups identically. The calibration's value is in adjusting the router
   to the composed topology, not in learning domain routing.

4. **Joint baseline has more total training steps**: Joint sees 1500 steps on all
   data; composition protocol uses 300 pretrain + 5×300 fine-tune + 200 calibrate
   = 2000 total steps. More total compute for composition.

5. **k=10 of G=20 is generous sparsity**: Activating 50% of groups provides limited
   computational savings. Real scaling would need lower k/G ratios, which may
   require larger expert capacity (deferred to macro).

---

## Implications for VISION.md

The composition protocol — pretrain base, fine-tune capsule groups per domain,
compose by concatenation, calibrate router — **survives scaling to N=5**.

The natural next question is **Exp 5: Beat 1.5B** — can a 0.5B base + N LoRA
experts match or exceed a 1.5B monolithic model? This requires moving from
micro scale (d=64, character names) to macro scale (real LLMs, real domains).

At micro scale, we have now validated:
- The composition mechanism works (Exp capsule_moe)
- Routing by task quality is the right approach (Exps 1, 1b)
- k=2 is minimum viable sparsity (Exp 2)
- Weight-space decomposition doesn't help nonlinear groups (Exp 3)
- **The protocol scales to N=5 with +1.6% degradation (this Exp 4)**

The micro arena has been thoroughly explored. Further micro experiments would
yield diminishing returns — the limitations are now scale-bound, not
mechanism-bound.

---

## Artifacts

- `micro/models/n_expert_scale/` — code, tests, MATH.md, PAPER.md
- Parent model: `capsule_moe`
- Data: `domain_split(docs, method="quintary")` added to `micro/data.py`
