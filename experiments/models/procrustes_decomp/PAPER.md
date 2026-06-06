# Shared/Unique Decomposition of Domain Capsule Groups: Research Digest

## Hypothesis

When domain-specific capsule groups are fine-tuned from a shared base, their
deltas can be decomposed into shared (always-on) + unique (routed) components.
The shared component absorbs routing errors, making composition more robust
than naive concatenation.

**Falsifiable**: If decomposed composition degrades >5% vs joint training, or
performs worse than concatenation, the approach adds complexity without benefit.

---

## What This Experiment Tests

Given the validated shared-base composition protocol from capsule_moe:
1. Pretrain base → fine-tune capsule groups per domain → compose

We decompose the fine-tuning deltas:
```
shared_delta = (delta_A + delta_B) / 2       -- what both domains learned
unique_delta_A = (delta_A - delta_B) / 2     -- what only domain A learned
unique_delta_B = (delta_B - delta_A) / 2     -- what only domain B learned
```

This decomposition is **exact in weight space**: base + shared + unique_k = base + delta_k.

The decomposed model has:
- 4 shared groups (always active, uniform weight): weights = base + shared_delta
- 8 unique groups (routed, 4 per domain): weights = unique_delta

---

## Results (3-seed aggregate)

### Decomposition Analysis

| Metric | Value |
|--------|-------|
| Shared fraction of delta norm | 53.9% (range: 53.6%-54.1%) |
| Max reconstruction error | <6e-08 (numerically exact) |

**54% of fine-tuning knowledge is shared between domains.** The decomposition
is non-trivial — both shared and unique components carry substantial weight.

### Composition Quality

| Method | Avg Val Loss | vs Joint | Groups | Routed? |
|--------|-------------|----------|--------|---------|
| Joint training | 0.5225 | baseline | 4 | k=2 |
| **Concat + calibrated** | **0.5213** | **-0.2%** | **8** | **k=4** |
| Concat + uniform | 0.6248 | +19.6% | 8 | none |
| Task arithmetic | 0.7540 | +44.3% | 4 | k=2 |
| Shared only | 0.6781 | +29.8% | 4 | k=2 |
| Decomp + calibrated | 0.5525 | +5.7% | 4+8=12 | k=4 unique |
| Decomp + uniform | 0.6206 | +18.8% | 4+8=12 | none unique |

### Kill Threshold Checks

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Decomp+cal vs joint | +5.7% | <5% | **KILL** |
| Decomp+cal vs concat+cal | +5.9% worse | Must not be worse | **KILL** |
| Shared fraction | 53.9% | >5% | PASS |
| Decomp+uniform vs concat+uniform | 0.7% better | Must be better | PASS (marginal) |

---

## Root Cause: Nonlinearity Breaks Weight-Space Decomposition

The decomposition is exact in weight space but **not in function space**.

Each capsule group computes: `output = B @ ReLU(A @ x)`

In the original model (single group per domain):
```
f(x) = (B_base + ΔB_k) @ ReLU((A_base + ΔA_k) @ x)
```

In the decomposed model (shared + unique as separate groups):
```
f_decomp(x) = (B_base + shared_ΔB) @ ReLU((A_base + shared_ΔA) @ x)
            + unique_ΔB_k @ ReLU(unique_ΔA_k @ x)
```

These are NOT equal because ReLU is nonlinear: `f(a+b) ≠ f(a) + f(b)`.

The unique groups operate on raw input x with tiny weights (unique_delta),
producing small and noisy activations. The ReLU kills many of them (negative
activations zeroed), losing information that would have survived if combined
with the larger base+shared weights before the nonlinearity.

**This is a fundamental limitation**: linear weight-space decomposition cannot
be cleanly mapped to a nonlinear architecture without approximation error.

---

## What We Learned

### 1. Shared Knowledge is Substantial (54%)

Domain-specific fine-tuning at micro scale produces ~54% shared knowledge.
This makes intuitive sense: a-m and n-z names share the same character
distributions and structural patterns. The fine-tuning deltas capture general
"name modeling" improvements (shared) plus small domain-specific adjustments
(unique).

### 2. Task Arithmetic Causes Massive Dilution (+44%)

Task arithmetic (base + 0.5 × (Δ_A + Δ_B)) degrades by +44% vs joint. This
confirms VISION.md's claim: naive merging at λ=0.5 halves each expert's
effective contribution. The dilution is catastrophic.

### 3. Shared Knowledge Alone is Insufficient (+30%)

The shared-only model (base + shared_delta, no unique) degrades by +30%. This
means the 46% unique knowledge is essential for domain-specific quality.

### 4. Nonlinear Decomposition Doesn't Beat Concatenation

The decomposed model (+5.7%) is worse than concatenation (-0.2%) despite using
50% more capsule parameters (12 groups vs 8). The nonlinearity penalty exceeds
any routing robustness benefit.

### 5. Marginal Robustness Under Uniform Routing

Under uniform routing (no learned router), the decomposed model (0.6206) is
0.7% better than concatenation (0.6248). This provides weak evidence for the
robustness hypothesis: shared groups DO absorb some routing error. But the
effect is too small to justify the complexity and parameter overhead.

---

## Implications for the Vision

### What Dies

- **Weight-space decomposition for nonlinear models.** The shared/unique split
  is exact in weights but approximate in function space. For ReLU-based
  capsules, the approximation error (5.7% degradation) exceeds the composition
  threshold. This approach cannot replace concatenation.

### What Survives

- **The concatenation protocol remains validated.** Shared-base concatenation
  with calibrated routing achieves -0.2% vs joint — essentially free
  composition. No decomposition needed.

- **The shared knowledge finding is informative.** 54% shared fraction suggests
  that at macro scale with more diverse domains (Python vs JavaScript), the
  shared fraction might be lower but the unique components more discriminative.
  This could make decomposition more valuable — if a linear architecture is
  used where decomposition IS exact (e.g., LoRA adapters without nonlinearity).

### What's Next

The decomposition idea has merit for **linear** expert components (e.g., LoRA
adapters: ΔW = A @ B, no nonlinearity in the delta). For the nonlinear capsule
groups, concatenation remains optimal.

**Exp 4: Scale to N experts** (5+ languages, verify subspaces stay orthogonal)
is the natural next step from VISION.md "What Remains."

---

## Artifacts

- `micro/models/procrustes_decomp/` — model, tests, MATH.md, PAPER.md
- Parent model: `capsule_moe`
- New params: 12 groups × 8192 params = 98304 capsule params/layer
  (vs 8 × 8192 = 65536 for concatenation)

---

## Micro-Scale Limitations

1. **Very similar domains.** a-m vs n-z names share character distributions.
   With truly distinct domains, shared fraction might differ substantially.

2. **Small deltas.** 300 steps of fine-tuning produces small deltas relative
   to base weights. The nonlinearity impact might differ with larger deltas.

3. **d=64 limits expressiveness.** The unique groups have very small weights
   at d=64. At larger dimensions, they might carry more useful signal.

4. **Only 2 domains.** With N>2 domains, shared knowledge across all N might
   be a smaller fraction, and decomposition might help more.
