# EigenLoRAx Subspace Extraction on Orthogonal Ternary Adapters: Research Digest

## Hypothesis

SVD of stacked adapter weight matrices from N=25 trained ternary LoRA adapters
reveals a shared principal subspace that can accelerate new adapter training
via coefficient-only learning (arXiv 2502.04700, EigenLoRAx ICML).

**Falsifiable claim**: A K=16 principal subspace captures >50% of adapter
variance, and a new adapter trained on subspace coefficients achieves PPL
within 20% of from-scratch LoRA training.

## What This Experiment Is

We applied the EigenLoRAx algorithm to our 25 trained ternary LoRA adapters
(15 domains + 10 capabilities) on BitNet-2B-4T. The algorithm stacks adapter
weight matrices, performs SVD to find principal components (shared directions),
then trains new adapters using only coefficients on those PCs instead of full
weight matrices. The paper reports 100x parameter reduction and 2x faster
convergence.

The critical tension: our adapters use Grassmannian-packed frozen A matrices
that guarantee near-zero cosine similarity (|cos| ~ 0.001). EigenLoRAx
assumes adapters share structure. This experiment tests whether the
orthogonality guarantee that makes composition work also prevents subspace
extraction.

## Key References

- **EigenLoRAx** (Ramezani et al., 2025, arXiv 2502.04700): SVD of stacked
  adapters reveals shared principal subspace. 100x fewer params, 2x faster.
- **LoRA** (Hu et al., 2021): Low-rank adaptation.
- **Prior experiment** (exp_bitnet_spectral_surgery): KILLED. Short-trained
  adapters already have efficient spectra -- no room for spectral refinement.

## Empirical Results

### VERDICT: KILLED (K1=PASS technically, K2=FAIL, K3=PASS)

### Kill Criteria Assessment

| Criterion | Metric | Threshold | Observed | Verdict |
|-----------|--------|-----------|----------|---------|
| K1 | Variance at K=16 | >= 50% | 65.6% avg | PASS (see caveat) |
| K2 | PPL gap vs scratch | <= 20% | +80.8% | **FAIL** |
| K3 | Extraction time | <= 10 min | 24.3s | PASS |

### K1: Variance Explained (Technically PASS, Practically Misleading)

| Matrix Type | Avg Variance at K=8 | Avg Variance at K=16 | Avg K for 50% |
|-------------|---------------------|----------------------|---------------|
| LoRA-A | 24.6% | 31.3% | 63.1 |
| LoRA-B | 70.3% | **100.0%** | 5.4 |
| Overall | 47.5% | **65.6%** | 34.3 |

The 65.6% average is driven entirely by LoRA-B's trivial 100% (B has rank 16,
N=25 > 16, so K=16 PCs span all of B-space). LoRA-A captures only 31.3% --
the Grassmannian skeleton spreads adapters across ALL dimensions, leaving no
dominant subspace.

The K1 metric technically passes at 65.6% but this average is misleading.
The EFFECTIVE subspace (A-space, where domain-specific information lives) has
only 31.3% variance captured -- well below 50%.

### K2: Subspace Training vs From-Scratch (FAIL)

| Method | Trainable Params | PPL (wikitext) | Training Time |
|--------|-----------------|----------------|---------------|
| Base (no adapter) | 0 | 25.44 | - |
| From-scratch LoRA | 21,626,880 | 13.52 | 137.9s |
| Subspace (K=16) | 6,720 | 24.44 | 158.4s |

- PPL gap: +80.8% (threshold: 20%) -- **KILLED**
- Compression: 3,218x fewer parameters
- Speed: No speedup (0.87x, actually slower due to matmul overhead)
- Holdout reconstruction error: 1.008 (essentially zero reconstruction)

The subspace adapter achieved almost no improvement over base (25.44 -> 24.44),
while from-scratch LoRA achieved 13.52. The 6,720 subspace coefficients
cannot capture the domain-specific signal that lives in the orthogonal A-space.

### K3: Extraction Speed (PASS)

SVD extraction for 420 module keys across 25 adapters: 24.3 seconds.
Well within the 10-minute threshold.

## Why It Failed: The Orthogonality-Subspace Tradeoff

This result reveals a fundamental tradeoff in our architecture:

**The Grassmannian skeleton that enables interference-free composition
also prevents shared subspace extraction.**

EigenLoRAx works when adapters share structure -- when multiple tasks learn
similar weight directions, SVD concentrates variance in a few PCs. Standard
LoRA (random A initialization) produces correlated adapters by construction:
all A matrices start from the same random seed family, and training pushes
them toward similar loss landscapes.

Our design deliberately breaks this correlation:
1. A matrices are Grassmannian-packed (maximally spread on Gr(r, d))
2. A matrices are frozen during training
3. Only B matrices are trained, but B is rank-16 (trivially captured)

The result: adapters are orthogonal (composition works), but there is no
shared "principal subspace" to extract (EigenLoRAx fails).

### Why B-Space Recovery Is Insufficient

B matrices at 100% variance recovery seem promising, but B alone cannot
reconstruct a useful adapter because:
- B captures output-space directions (what to add to each hidden dim)
- A captures input-space selection (which hidden dims to attend to)
- Without A-space recovery, the subspace adapter randomly selects input
  dimensions, destroying domain specificity

## Implications for the Evolve Track

1. **EigenLoRAx is incompatible with Grassmannian composition**. The same
   property that makes composition stable (orthogonal A) prevents subspace
   acceleration. This is a fundamental architectural constraint, not a
   parameter tuning issue.

2. **Evolve must use independent training**, not subspace transfer. Each new
   adapter starts from scratch with a fresh Grassmannian A matrix. This was
   already the working model (retrain-from-scratch + quality gate), and this
   experiment confirms it is the only viable path.

3. **B-matrix subspace could still be useful** for initialization: freeze a
   new Grassmannian A, then initialize B from the B-subspace mean. This is a
   weaker version of EigenLoRAx that might provide modest speedup without
   breaking orthogonality. Worth testing separately.

## Limitations

1. **Single holdout domain**: Only wikitext tested. Other domains might show
   different reconstruction quality, though the orthogonality mechanism
   affects all domains equally.

2. **Short training**: 400 steps. Longer training might reveal more shared
   structure in B-space, but A-space orthogonality is a design constraint,
   not a training duration issue.

3. **K=16 only**: We did not sweep K. However, A-space variance at K=24
   (maximum) would still only reach ~100% of 24/25 dimensions, and the
   holdout would still be orthogonal to all of them.

4. **No augmented subspace**: The paper suggests adding random orthogonal
   "pseudo-PCs" for low-resource domains. This could help but fundamentally
   the issue is that A-space needs d_in * r dimensions, not K << d_in * r.

## What Would Kill This (Already Killed)

- K1: Principal subspace <50% variance -- **PARTIALLY CONFIRMED** (A-space 31.3%)
- K2: Subspace adapter >20% worse PPL -- **CONFIRMED (+80.8%)**
- K3: Extraction >10 min -- NOT killed (24.3s)

## Connection to Spectral Surgery Kill

This result is consistent with the spectral surgery kill (exp_bitnet_spectral_surgery):
- Spectral surgery found adapters already have efficient spectra (nothing to remove)
- EigenLoRAx found adapters share no subspace (nothing to extract)
- Both stem from the same root cause: Grassmannian A matrices + short training
  produce clean, orthogonal adapters with no redundancy between them

The architecture is doing exactly what it was designed to do -- making adapters
independent. The cost of independence is that cross-adapter learning transfer
is impossible by construction.
