# Research Synthesis: 5 Parallel Deep-Dive Topics
## Date: 2026-04-08

## Executive Summary

5 research agents analyzed 1000+ sources via NotebookLM. The findings converge on a clear diagnosis and a prioritized experiment plan.

**Root cause of M2P 4B failure:** Not one problem but three interacting issues:
1. **Architecture bug:** Pre-layernorm in M2P transformer (SHINE explicitly warns against this)
2. **Information bottleneck:** Single d=1024 shared bottleneck cannot preserve 36 layers of d=2560 structure (Information Ceiling Theorem)
3. **Training insufficiency:** 2K examples with no pretraining vs SHINE's 6B-token pretraining

**Most promising fix (combinable):**
- Fix 1 (hours): Switch to post-layernorm + remove per-layer linear heads
- Fix 2 (hours): Add diversity loss (Coulomb repulsion + VICReg) to prevent centroid collapse
- Fix 3 (days): Add M2P pretraining phase before task fine-tuning
- Fix 4 (hours): LII-gated LOD to reduce M2P output space by ~50%

---

## Topic-by-Topic Key Findings

### Topic 1: Hypernetwork Scaling (BIGGEST FINDING)

**SHINE already scaled to Qwen3-8B.** Our 4B failure is likely caused by specific architectural deviations from SHINE's working implementation:

| Our Implementation | SHINE (works at 8B) | Impact |
|-------------------|---------------------|--------|
| Pre-layernorm | **Post-layernorm** | SHINE: "pre-norm causes trouble at scale" |
| Per-layer linear heads (1024→16384) | **Direct reshape** | Eliminates bottleneck |
| No pretraining (2K examples) | **6B-token pretraining** + IFT | Learns general context→params mapping |
| Rank-4 A-matrices | **Meta-LoRA rank 128** | More expressivity in memory extraction |

**Priority:** Fix post-layernorm + direct reshape FIRST (hours, not days).

Papers: SHINE (arXiv:2602.06358), Doc-to-LoRA (2602.15902), Profile-to-PEFT (2510.16282), HoRA (2510.04295)

### Topic 2: Information Bottleneck & Jacobian

**Theoretical confirmation** that the single bottleneck is the mathematical cause:
- Information Ceiling Theorem (MASA, arXiv:2510.06005): a bottleneck of width r preserves at most r orthogonal channels
- At 4B: needs 737,280 values, capacity is 589,824 — 20% deficit
- Gradient norm healthy (38.06) but Jacobian condition number likely >>1000
- Gradient norm dominated by σ_max; output diversity by σ_min — completely disconnected

**Key approaches:**
1. Multi-bottleneck architecture (4 independent d=256 bottlenecks instead of 1×d=1024)
2. Jacobian clamping + bi-Lipschitz regularization (forces distinct inputs → distinct outputs)
3. Muon optimizer + hard-tanh activation (preserves dynamical isometry)

Papers: MASA (2510.06005), DNP (NeurIPS 2026 #505), Jacobian Clamping (PMLR v108), Muon (2512.14366), Dynamical Isometry (NIPS 2017)

### Topic 3: Game Engine × LLM

**Confirmed:** Everything works per-layer, nothing works cross-layer. Room Model kill was correct.

**3 novel per-layer approaches:**
1. **Deferred Inference (Neural G-Buffer):** Cache base matmul, apply lightweight adapter composition per-query
2. **Tangent Space Task Arithmetic:** Linearize around base model per-layer. Superposition exact in tangent space
3. **Continuous LOD (CLoD):** Learnable input-dependent decay per adapter module

Papers: Model Merging Survey (2603.09938), Dynamic Octree (2504.18003), Neural Harmonic Textures (2604.01204), CLoD-GS (2510.09997)

### Topic 4: Physics Simulation for Centroid Fix

**3 mechanisms to prevent B-matrix centroid collapse (cos=0.9956):**

1. **Coulomb Repulsion + VICReg Variance Hinge (RECOMMENDED):**
   - `L_repulsive = log Σ exp(-t||B_i - B_j||²)` — force proportional to proximity
   - `L_var = max(0, γ - std(B))` — hard floor makes collapse mathematically impossible
   - Provides auxiliary gradient channel bypassing the low-rank Jacobian

2. **DPP Log-Determinant:** `L = -log det(K)` — provable diversity via volume measure
3. **Lagrangian Hard Constraints:** `||B_i - B_j||² ≥ δ` enforced exactly via Augmented Lagrangian

Papers: Wang & Isola alignment+uniformity (2005.10242), VICReg (2105.04906), MO-PaDGAN (2007.04790), Lagrangian DL (2001.09394), Rep-MTL (2507.21049)

### Topic 5: LOD for M2P Capacity

**3 adaptive capacity allocation approaches:**

1. **LII-Gated LOD with Three Tiers (RECOMMENDED):**
   - Layer Instability Index (forward-pass only, no gradients)
   - Classify layers: Full-rank / Low-rank / Skip BEFORE M2P generation
   - Expected to cut M2P output by ~50%

2. **Fisher-Guided Water-Filling:** 3 samples + 1 backward pass → optimal rank per layer
3. **Progressive Coarse-to-Fine:** Rank-1 for all layers first, then iteratively refine important layers

Papers: MoD (2404.02258), AdaLoRA (2303.10512), LoRA² (2512.04555), COLA (2510.10150), Surgical Fine-Tuning (2502.11466)

---

## Cross-Topic Synergies

### Synergy 1: SHINE Architecture Fix + Multi-Bottleneck (Topics 1+2)
Post-layernorm from Topic 1 fixes the training instability. Multi-bottleneck from Topic 2 fixes the information ceiling. These are orthogonal — combine both.

### Synergy 2: Diversity Loss + LOD (Topics 4+5)
LII-gated LOD reduces the output space (Topic 5). Coulomb repulsion ensures the remaining outputs are diverse (Topic 4). Together: fewer but better B-matrices.

### Synergy 3: Tangent Space Arithmetic + Deferred Inference (Topics 3)
Tangent space composition works per-layer (Topic 3). Deferred inference caches the base computation (Topic 3). Together: exact composition + efficient serving.

---

## Recommended Experiment Sequence

### Phase A: Quick Wins (1-2 days)

1. **Diagnostic: Measure JER and condition number** of current M2P v5 at 4B
   - Runtime: <1 hour
   - Confirms/refutes low-rank Jacobian hypothesis
   - If JER << d_output: confirms bottleneck is the issue

2. **Fix M2P architecture (post-layernorm + direct reshape)**
   - Change M2PBlock to post-norm
   - Remove per-layer b_heads, use direct reshape
   - Rerun v5 at 4B with same 300 steps
   - Predicted: quality_ratio goes from -0.187 to >0 (positive territory)

3. **Add Coulomb + VICReg diversity loss** to training
   - Add L_repulsive and L_var_hinge to M2P training loss
   - Measure B-matrix cosine similarity during training
   - Predicted: cos drops from 0.9956 to <0.70

### Phase B: Structural Changes (3-5 days)

4. **LII-gated LOD** — profile Qwen3-4B layers, skip B-generation for low-LII layers
5. **Multi-bottleneck architecture** — 4×d=256 instead of 1×d=1024
6. **M2P pretraining phase** — reconstruction + completion on generic text

### Phase C: If Phase B Works

7. **Combine all fixes** — post-norm + direct reshape + diversity loss + LOD + pretraining
8. **Full GSM8K evaluation** at n=500 with statistical closure
9. **MATH.md** with theorem/proof for the combined architecture

---

## Papers to Investigate Further

| Priority | Paper | Why |
|----------|-------|-----|
| P0 | SHINE (2602.06358) | Already works at 8B. Our architecture deviations are likely the fix |
| P0 | MASA (2510.06005) | Information Ceiling Theorem proves bottleneck is the issue |
| P0 | VICReg (2105.04906) | Variance hinge makes centroid collapse impossible |
| P1 | Wang & Isola (2005.10242) | Uniformity loss = Coulomb repulsion on hypersphere |
| P1 | MoD (2404.02258) | Per-token layer skipping for LOD |
| P1 | ELA-ViT (LII metric) | Forward-pass-only layer importance |
| P1 | Muon optimizer (2512.14366) | Orthogonalizing optimizer for rank preservation |
| P2 | Neural Harmonic Textures (2604.01204) | Periodic activations for expert blending |
| P2 | DPP-GAN (1812.00802) | Determinantal Point Process for diversity |
| P2 | Lagrangian DL (2001.09394) | Hard constraints on B-matrix separation |
