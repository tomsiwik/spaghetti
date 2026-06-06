# Learnings: exp_bitnet_basefree_exploration

## Core Finding

Replacing pretrained BitNet-2B-4T base weights with random ternary scaffolds fails
catastrophically (27.6M x PPL ratio, threshold 5x). The pretrained base IS the
computation — it is not a replaceable scaffold. Adapters trained on the pretrained
base contribute exactly zero value when transferred to a random scaffold (319M vs
320M PPL), because they encode directions in the pretrained coordinate system.

## Why This Happened (Literature-Grounded)

Three converging mechanisms explain the 7-order-of-magnitude failure:

**1. Coordinate System Coupling (Cross-LoRA, arxiv 2508.05232; LoRA-X, OpenReview)**

LoRA adapters learn low-rank perturbations in the coordinate system defined by the
base model's weight matrices. Cross-LoRA (2025) and LoRA-X (2024) independently
showed that transferring LoRA adapters between different base models requires
explicit subspace alignment via rank-truncated SVD and Frobenius-optimal linear
transformation. Without alignment, the adapter's learned directions are meaningless
in a new coordinate system. Our result is the extreme case: a random ternary scaffold
has zero correlation with the pretrained coordinate system, so adapter contributions
are pure noise. The 319M vs 320M PPL (with/without adapters on scaffold) confirms
this — the adapters are literally invisible.

**2. Lottery Ticket Hypothesis (Frankle & Carlin, arxiv 1803.03635; Sparse ReBasin, arxiv 2505.05143)**

The Lottery Ticket Hypothesis established that winning subnetwork masks are tightly
coupled to their original initialization — using the same mask with a different
random initialization causes significant performance drops. Sparse ReBasin (2025)
showed this is because different initializations land in different loss basins, and
permutation symmetry must be resolved to transfer structure. Our scaffold replacement
is an even more extreme violation: we replace ALL weights (not just a sparse mask)
with random values, destroying all learned basin structure simultaneously.

**3. Cornerstone Layer Criticality (arxiv 2409.14381)**

"Investigating Layer Importance in Large Language Models" found that early layers
(termed "cornerstone layers") are disproportionately important — removing a single
cornerstone layer reduces performance to near-random. The top 3 layers account for
37% of total contribution (Shapley values). Our U-shaped criticality profile at
BitNet-2B-4T (layers 0-4 critical, 10-16 replaceable, 28-29 critical) is fully
consistent with this finding, now confirmed for ternary architectures. The
catastrophic scaffold failure is driven primarily by these cornerstone layers:
layer 0 alone causes 290,373% PPL increase when zeroed.

## Confirming Evidence

| Paper | Key Finding | Relation |
|-------|-------------|----------|
| Cornerstone Layers (arxiv 2409.14381) | Early layers dominate model performance; removing one → near-random | **CONFIRMS** our U-shaped criticality at d=2560 |
| ShortGPT (ACL 2025) | 25% of layers removable with 85% accuracy retained; Block Influence metric | **CONFIRMS** middle layer redundancy (our 9/30 replaceable) |
| Cross-LoRA (arxiv 2508.05232) | LoRA adapters require subspace alignment for cross-model transfer | **CONFIRMS** adapters are invisible on mismatched bases |
| Lottery Ticket Hypothesis (arxiv 1803.03635) | Random reinitialization with same mask fails | **CONFIRMS** weight-initialization coupling |
| Sparse ReBasin (arxiv 2505.05143) | Different inits → different basins; permutation alignment needed | **CONFIRMS** coordinate system matters fundamentally |
| Phase transitions in LLM compression (Nature, 2026) | Compression shows sharp phase transitions, not gradual degradation | **CONFIRMS** our super-exponential progressive ablation pattern |

## Contradicting Evidence

| Paper | Key Finding | Discrepancy Explanation |
|-------|-------------|----------------------|
| ReLoRA (ICLR 2024, arxiv 2307.05695) | Can train from random init via iterative LoRA + warm start | NOT contradicting — ReLoRA trains iteratively FROM scratch, not scaffold replacement. Requires 5K-step full-rank warm start. Our experiment tested replacement, not training. |
| GaLore (arxiv 2403.03507) + GaLore 2 (arxiv 2504.20437) | Full-parameter training from scratch via gradient low-rank projection; 7B on single 24GB GPU | NOT contradicting — GaLore grows a full model from random init through training, not replacement. Confirms random init CAN work if you TRAIN through it. |
| base_free_composition (d=64, this project) | Scaffold-only expert loss = 1.27x (marginal) | APPEARS contradicting but explained by scale: at d=64, r/d = 0.125 (adapters are 12.5% of model dimension) vs r/d = 0.006 at d=2560. Adapters are proportionally 20x larger at toy scale. |

**Critical distinction**: No paper contradicts the finding that scaffold REPLACEMENT
fails. The literature unanimously confirms that pretrained weights cannot be replaced
with random values. What ReLoRA and GaLore show is that you can TRAIN from random
initialization — a fundamentally different operation.

## Alternative Approaches (What We Could Try Instead)

### 1. GaLore-Grown Ternary Base (Most Promising — exp_bitnet_galore_scaffold)
GaLore 2 (arxiv 2504.20437) demonstrated pre-training Llama 7B from scratch with
500B tokens using gradient low-rank projection. Apply this to grow a ternary base
from scratch on Apple Silicon. The base would be OUR scaffold, not Microsoft's,
solving the IP dependency. GaLore reduces memory by 65.5%, making 2B training
feasible on M-series chips.

### 2. ReLoRA Iterative Training (exp_bitnet_scaffold_fresh_adapters)
ReLoRA (ICLR 2024) shows iterative low-rank updates can approximate full-rank
training. Train adapters ON a random scaffold iteratively, merging back periodically.
Requires a 5K-step full-rank warm start — key question is whether this warm start
can use ternary weights. This is the most direct test of "can adapters learn on
a scaffold?"

### 3. Hybrid Scaffold (Novel — Not Yet in Literature)
Keep the 7 critical layers (0-4, 28-29) from the pretrained base, replace the 9
replaceable layers (10-16, 24, 26) with random ternary. Progressive ablation at
K=5 showed 1.40x PPL — acceptable for many applications. This hybrid approach
has no direct precedent in literature but is well-motivated by cornerstone layer
theory.

### 4. Cross-Model Adapter Transfer (Cross-LoRA / LoRA-X)
If we train a new base (GaLore or otherwise), Cross-LoRA's subspace alignment
could potentially transfer existing adapters to the new base without retraining.
This would preserve the adapter library while swapping the scaffold.

### 5. LoTA-QAF Ternary Adapters on Ternary Base (arxiv 2505.18724)
Ternary adapters stay in the integer quantization grid, enabling lossless merge.
If combined with a GaLore-grown ternary base, the entire system (base + adapters)
would be ternary — maximally efficient for CPU serving.

## Implications for Next Experiments

1. **exp_bitnet_galore_scaffold (P1)** is now the highest-priority base-free path.
   GaLore 2's success at 7B scale strongly motivates trying it at 2B ternary scale.
   Key risk: GaLore was demonstrated on FP16/BF16 — ternary QAT integration is
   unproven.

2. **exp_bitnet_scaffold_fresh_adapters (P1)** should use ReLoRA-style iterative
   training, not single-shot LoRA. The warm start requirement is the key variable
   to test. If a ternary warm start works, this path is viable.

3. **exp_bitnet_effective_delta_cosine (P1)** is unaffected by this kill — it
   measures composition quality on the pretrained base, which remains the
   production path.

4. **Hybrid scaffold** (keeping cornerstone layers + replacing middle layers) is
   a viable fallback if full base-free fails. Should be a follow-up experiment
   if both GaLore and fresh-adapter paths stall.

5. **Scale extrapolation is unreliable** — the 4-million-fold gap between d=64
   and d=2560 means ALL toy-scale results should be validated at production scale
   before drawing conclusions. This is a meta-learning for all future experiments.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| Cross-LoRA: Data-Free LoRA Transfer | 2508.05232 | Adapter coordinate system coupling; subspace alignment for transfer |
| Sparse ReBasin (Weight Symmetry) | 2505.05143 | Why random reinitialization fails; permutation alignment |
| GaLore 2: Large-Scale Pre-Training | 2504.20437 | Training from scratch at 7B scale; direct input for galore_scaffold |
| ShortGPT (ACL 2025) | 2403.03853 | Layer redundancy measurement; Block Influence metric |
| Phase Transitions in LLM Compression | Nature 2026 | Sharp phase transitions in compression, not gradual |
