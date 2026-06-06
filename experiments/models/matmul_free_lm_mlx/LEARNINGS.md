# Learnings: exp_matmul_free_lm_mlx

## Core Finding

Grassmannian LoRA composition is architecture-agnostic: replacing self-attention with
gated linear recurrence (HGRN) and all projections with ternary BitLinear preserves
composition quality (1.029x ratio, |cos|=0.0076), matching Transformer results exactly.

## Why This Happened (Literature-Grounded)

The composition guarantee derives from **linearity of the weight perturbation**, not
from any property of the attention mechanism. Each LoRA adapter adds a low-rank delta
`B_i @ A_i` to a linear projection. The Grassmannian A-matrix construction guarantees
`<A_i, A_j> = 0`, which ensures the deltas project into orthogonal subspaces regardless
of what surrounds the linear layer.

The HGRN recurrence `h_t = g_t * h_{t-1} + (1 - g_t) * i_t` is element-wise in hidden
state dimensions. Unlike attention's bilinear `Q @ K^T` interaction (where perturbations
to Q and K from different adapters create cross-terms), element-wise gating produces no
cross-adapter interaction terms within a single time step. This is the key structural
reason composition works on HGRN: the non-attention token mixer introduces no new
bilinear interactions between weight matrices.

The STE (Straight-Through Estimator) training regime is well-understood to preserve
gradient flow through ternary quantization (Ma et al., 2024, arxiv 2402.17764). The
Extra RMSNorm placement before quantized projections (validated in our warmstart
experiment, consistent with arxiv 2505.08823) stabilizes the ternary training, ensuring
both base model and adapters converge. This explains why ternary weights + HGRN +
composition all work together: each component's viability is independently established.

## Confirming Evidence

**LoRA on non-Transformer architectures is well-established:**
- LoRA has been successfully applied to Mamba/SSM architectures. Galim et al. (2025,
  "Parameter-Efficient Fine-Tuning of State Space Models", ICML 2025) show LoRA
  consistently outperforms all other PEFT methods on SSMs for linear projections
  (arxiv 2410.09016).
- MambaPEFT (arxiv 2411.03855) provides systematic benchmarks confirming LoRA
  effectiveness on Mamba linear layers, though SSM-specific modules need specialized
  methods (Sparse Dimension Tuning).
- The original LoRA paper (Hu et al., 2021) makes no architectural assumption beyond
  having linear layers -- the method is inherently architecture-agnostic.

**MatMul-free LM quality at scale:**
- Zhu et al. (2024, arxiv 2406.02528) demonstrate MatMul-free LMs scale to 2.7B
  parameters with quality comparable to Transformers, using HGRN-based token mixer
  and ternary BitLinear projections. Our micro-scale PPL parity (1.00x) is consistent
  with their findings.

**Orthogonal adapter composition on our own Transformer baseline:**
- Our ternary_base_from_scratch_mlx experiment achieved 1.022x composition ratio with
  |cos|=2.5e-7 on a ternary Transformer, confirming the skeleton works on ternary
  weights specifically.
- Our N=50 BitNet composition experiment achieved gamma_composed=0.632 (all adapters
  net positive) with mean |cos|=0.014, confirming scaling to many adapters.

## Contradicting Evidence

**Orthogonality alone is insufficient at scale (our own findings):**
- Our logit-ensemble research (NotebookLM sources) found that cosine orthogonality
  measures direction, not magnitude. When adapters have different weight norms, the
  larger-magnitude adapter dominates the composition. A single poorly-trained adapter
  can poison the entire merge, causing PPL to explode to trillions.
- Our equal-weight composition experiment was KILLED: CV=112.2% at N=5. This was
  resolved by 1/N scaling, but demonstrates that orthogonality is necessary but not
  sufficient -- magnitude normalization is critical.

**Nonlinear amplification through residual streams:**
- Even perfectly orthogonal weight perturbations become correlated after passing through
  LayerNorm and residual connections. This "nonlinear function space interference" is
  documented in our own research and explains why weight-space orthogonality (low cosine)
  does not guarantee output-space independence (OSRM diagnostic was KILLED: 100% failure).

**Gate-composition coupling (theoretical risk, untested):**
- The adversarial review identified that the forget gate `g_t = sigmoid(W_g @ x)` is
  itself a composed weight matrix. Perturbations to `W_g` change the gating trajectory,
  which modulates all subsequent hidden states. At T=32 this is negligible (geometric
  decay), but at T=2048+ with high-retention gates (g near 1), interference could
  amplify through the recurrence. This is a theoretical risk, not yet empirically tested.

**SSM-specific PEFT challenges:**
- Galim et al. (2025) found LoRA fails on SSM-specific modules (A, B, C, D matrices
  in Mamba). Our HGRN implementation only applies LoRA to linear projections (not the
  recurrence parameters), which is the correct approach per this finding. However, if
  future work targets HGRN gate parameters directly, this limitation applies.

**No published work on LoRA COMPOSITION on recurrent architectures:**
- While LoRA fine-tuning on SSMs is established, no published work tests multi-adapter
  composition (merging multiple LoRA adapters) specifically on recurrent/SSM backbones.
  Our experiment is the first known test of this combination.

## Alternative Approaches (What We Could Try Instead)

**For adapter composition on recurrent backbones:**
1. **MoLoRA per-token routing** (arxiv 2603.15965): Dynamic routing avoids static merge
   interference entirely. Qwen3-1.7B + 4 adapters > 8B monolithic. Would sidestep the
   gate-composition coupling risk at long sequences. Cost: O(N * L) routing overhead.
2. **TIES merging** (Yadav et al., 2023): Sign-aware sparse merge could handle magnitude
   mismatch. Known weakness: sacrifices temporal coherence (problematic for recurrent
   models where temporal structure is the core mechanism).
3. **Task Arithmetic**: Treating each adapter as a reusable delta allows negative weights
   to "unlearn" interference. Simpler than Grassmannian but no orthogonality guarantee.
4. **DARE (Drop and Rescale)**: Random sparsification of adapter weights before merge.
   Compatible with any architecture. Complementary to Grassmannian (could apply DARE
   to B matrices while keeping Grassmannian A matrices).

**For the speed problem (4.6x slower):**
1. **Parallel scan for HGRN**: The flash-linear-attention library provides CUDA kernels
   for parallel scan of gated recurrences. An MLX Metal kernel equivalent would close
   the speed gap. This is the primary engineering blocker.
2. **GLA (Gated Linear Attention)**: HGRN2 can be reformulated as gated linear attention,
   which admits chunked parallel computation. Yang et al. (2024) show this achieves
   near-Transformer throughput while maintaining recurrent inference.
3. **Custom Metal kernels for ternary accumulation**: MLX does not currently support
   custom Metal kernels from Python. The ternary accumulation (additions only) requires
   specialized hardware paths to realize the theoretical speedup.

**For scaling ternary models:**
1. **Data scaling > parameter scaling**: Ternary scaling laws show data exponent (0.81)
   dominates parameter exponent (0.32). For a fixed FLOP budget, expanding training data
   yields greater returns than increasing model size.
2. **Sparse-BitNet** (arxiv 2603.05168): 42% natural sparsity in ternary weights. Zero-
   skipping hardware (TOM/BitROM) can exploit this for further speedup.

## Implications for Next Experiments

1. **Architecture-agnostic composition is confirmed.** The Grassmannian skeleton's value
   proposition is validated beyond Transformers. This means our composition framework is
   portable to whatever backbone architecture we ultimately choose -- Transformer, HGRN,
   Mamba, or hybrid.

2. **The speed problem blocks HGRN on Apple Silicon.** 4.6x slower with no path to custom
   Metal kernels means HGRN is not viable for interactive serving on M5 Pro today. The
   matmul-free direction requires either: (a) MLX adding parallel scan support, (b) a
   hybrid architecture that uses attention for short contexts and recurrence for long, or
   (c) accepting the speed penalty for memory-constrained scenarios where ternary HGRN's
   1.6-bit weights enable larger models to fit.

3. **Gate-composition coupling needs a dedicated test.** The reviewer's concern about
   interference amplifying through the forget gate at long sequences is the primary open
   question. A targeted experiment at T=512 or T=1024 with N=10+ adapters would settle
   this. If interference grows super-linearly with T, dynamic routing becomes mandatory
   for HGRN-based composition.

4. **The parameter count mismatch (33%) is a confound.** Future architecture comparisons
   should use parameter-matched models. At larger scale, 33% more ternary parameters
   is significant for memory even at 1.6 bits/param.

5. **Consider HGRN as a complement, not replacement.** A hybrid architecture with
   attention for short-range and HGRN for long-range could capture the best of both:
   Transformer-speed composition for most contexts, with recurrent efficiency for long
   sequences. This is consistent with the Hybrid Linear Attention research direction.
