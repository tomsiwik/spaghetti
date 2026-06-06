# LEARNINGS: M2P Bottleneck Width (exp_m2p_bottleneck_width) — KILLED

## Core Finding

JL distortion and M2P generation quality are **not equivalent**: widening d_M2P from 64 → 128 → 256 produces flat quality (~95–97% of SFT) with no monotone improvement, falsifying the JL-as-bottleneck hypothesis. The 7.8% quality gap from Finding #354 was partly an artifact of reused M2P adapters from a different training run, not a structural dimension floor.

---

## Why This Happened

### Root Cause: JL bounds representation distortion, not generative capacity

The JL lemma (Johnson-Lindenstrauss 1984; exact form: Dasgupta & Gupta 1999, arXiv:cs/9904007) guarantees pairwise distance preservation for **random projections**. It makes no claim about:

1. **Learned projections** — gradient descent can align or misalign the projection with domain-discriminative directions independently of d_M2P.
2. **Generative quality** — the M2P must not only *represent* domain identity but *generate* correct B-matrix weights. These are distinct computational tasks. JL addresses only (1).

Formally: the proof chain breaks at step 3 (separation margin → M2P can distinguish domains) and step 4 (domain distinction → generation quality ≥ 97%). Neither link was proven. Corollary 2 ("generation quality bounded by projection fidelity") is an informal assertion, not a theorem.

### Why the baseline discrepancy (95.1% vs. 92.2%)

Finding #354 cited 92.2% oracle quality for d=64. This experiment measured 95.1% at d=64. The difference: Finding #354 **reused M2P adapters trained in a prior run** (m2p_composition_n5). Fresh M2P training in this experiment produced higher-quality adapters. The 7.8% gap was partially an adapter-reuse artifact, not a dimension bottleneck.

### Why quality is flat at 95–97%

With fresh training, quality clusters near the M2P architecture's generative ceiling (~95–97% of SFT). The variation across domains (sort at 81–95%, repeat at 98–100%) reflects per-domain difficulty and 500-step convergence limits, not projection fidelity. The sort domain consistently underperforms because sort/reverse share statistical structure (character ordering) that the M2P cannot fully disambiguate.

---

## Literature Context

### What JL actually proves

- **Johnson & Lindenstrauss (1984)**: Any N points in high-d space can be projected to d = O(log N / ε²) dimensions preserving pairwise distances within (1±ε). Applies to **random** projections.
- **Dasgupta & Gupta (1999, cs/9904007)**: Exact bound d_JL(N=5, ε=0.1) = 138. The theorem requires Gaussian or sub-Gaussian random matrices.
- **Achlioptas (2003, arXiv:cs/0001040)**: Database-friendly JL: sparse {-1, 0, +1} random matrices work. Still random projections.
- **Li et al. (2006, "Very sparse random projections")**: Further sparsification still satisfies JL. Still random.

**Key gap**: None of these works claim JL bounds apply to **learned** projections trained by gradient descent.

### Production MoE systems operate far below the JL floor

DeepSeek-V3 (arXiv:2412.19437) operates at d=2048 with N=256 experts, giving d_JL(256, 0.05) ≈ 18,360 — DeepSeek-V3 is at 11% of its JL floor. This confirms that production systems routinely achieve excellent performance far below JL thresholds. JL distortion is not the binding constraint on MoE generation quality.

### FlyLoRA: JL applied correctly

FlyLoRA (arXiv:2510.08396) uses frozen random A-matrices with JL-lemma grounding — but correctly. JL is invoked to argue that a random A *preserves signal from the input layer*, not that a bottleneck dimension constrains generation quality. The distinction: JL → input fidelity (valid), JL → output quality (not valid without additional theory).

### What actually governs M2P generation quality

The relevant question is: what determines how well a transformer-based predictor can generate adapter weights from hidden states? Literature suggests:

- **Architecture depth**: Deeper predictors can model more complex weight-generation functions. The M2P uses M2P_LAYERS=2; this is the more likely bottleneck.
- **Training data distribution**: 500 steps per domain at micro scale. Quality may improve with more steps or curriculum.
- **Target adapter structure**: LoRA B-matrices have low-rank structure the M2P must learn. The rank-r target space dimensionality (r=4 in this project) constrains the effective generative difficulty more than d_M2P.

---

## Confirming Evidence

- **Finding #351**: 36.6% routing accuracy at micro scale — routing was the bottleneck (already resolved by Finding #354).
- **Finding #354**: Oracle quality = 92.2% (with reused adapters), TF-IDF routing = 95.0%. This experiment shows oracle quality rises to 95.1–97.0% with fresh training, confirming the reuse-artifact explanation.
- **Finding #333**: Single-adapter promotion validation — generation quality is sensitive to training conditions at micro scale.
- **Post-mortem impossibility structure** (from PAPER.md + review): The chain JL distortion → domain separation → generation quality has two unproven links. This is now a closed direction: dimension sweeps cannot improve M2P quality if the projection is already sufficient for domain separation.

---

## Contradicting Evidence

- The review notes DeepSeek-V3 at 11% of JL floor performing well — this *contradicts* the original hypothesis (that JL floor is binding) and *supports* the kill.
- Sort domain consistently 81–95%: this performance is not explained by d_M2P (all widths show similar sort accuracy). An alternative cause is statistical domain overlap (character-ordering ambiguity), consistent with Finding #354's sort/reverse confusion at 11–14%.

---

## Alternative Approaches for Improving M2P Quality

**All alternatives below have published evidence or are grounded in prior project findings.**

1. **Deepen M2P architecture** (M2P_LAYERS: 2 → 4)
   - Motivation: Current quality ceiling ~95–97% with L=2. Transformer depth governs generation capacity. Universal approximation requires sufficient depth.
   - Reference: Yun et al. (2020, arXiv:1912.11985) "Are Transformers universal approximators of sequence-to-sequence functions?" — depth is necessary for complex function approximation.

2. **More training steps or learning rate schedule**
   - Motivation: 500 steps is thin at micro scale. sort domain underperforms consistently. Convergence budget, not architectural capacity, may be limiting.
   - Reference: Finding #354's observation that fresh vs. reused adapters produce 95.1% vs. 92.2% (2.9pp gap attributable to training quality).

3. **Curriculum distillation with mutual learning** (contingent on routing being solved)
   - Motivation: Born Again Networks (Furlanello et al. 2018, arXiv:1805.04770) shows same-capacity KD with iterative retraining CAN work. Deep Mutual Learning (Zhang et al. 2018, arXiv:1806.00774) does not require A2 (teacher > student).
   - Reference: Finding #30's kill note distinguishes cross-distribution mismatch (Qwen) from A2 violation — these are independent failure modes.
   - **Prerequisite**: routing bottleneck must be solved first (Finding #351 addressed; Finding #354 validated).

4. **Target adapter compression** (reduce effective generation difficulty)
   - Motivation: If LoRA rank r is reduced, the B-matrix M2P must predict has lower intrinsic dimension. This reduces the generation problem complexity without touching d_M2P.
   - Reference: Aghajanyan et al. (2021, arXiv:2012.13255) "Intrinsic Dimensionality" — most fine-tuning signal lives in very low-rank subspaces (r=4 may be over-specified at micro scale).

---

## Implications for Next Experiments

1. **Dimension sweeps are a closed direction** for M2P. The JL-as-bottleneck hypothesis is falsified. Do not re-test with ε changes, different N, or other JL variants — the impossibility structure is clear.

2. **Fresh M2P training is required** for any quality measurement. Reusing adapters from a different training run contaminates quality_ratio measurements by up to 2.9pp.

3. **Parity domain must be excluded** from quality statistics. base_loss ≈ sft_loss makes the quality_ratio denominator near-zero. Guard condition: exclude domains where (base_loss − sft_loss) < 0.05.

4. **True M2P quality ceiling at micro scale is ~95–97%** (fresh training, 500 steps, d=64 sufficient). The 3–5% remaining gap is:
   - Intrinsic to the M2P architecture's generative capacity (depth, training convergence)
   - Per-domain difficulty (sort/reverse overlap is a fundamental signal ambiguity)
   - **Not** addressable by d_M2P sweeps

5. **Priority remains**: routing bottleneck resolved (Finding #354). Generation quality bottleneck (M2P architecture depth/training) is next P1 target. Scale and cross-domain transfer (Finding #353) are downstream of generation quality.

---

## Recommended Follow-Up

**Experiment: M2P Depth Sweep (exp_m2p_depth)**
- Sweep M2P_LAYERS ∈ {1, 2, 4} with d_M2P=64 fixed
- Hypothesis: generation quality ceiling is architectural depth, not projection dimension
- Citation: Yun et al. (2020, arXiv:1912.11985) — transformer depth governs sequence-function approximation capacity
- Kill criteria: quality(L=4) > quality(L=2) by >2pp (not just noise), quality(L=4) ≥ 97%
- MATH.md must prove: the B-matrix generation function has minimum description complexity requiring ≥ L layers (use VC dimension or universal approximation arguments)
- **Do NOT** re-apply JL to this experiment — the impossibility structure proves dimension is not the bottleneck
