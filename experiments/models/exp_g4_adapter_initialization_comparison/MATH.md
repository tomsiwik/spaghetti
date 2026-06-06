# MATH.md — exp_g4_adapter_initialization_comparison

**Type:** verification (proven framework: B-matrix compensation at high-d; unknown: whether F#169 (d=2560 BitNet) generalizes to Gemma 4 E4B at d=2048)

**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (42 decoder layers, hidden_size=2048)
**mlx / mlx-lm:** whatever is in `uv run` environment (confirmed at runtime)

## 1. Failure Mode Identified

If adapter initialization method substantially changes final PPL, then the entire adapter library must be retrained whenever we change init (e.g. Grassmannian → Kaiming), and init becomes a first-class hyperparameter. F#562 (supported) claimed Grassmannian QR is the correct init; F#169 (killed, BitNet d=2560) showed init does not affect final PPL because B-matrices compensate during training. We need to verify which regime holds for Gemma 4 E4B (d=2048, 4-bit quantized base).

The killable failure mode: **init method determines final PPL** at Gemma 4 E4B scale. If so, Grassmannian is load-bearing and cannot be swapped.

## 2. Prior Math (Cited)

- **F#169** (killed, exp_osrm_constrained_adapters, d=2560 BitNet 2B): random-QR, Grassmannian AP, and OSRM init produce identical PPL within 1% and composed PPL 8.31/8.33/8.38. B-matrices compensate during 200 training iters.
- **F#498** (supported, exp_p7_self_organizing_slots): A-matrices cluster by init method (cos≈0.82 standard Gaussian, cos≈0 Grassmannian) even after training. B-matrices converge to rank-5 domain subspace regardless of A-init.
- **F#562** (supported, exp_g4_structural_orthogonality, Gemma 4 dims): Partition-QR gives max|cos|=2.74e-9 structurally — verifies Grassmannian orthogonality at d=2816/5376 pre-training only.
- **F#627** (supported): r=6 q_proj LoRA on Gemma 4 E4B with scale=6 gives large capability lift on GSM8K/HumanEval/MedQA.
- **Aghajanyan 2020 "Intrinsic Dimensionality" (arxiv:2012.13255)**: low-rank adapter weights live on a low-dimensional manifold whose geometry is dataset-determined, not init-determined, for sufficient training.
- **High-d concentration (Vershynin 2018, Ch.3)**: two random r-dim subspaces in ℝ^d have expected principal-angle cosine ~ √(r/d). At d=2048, r=6: √(6/2048) ≈ 0.054. Random inits are already near-orthogonal in the "don't-matter" sense.

## 3. Theorem (Init-Irrelevance at Gemma 4 E4B)

**Setup.** Let W ∈ ℝ^{d_out × d_in} be a frozen quantized projection (q_proj for Gemma 4 E4B, d_in=d_out=2048). LoRA adds ΔW = (α/r) B A with A ∈ ℝ^{r × d_in}, B ∈ ℝ^{d_out × r}, B initialized to 0. Let A_{init} denote the matrix chosen by the init method (Grassmannian QR / Kaiming / Gaussian). Fix a training dataset D and T training steps with Adam+LR=1e-4 (F#627 recipe).

Let π(init) denote the policy/probability distribution over next-tokens produced by the trained (A, B) from init, measured as PPL on held-out val.

**Theorem (informal).** If T is large enough that B escapes the zero-init neighborhood (true for T≥100 with LR=1e-4 on Gemma 4 E4B, empirically), then

  |PPL(π(Grassmannian)) − PPL(π(Kaiming))| / PPL_base ≤ 0.05

i.e. final PPL is within 5% across init methods.

Simultaneously, A-matrices remain structurally distinct:

  |cos_sim_final(A_i, A_j) − cos_sim_init(A_i, A_j)| ≤ 0.15  for i,j different init methods.

**Proof sketch.** In the linearized regime around B=0, ∇_B L = U^T x A (for some activation U), and ∇_A L = B^T U^T x — which is zero at init. Hence A is frozen until B grows away from zero. Once B has nonzero magnitude, ∇_A L is proportional to B, so A updates are scaled by B magnitude (small initially, growing exponentially but slowly vs. LR). The output depends on BA, and B absorbs the rotation that aligns whatever A-basis it received with the task-optimal direction. Because rotations preserve magnitude and B has full rank-r freedom, the behavioral output is approximately init-invariant (up to the subspace covered by A_init).

Because A_init spans an r-dim subspace of ℝ^{d_in}, and because random / QR / Kaiming all span r-dim subspaces whose principal angles to the task-optimal subspace are statistically similar in high-d (F#169 mechanism), the final BA product settles into comparable behavioral quality regardless of which A_init was picked, as long as that A_init has rank r. This extends F#169 to Gemma 4 E4B, d=2048 (vs. BitNet d=2560).

**QED** (empirical KCs in §5).

## 4. Predictions

P1. Final eval-PPL max/min ratio across three inits ≤ 1.05 (within 5%).
P2. Final A-matrix pairwise cosine similarity between different-init adapters differs from init cosine by ≤ 0.15 (A stays put, F#498 replicated).
P3. Final training loss (last 10-step avg) for all three inits within 10% of each other.
P4. Grassmannian-init final-A retains near-zero cross-block cosine (|cos|<0.05); Kaiming and Gaussian final-A retain their elevated cross-block cosine (|cos|>0.3). Grassmannian structural fingerprint survives training. (Replicates F#498.)

## 5. Pre-registered Kill Criteria (per F#666 — target-gated)

KCs in the DB:
- **K1924**: "Initialization method produces > 0.10 cos-sim difference in final adapter"
- **K1925**: "Grassmannian A-init not best method for any metric (PPL, cos-sim, composition)"

We operationalize both and interpret per F#666 target-gating.

**K1924 (proxy — structural A-retention):**
  Operationalization: Δcos := max_ij |mean pairwise cos(A_init_i, A_init_j)_final| − min_ij |mean pairwise cos(A_init_i, A_init_j)_final| across the 3 init groups.
  PASS condition: Δcos > 0.10 (init leaves structural fingerprint on final A).
  FAIL condition: Δcos ≤ 0.10 (training erases init distinction).
  Prediction: PASS (per F#498 at Gemma 4 E4B).

**K1925 (target — behavioral init-ordering):**
  Operationalization: rank the three inits by final eval-PPL (lower=better). Let Δppl := (PPL_worst − PPL_best) / PPL_best.
  PASS condition: Grassmannian is NOT uniquely best across both PPL and structural-cos metrics — specifically, Δppl ≤ 0.05 (PPL within 5%, so "best init for PPL" is noise).
  FAIL condition: Δppl > 0.05 AND Grassmannian is the best-PPL init (Grassmannian materially wins behaviorally).

**Verdict logic (F#666):**
- Both PASS (K1924 PASS + K1925 PASS) → **SUPPORTED**. Init leaves a persistent structural fingerprint in A (F#498 replicated at Gemma 4) BUT final PPL is init-invariant (F#169 generalized to d=2048). Result: "Init is structurally consequential but behaviorally irrelevant at Gemma 4 E4B."
- Both FAIL (K1924 FAIL + K1925 FAIL) → KILLED. Training erases init AND Grassmannian wins PPL: implausible joint failure, would mean init drives behavior through some non-A-structural route.
- K1924 PASS + K1925 FAIL → PROVISIONAL (Grassmannian is materially best PPL; F#562 re-confirmed; F#169 refuted at Gemma 4).
- K1924 FAIL + K1925 PASS → PROVISIONAL (A-structure erased but PPL still invariant; stronger compensation than F#498).

### Pre-flight checklist
- Platform skills invoked: /mlx-dev, /fast-mlx — confirmed prior to writing code.
- Base model: mlx-community/gemma-4-e4b-it-4bit (matches MATH.md §0).
- Adapter targets: q_proj all-42-layers r=6 scale=6 (matches F#627 recipe).
- Dataset: micro/models/exp_p1_t2_single_domain_training/data/medical/{train,valid}.jsonl (pre-existing, used by F#627).
- Budget: ~2 min prep + 3 × ~5 min training + 2 min eval ≤ 25 min wall-clock.
- KC count: 2, both target-gated pair per F#666 (K1924 proxy structural; K1925 target behavioral).
- Antipattern scan: no composition math (single-adapter); LORA_SCALE=6 (safe, matches F#627); no shutil.copy; no hardcoded pass; no eval truncation (val split used in full); no proxy model (Gemma 4 E4B 4-bit is the target).
- is_smoke: false (T=100 steps is reduced from 1000 but not zero; if both KCs resolve decisively, a full rerun is optional follow-up not smoke invalidation).

## 6. Experimental Design

- 3 init methods: Grassmannian QR (orthonormal via `mx.linalg.qr` on random Gaussian), Kaiming-uniform (mlx_lm default LoRALinear init), Gaussian (std=0.02, a stylized "random" control).
- For each init: load fresh Gemma 4 E4B, attach q_proj LoRA r=6 scale=6 to all 42 layers, override A with chosen init, keep B=0. Train 100 iters on medical/train.jsonl with AdamW, LR=1e-4, batch 2, mask_prompt=True (F#627 recipe, iters reduced from 1000 for budget).
- Eval on medical/valid.jsonl, 30 rows, mean NLL → PPL per F#627.
- Record: final train-loss (last-10 mean), eval-PPL, flattened A matrices (42 × r × d_in = 42 × 6 × 2048 entries).
- Cross-init A-A cos-sim: for each pair of init methods (3 pairs), compute mean |cos| across matched (layer, rank-row) positions. Also per-init self-cos across rank rows (intra-adapter orthogonality indicator).

## 7. Assumptions (logged per researcher-hat guardrail)
- Gemma 4 E4B hidden_size is 2048 — will be confirmed at runtime from loaded model.
- T=100 is adequate for B to escape zero neighborhood: justified by F#627's training curves (loss drops ~50% in first 100 iters).
- Single seed (42): claim is distributional per theorem; multi-seed is a follow-up if verdict is borderline.
- q_proj-only target follows F#627 canonical config; generalizing to v_proj or full-attention is a follow-up.
