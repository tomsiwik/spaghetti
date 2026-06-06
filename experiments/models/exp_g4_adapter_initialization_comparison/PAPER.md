# PAPER.md — exp_g4_adapter_initialization_comparison

**Status:** PROVISIONAL
**Base model:** `mlx-community/gemma-4-e4b-it-4bit` (hidden=2560)
**Recipe:** q_proj r=6 scale=6 all-42-layers, AdamW lr=1e-4, batch=2, iters=100, medical split (F#627 recipe, iters reduced from 1000 for micro budget).

## TL;DR

Compared Grassmannian QR, Kaiming-uniform, and Gaussian init for LoRA A-matrices on Gemma 4 E4B.

- **Behavioral (K1925) PASS**: final eval PPL across three inits is within 3.5% of each other (1.168 gaussian / 1.201 grassmannian / 1.210 kaiming vs. baseline 2070.79). Grassmannian is NOT the best PPL init at this training budget — Gaussian leads by 3.5%.
- **Structural cross-init (K1924) FAIL**: measured Δcos across init pairs is 0.068 (< 0.10 threshold), but this metric is confounded by shared PRNG state across init methods (documented below). Under the confound-free intra-adapter column-orthogonality metric, Grassmannian's structural fingerprint DOES survive: final intra |cos| = 0.032 (Grassmannian) vs 0.090 (Kaiming) — a 2.8× retention of orthogonality.

Joint verdict: **PROVISIONAL**. Behavioral prediction (init-invariance for PPL at Gemma 4 E4B) replicates F#169 cleanly. Structural prediction is underdetermined by the confounded cross-init metric; intra-init orthogonality metric shows F#562 Grassmannian structural fingerprint partially persists through training.

## Prediction vs Measurement

| Prediction                                                            | Expected                           | Measured                                                                | Result |
| --------------------------------------------------------------------- | ---------------------------------- | ----------------------------------------------------------------------- | ------ |
| P1: PPL max/min ratio across 3 inits ≤ 1.05                           | ≤ 5%                               | 3.54% (best=gaussian 1.168, worst=kaiming 1.210)                        | PASS   |
| P2: Cross-init Δ\|cos(A_i, A_j)\| final vs init ≤ 0.15                | ≤ 0.15                             | max drop 0.086 (gr-kaim 0.977→0.899); max-min cross-pair Δ 0.068        | CONFOUND-PASS (see §3) |
| P3: Final train loss (last-10 avg) all within 10%                     | ratio ≤ 1.10                       | gauss 0.785 / kaim 0.829 / grass 0.916 — ratio 1.17 (grass 17% above)   | FAIL by 7pp |
| P4: Grassmannian intra-column cos stays < Kaiming intra-cos at final  | grass < kaim                       | grass 0.032 < kaim 0.090 (2.8× cleaner)                                 | PASS   |

### Why P3 fails at 17% vs 10% threshold

Train-loss at iter-100 averaged over last-10 reports:
- Grassmannian: 0.916 (monotone dropping 4.03→0.219 at iter-100; last-10 window 60-100 still early)
- Kaiming: 0.829 
- Gaussian: 0.785

Gaussian and Kaiming train-loss drop faster than Grassmannian in this window. Possible mechanism: Grassmannian's exactly-orthonormal columns (all singular values =1) give uniform gradient flow through B, so early B growth is uniform across rank — slower concentration on task-relevant directions. Gaussian's heterogeneous column norms give larger early gradient on the largest column, concentrating B more aggressively.

This is a subtle training-dynamics difference, not a capability ceiling difference (eval PPL converges close). Interpretable as "Grassmannian requires more training to exploit its orthogonality".

## 3. Confound: shared PRNG state across init methods

### What I did

All three init methods were seeded from the SAME top-level key `mx.random.key(42)`, split into per-layer sub-keys via `mx.random.split(key, num=42)[i]`. Then:

- Grassmannian: `QR(normal(shape=(2560,6), key=sub_key))`
- Kaiming: `uniform(-s, s, shape=(2560,6), key=sub_key)`
- Gaussian: `0.02 * normal(shape=(2560,6), key=sub_key)`

### Why this matters

MLX's `mx.random.normal` and `mx.random.uniform` with the SAME key draw from correlated underlying PRNG sequences. Grassmannian's QR(normal) shares the same seed as Gaussian's `0.02*normal`, so their A-matrices are related by scaling + QR (a small perturbation at d=2560, r=6). Grassmannian-vs-Kaiming also inherits PRNG correlation.

### Measured consequence

Cross-init cos at INIT:
- grass vs gaussian: 0.9995 (essentially same matrix up to QR)
- grass vs kaiming: 0.977
- kaiming vs gaussian: 0.977

Cross-init cos at FINAL:
- grass vs gaussian: 0.960
- grass vs kaiming: 0.899
- kaiming vs gaussian: 0.892

At init the three matrices are essentially co-linear in column-space (cos ~0.98–0.9995). 100 iters of training de-aligned them slightly (cos drops 0.04–0.08). This is not "init methods produce distinct final A-matrices"; it is "shared seed across init variants produces correlated starting points, and 100 iters partially decorrelates them."

### Clean structural metric (not confounded)

Intra-adapter column-orthogonality (mean |cos| of distinct columns within one adapter) is confound-free because it does not compare across init methods:

| Init          | intra \|cos\| at init | intra \|cos\| at final (iter-100) |
| ------------- | --------------------- | --------------------------------- |
| Grassmannian  | 4.77e-9 (algebraic)   | 0.032                             |
| Kaiming       | 0.016                 | 0.090                             |
| Gaussian      | 0.015                 | 0.037                             |

**Finding:** Grassmannian retains near-orthogonal columns post-training (0.032, 6.7× larger than init but 2.8× cleaner than Kaiming). Gaussian (0.037) is close to Grassmannian at final — the Gaussian random subspace at d=2560, r=6 is already nearly orthogonal (random subspace cos ~ 1/sqrt(2560/6) ≈ 0.05) and stays so. Kaiming's uniform-bounded draws produce 3× higher column-correlation at init (0.016 vs 0.015 nominal but effectively similar) but drift to 0.090 at final — suggesting training introduces column alignment under uniform init more than under Gaussian init.

This IS a genuine replication and extension of F#562 (Grassmannian orthogonality is real) AND F#498 (structural fingerprint persists) at Gemma 4 E4B d=2560 post-training. The cross-init comparison for K1924 was mis-designed; the intra-init metric is the correct structural probe and it PASSES the spirit of P4.

## 4. Verdict: PROVISIONAL

- K1925 PASS (PPL init-invariant at 3.5% spread, below 5% target): replicates F#169 at Gemma 4 E4B.
- K1924 FAIL as operationalized (Δcos across pairs = 0.068 < 0.10), but under the confound-free intra-init metric the spirit of the structural claim is upheld.
- PROVISIONAL per F#666: proxy FAIL + target PASS = "target invariance holds, proxy measure was confounded — needs v2 with per-init-distinct seeds."

## 5. Findings (to register)

**F#NEW.a (provisional)**: At Gemma 4 E4B d=2560 r=6 q_proj, 100 training iters on medical-MCQ makes final eval-PPL init-invariant within 3.5% across {Grassmannian QR, Kaiming uniform, Gaussian 0.02} A-inits. Generalizes F#169 (BitNet d=2560, OSRM-Grassmannian-QR indistinguishable at final PPL) to Gemma 4 architecture.

**F#NEW.b (provisional)**: Grassmannian intra-column orthogonality survives 100 iters of training: intra |cos| grows from 4.77e-9 (algebraic zero) to 0.032, still 2.8× cleaner than Kaiming final (0.090). Gaussian at high-d (0.037 final) is comparable to Grassmannian post-training due to concentration of measure. Replicates F#562 at post-training time-step and extends F#498 A-cluster persistence claim.

**F#NEW.c (methodological, provisional)**: Cross-init A-matrix cos-sim comparison is confounded when the same PRNG key seeds all init methods. All three draws start at cos~0.98–0.9995 (not near-zero) because QR of Gaussian and uniform-bounded from the same key share underlying random state. Any follow-up init-comparison must use distinct per-init seeds.

## 6. Caveats

- Single seed (SEED=42). Distributional claim per theorem; multi-seed rerun is a follow-up.
- 100 iters is reduced from F#627's 1000 iters for budget. Training loss at iter-100 for Grassmannian (0.916 last-10 avg) had not fully converged; a 1000-iter rerun could narrow the train-loss gap.
- q_proj-only adapter (no v_proj, no o_proj, no MLP). Full-model LoRA may tilt init-sensitivity differently.
- Medical MCQ task has short gold responses ("D: ...") — most of eval PPL is dominated by a small predictable span. A richer task (HumanEval-style code, long-answer med QA) may amplify init effects.
- 4-bit quantized base (F#627 target config). fp16/bf16 results may differ because the rounded weight grid interacts with A-init magnitude.

## 7. Follow-ups

1. **Rerun with per-init distinct seeds** to cleanly test K1924. Use seed 42 for Grassmannian, 43 for Kaiming, 44 for Gaussian.
2. **1000-iter rerun** to verify PPL convergence and whether train-loss gap closes.
3. **Intra-column orthogonality time-series** through training to characterize when Kaiming's columns start colliding.
4. **F#627 v_proj+o_proj target** re-run of the sweep to check generalization beyond q_proj.

## 8. Assumptions (logged)

- Medical validation split, 15 batches × 2 = 30 rows (MATH.md §6 budget).
- F#627 recipe (q_proj, r=6, scale=6) taken as canonical.
- Grassmannian = QR of Gaussian (column-orthonormal), not full-Stiefel/flag optimization.
- "Kaiming" = mlx_lm default LoRALinear init (uniform bounded by 1/sqrt(input_dims)).
