# LEARNINGS: M2P Data Scale (exp_m2p_data_scale) — SUPPORTED

## Core Finding

Early stopping (GL criterion) is the dominant regularizer at micro scale (+7.6pp),
not data scale alone (+0.6pp). Together, they break the cyclic-overfitting ceiling:
M2P quality reaches **97.6% of SFT** (up from 89.4%) once n_train ≥ T=1000 and GL
early stopping is active. Micro-scale architecture search is now complete: width
(Finding #355), depth (#357), training budget (#358), and data scale (#359) are all
mapped. The quality ceiling at micro scale is ~97–98% of SFT once overfitting is
controlled.

---

## Why This Happened

### Root Cause: Cyclic memorization violates SGD i.i.d. assumption

The Ghadimi-Lan convergence theorem (arXiv:1309.5549) requires unbiased gradient
estimates. When T > n_train, the gradient at step t cycles deterministically over the
same samples (`x_{t mod n_train}`), so the estimate becomes increasingly biased as the
model memorizes training tokens. Finding #358 confirmed this: at T=2000, n=500 (5×
cycling), eval loss spiked while train loss fell — textbook overfitting. The fix is
structural: set n_train ≥ T so no sample is revisited within one training pass.

### Why early stopping dominated (+7.6pp of the +8.2pp gain)

The data scale fix alone (+0.6pp from n=500→n=2000 within experiment) is surprisingly
small. The dominant gain came from two confounded changes vs. Finding #358: doubling T
from 500→1000, AND adding GL early stopping. The GL criterion (Prechelt 1998) halted
training in 3/4 domains at n=500 — acting as implicit L2 regularization (Ying,
arXiv:1901.09415) that prevents memorization even when n_train < T. This is the
counter-intuitive result: the backward direction of Theorem 1 ("T > n implies
degradation") does NOT hold when early stopping is active. The structural guarantee
(n_train ≥ T) is sufficient, not necessary.

### Why the Hardt bound was off by 270×

Theorem 2 uses the Hardt et al. (2016) bound: train_val_gap ≤ 2L/n_train ≤ 0.001 nats.
Measured: 0.337 nats. The Hardt bound requires convex loss and is inapplicable to
non-convex M2P transformers with GELU activations. The qualitative threshold (< 0.5 nats,
grounded in Finding #358 observed values) correctly predicted K879. For future
experiments: do not use the Hardt formula to calibrate thresholds for non-convex losses.

---

## Literature Context

### Early stopping as implicit regularization

- **Prechelt (1998), "Early Stopping — But When?"** — Defines the Generalization Loss
  (GL) criterion with threshold α=5.0. GL(t) > α triggers stop. This is the standard
  textbook method; our experiment confirms it is highly effective even at micro scale
  with small validation sets (80 samples per domain at n=500).

- **Ying (2019, arXiv:1901.09415), "An Overview of Overfitting and its Solutions"** —
  For quadratic objectives, early stopping at T* is equivalent to L2 regularization with
  λ_reg = 1/(2ηT*). The implicit regularization interpretation explains why early stopping
  at n=500 achieved 97.0% despite being in the cycling regime: it stopped training before
  memorization compounded.

- **Hardt, Recht & Singer (2016), "Train Faster, Generalize Better"** — Provides
  stability-based generalization bounds for SGD. Bound is tight for convex objectives but
  vacuous (~270× off) for non-convex transformers. Useful for qualitative analysis only;
  do not use quantitatively for non-convex losses.

### Benign vs. non-benign overfitting

- **Bartlett et al. (2020, arXiv:1906.11300)** — Benign overfitting in linear regression
  requires n >> d_eff (effective parameters). At n=500 << d_eff≈O(1000) for the M2P
  B-matrix regression targets, the regime is non-benign. This predicts generalization
  degrades with more training on the same data — confirmed in Finding #358 (T=2000 on
  n=500: quality 89.4% → 83.0%). With n=2000, n_train=1600 ≥ d_eff threshold for most
  domains — exits the non-benign regime.

- **Ghadimi & Lan (2013, arXiv:1309.5549)** — The i.i.d. gradient assumption is the
  load-bearing condition for O(1/T) training-loss convergence. The theorem bounds training
  loss, NOT generalization loss — this gap (training vs. generalization) is what cyclic
  sampling exploits.

### Confirming literature on data sufficiency for hypernetworks

- **SHINE (2026, arXiv:2602.06358)** — identifies prior hypernetwork failures as
  data-starved, not architecture-limited. Confirms that training data quantity is the
  primary bottleneck once architecture is sufficient. Our experiment is consistent:
  width (Finding #355) and depth (Finding #357) were already closed before data scale
  became the active variable.

- **HyperLoader (2024, arXiv:2407.01411)** — trains LoRA-generating hypernetworks
  across many tasks; more diverse training data, not deeper generators, was key to
  quality. Confirms data sufficiency > architectural complexity for weight generators.

---

## Confirming Evidence

- **Finding #358 (KILLED)**: Cyclic overfitting (T=2000 on n=500) was confirmed, not
  hypothesized. Reverse domain: train loss 2.01→1.45, eval loss 2.80→3.86. This
  experiment reproduces the same overfitting signature at n=500/T=1000 (K879: gap=0.873)
  and eliminates it at n=2000/T=1000 (gap=0.337 < 0.5 threshold). The structural
  prediction (n_train ≥ T kills memorization) is verified.

- **Arithmetic domain trend**: The only domain without overfitting at n=500 (base model
  already solves it) showed correct O(1/T) behavior in Finding #358 (89.6→92.0→93.5%).
  This confirmed the theorem was structurally sound. The same O(1/T) logic now governs
  all valid domains with n=2000.

- **GL early stopping triggers**: n=500 triggered GL in 3/4 domains (overfitting
  confirmed); n=1000 triggered in 1/4; n=2000 triggered in 1/4 (only the hard reverse
  domain). The trigger count directly tracks overfitting severity — consistent with the
  structural prediction.

---

## Contradicting Evidence

- **Theorem 1 backward direction refuted**: n=500/T=1000 (2.5 epochs, violates n_train≥T)
  achieved quality=97.0% — far better than the predicted <89.4%. This directly falsifies
  the "if and only if" direction: early stopping rescues quality even when the structural
  guarantee doesn't hold. The finding is corrected to a sufficient condition only.

- **Hardt bound fails quantitatively**: Predicted ≤0.001 nats, measured 0.337 nats.
  Non-convex losses invalidate the Hardt stability approach. This is a known limitation
  of the theory for neural network training.

- **K881 monotonicity FAILS**: Per-domain quality is NOT monotone in n (parity:
  96.5→101.7→98.4, repeat: 94.5→91.0→93.1). The variance of 2–5pp at micro scale (single
  run per condition) exceeds the step-size of the effect (+0.6pp for data scale alone).
  Monotonicity across 3 noisy single-run measurements was too strong a prediction.

---

## Alternative Approaches

**All approaches below have paper references or prior experiment support.**

1. **Macro-scale M2P training (Qwen3-4B base)**
   - Motivation: Micro-scale architecture search is complete. The ~97% ceiling at micro
     is consistent with the SHINE finding that training scale (data volume) governs
     hypernetwork quality. At macro scale (d_model=3584), n* scales by the ratio of
     B-matrix target dimension → may require n_train >> 2000.
   - Evidence: SHINE (arXiv:2602.06358) scales to 6B tokens; our n=2000 at micro is
     orders of magnitude below that. The n* = T/0.8 formula gives the minimum, not the
     optimum.
   - What to check: verify n* = T/0.8 formula against actual convergence curves at macro
     scale; GL threshold may need tuning (modern alternative: patience on raw val loss).

2. **Diverse synthetic data (cross-domain M2P training)**
   - Motivation: HyperLoader (arXiv:2407.01411) shows diverse training tasks, not just
     data quantity, improve hypernetwork generalization. The current 5-domain synthetic
     setup is narrow. Cross-domain diversity may lower n* by providing more varied
     gradient signal.
   - Evidence: SHINE was trained across diverse tasks from OpenHermes; our synthetic
     domains (sort, reverse, repeat, parity, arithmetic) share token-level statistics.
     Qualitatively different domains would stress the M2P mapping differently.

3. **Real-data adapters as M2P training targets**
   - Motivation: The micro-scale M2P is trained on synthetic-domain adapters; the macro
     goal is real adapters (medical/code/math/legal/finance). The quality ceiling may
     differ because real adapters have higher effective intrinsic dimensionality (they
     capture complex real-world structure, not algorithmic patterns).
   - Evidence: Finding #225 (near-lossless composition at N=5 with real adapters);
     LoRAtorio (arXiv:2508.11624) shows real-data adapter ranks have higher effective
     intrinsic dimension than synthetic targets.

4. **Early stopping protocol modernization**
   - Motivation: REVIEW-adversarial.md flags that GL criterion (Prechelt 1998) may
     need tuning at macro scale. Modern alternatives: cosine annealing + patience on
     raw val loss; NoamDecay with warm-up (Vaswani 2017). The GL threshold (α=5.0) is
     a heuristic from small networks.
   - Evidence: Macro-scale training typically uses learning rate scheduling rather than
     GL-based early stopping. At T=1000 steps (micro), GL is fine; at T>>10K steps
     (macro), a warm-up + cosine schedule is more robust.

---

## Implications for Next Experiments

1. **Micro-scale M2P architecture search is closed.** Width (d_M2P), depth (M2P_LAYERS),
   training budget at fixed n, and data scale are all mapped. Quality ceiling at micro
   is ~97–98% of SFT. Do not re-open any of these dimensions without a proof that the
   closed dimension is different at macro scale.

2. **Early stopping is non-negotiable.** The +7.6pp from GL criterion is the dominant
   regularizer at micro scale. All future M2P training (micro and macro) must include
   GL or equivalent early stopping. Without it, quality degrades by ~8pp.

3. **n* = T/0.8 is the minimum for the structural guarantee.** Use n_train ≥ T as
   the design constraint when setting training parameters. At T=1000, n_per_domain ≥ 1250.
   At macro scale T, scale n proportionally.

4. **Effect decomposition is mandatory.** When changing multiple variables between
   experiments (T, n, early stopping), always run within-experiment decomposition to
   isolate individual effects. The confounded +8.2pp headline was corrected to +7.6pp
   (early stopping) + +0.6pp (data scale) only after adversarial review.

5. **Hardt bound is not usable for non-convex losses.** Use empirically-grounded
   thresholds (from prior experiment observations) rather than theoretical bounds from
   the convex optimization literature when setting kill criteria for transformer training.

6. **Quality > 100% needs to be addressed.** The quality_ratio metric allows values
   > 100% (parity at n=1000: 101.7%). For macro-scale experiments, clip or redefine
   the metric to avoid confusion.

---

## Recommended Follow-Up

**Priority 1: Macro-scale M2P training (direct continuation of P0 roadmap)**
- Move from micro (d_model=256) to macro (d_model=3584, Qwen3-4B base)
- Compute n* = T/0.8 for the macro M2P training budget
- Use GL early stopping (α=5.0) as validated; optionally test cosine annealing
- MATH.md must: verify the n_train ≥ T structural guarantee holds at macro scale;
  compute d_eff estimate for macro B-matrix targets (rank-4 LoRA on 3584-dim attention)
- Kill criteria: quality(macro) > 90% of SFT at macro scale (a conservative floor;
  micro achieved 97.6%, macro may be harder due to larger B-matrix target space)
- Citations: SHINE (arXiv:2602.06358) + this experiment (#359) as the micro baseline

**Priority 2: Composition quality at n=5 real domains with M2P routing**
- The exp_m2p_cross_domain_graph (#353) showed M2P routing works across 5 domains.
  Now that quality ceiling is confirmed at 97.6%, test composition of M2P-generated
  adapters (not just SFT adapters) for the Room Model serving scenario.
- Citations: Finding #353 (cross-domain composition), Room Model architecture,
  arXiv:1609.09106 (HyperNetworks) for the compositional setup.
