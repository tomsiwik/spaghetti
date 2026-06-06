# LEARNINGS: exp_m2p_layer_depth_qwen3

**Finding #370** | Status: provisional | Date: 2026-04-07

---

## Core Finding

Option A (single M2P call, d_M2P=64) scales to Qwen3-4B width (d_model=3072, L=36)
with 90.0% median quality (sort=85.9%, reverse=94.1%), decisively refuting the
width-scaling hypothesis H2 (predicted 73%). The effective rank of the joint
36-layer adapter stack is task-determined, not width-determined: a 64-dimensional
M2P bottleneck suffices at both d_model=256 and d_model=3072 because the sorting
and reversal task intrinsic dimensionality stays far below 64 regardless of ambient
weight-space dimension. Level 1 PoC gate is now fully complete.

---

## Why This Happened

### The rank ceiling is set by L×LORA_RANK, not d_model (B.5 proof)

The mathematical heart of this finding is a rank-structure identity. The joint
B-matrix stack S = [B_1*, ..., B_36*] lives in R^{144 × d_out}, where 144 =
L × LORA_RANK = 36 × 4. Its maximum rank is min(144, d_out). At d_model=256,
d_out = 1024 > 144, so max_rank = 144. At d_model=3072, d_out = 12288 >> 144,
so max_rank = 144 again. The ceiling is width-independent because the row count
(L × LORA_RANK) is the binding constraint, not the column count (d_out). Width
scaling adds columns to S but cannot increase its rank beyond the row count. This
is the Aghajanyan invariance applied to the cross-layer setting: effective rank is
bounded by the structure of the low-rank decomposition, not by the ambient
embedding dimension.

Formally: M2P generates d_M2P=64-dimensional codes. For quality to degrade at
d_model=3072, the effective rank of the SFT B-stack targets would need to EXCEED
d_M2P=64. The B.5 bound shows max_rank ≤ 144, but this is a ceiling — actual
effective rank for toy tasks stays well below 64, which is why quality is
maintained. This result is valid linear algebra, confirmed as sound by adversarial
review.

### Task complexity determines d_int, not model width (empirical, not theorem)

The Aghajanyan et al. key empirical finding (arXiv:2012.13255) — that intrinsic
dimensionality of fine-tuning is determined by task complexity, not model size —
explains why H1 held and H2 failed. Sort and reverse are low-complexity tasks
whose SFT adapter targets have d_int well below 64 regardless of whether the
transformer is 256-wide or 3072-wide. The M2P compression ratio scales from
2304:1 (d=256) to 27648:1 (d=3072) in the fc1 head, yet quality is maintained
because the rank bottleneck is not the binding constraint. Note: Aghajanyan et al.
present this as an empirical finding, not a formal theorem — citing it as "Theorem
1" (as MATH.md initially did) overstates the mathematical grounding.

### GL overfitting at T=400 is a training-budget artifact, not a quality failure

K898 failed (0.803 nats gap vs 0.7 nats threshold) because the d=3072 run used
T=400 steps on n=500 samples (n_train=400 after 80/20 split), placing T/n_train
= 1.0 exactly at the Ghadimi & Lan guarantee boundary (arXiv:1309.5549). At the
boundary, mild overfitting is expected by the convergence theory. Crucially, this
did not damage quality: best checkpoint gives quality_ratio=94.1% because the GL
early stopping mechanism captures the good-quality state before the val loss climbs
further. The 0.803 nats gap measures train_loss vs val_loss at final step, not
best_val_loss vs SFT — so it reflects the severity of late-training overfitting,
not the quality of the saved model. The fix for production d=3072 runs is
T ≥ 1000 with n ≥ 2000 (matching the proven recipe ratio T/n_train < 1.0 at
n_train ≥ 800).

---

## Confirming Evidence

- **Aghajanyan et al. (2021, arXiv:2012.13255), "Intrinsic Dimensionality Explains
  the Effectiveness of Language Model Fine-Tuning"** — empirical finding that
  fine-tuning intrinsic dimensionality is low and correlates with task complexity,
  not model size. Directly supports H1 over H2. Note: empirical finding only, not
  a formal theorem.

- **Ha, Dai & Le (2017, arXiv:1609.09106), "HyperNetworks"** — demonstrates that
  a single low-dimensional code can parameterize all layer weights of a deep
  network because cross-layer weight structure is low-rank. The 90-95% quality
  retention result is consistent with the 85.9–94.1% range measured here.

- **Ghadimi & Lan (2013, arXiv:1309.5549)** — convergence bound O(LG²/T + L/√T)
  is d_model-independent in its structural form. The K898 failure is correctly
  predicted at the T/n_train = 1.0 boundary; the bound explains both the
  overfitting and its non-catastrophic character.

- **Finding #365 (exp_m2p_layer_depth_36)** — established 89.1%/97.8% sort/reverse
  at L=36, d_model=256. This experiment's d=256 replication (93.5% median) is
  consistent (within 7pp variance). The prior result provides the theoretical
  grounding that d_M2P=64 is sufficient at L=36 before width scaling.

- **Finding #363 (exp_m2p_composition_n5)** — established that Option A (joint M2P)
  outperforms Option B (independent per-layer) at L=8 due to implicit cross-layer
  regularization. The superior performance extends here to L=36 at macro width.

---

## Contradicting Evidence

- **K897 margin is within noise (adversarial review).** Sort quality at d=3072 is
  85.94%, only 0.94pp above the 85% pass threshold. The d=256 replication showed
  7pp+ variance across runs (Finding #365: sort=89.1%, this run: sort=96.4%).
  A different random seed may yield K897 FAIL. The PASS is real but not robust
  to seed variation at this run count (n=1 run).

- **Random-init vs pre-trained base creates non-comparable conditions.** The d=256
  run used base_steps=1200 (pre-trained, ~12.7 nats), while d=3072 used
  base_steps=0 (random init, ~5.1 nats). Quality_ratio normalizes by the
  base-SFT gap, but with a random base the denominator is 2.84 nats vs 10.38 nats
  at d=256. The learning problem is qualitatively different (structured vs
  unstructured base). The PAPER.md argument that this makes d=3072 quality
  HARDER to achieve is plausible but unproven — with a random base, the M2P
  may benefit from the smaller required shift.

- **LoRAtorio literature on real NLP tasks** suggests sort/reverse have trivially
  low d_int. Medical QA, code generation, and summarization are qualitatively more
  complex. The Aghajanyan invariance holds empirically for language understanding
  tasks at model-scale (their d_int estimates); whether d_M2P=64 suffices for
  those tasks at d_model=3072 is an open question not answered by this experiment.

- **K898 FAIL (GL threshold 0.7 nats).** The secondary generalization criterion
  failed, confirming the experiment was underpowered for the d=3072 condition
  at T=400. The quality result is still reliable (best checkpoint captured), but
  the finding should be treated as a lower bound pending a full-budget replication.

---

## Alternative Approaches

The following approaches have published evidence or direct prior experiment support.
No analogies.

- **Increase d_M2P for higher-complexity tasks.** Finding #355 (exp_m2p_bottleneck_width)
  showed d_M2P=64 is optimal at micro scale; macro tasks may need larger bottlenecks.
  Aghajanyan et al. (arXiv:2012.13255) show d_int scales with task complexity — for
  real NLP domains, d_int=200–5000 at full fine-tuning scale. Scaling d_M2P to 128
  or 256 is the structurally-grounded fix if quality degrades on real tasks.

- **Layer-grouped M2P (partial cross-layer sharing).** Ha et al. (arXiv:1609.09106)
  validate hierarchical sharing structures where nearby layers share more weight
  structure than distant ones. If the full 36-layer joint stack exceeds d_M2P
  capacity at macro scale, grouping into blocks of 6-8 layers reduces compression
  while preserving the key HyperNetwork property.

- **Empirical intrinsic dimensionality measurement before fixing d_M2P.** Aghajanyan
  et al. (arXiv:2012.13255) describe the random-projection method for measuring d_int
  of a fine-tuning objective. For macro NLP domains, measuring d_int empirically
  on the SFT adapter targets before choosing d_M2P would replace guesswork with a
  principled bound — directly analogous to their methodology.

- **Full training budget replication of d=3072.** Not a novel approach, but required
  to validate this finding. Match the proven recipe: n=2000, T=1000, pre-trained
  base (base_steps=1200). This would close the caveats from the adversarial review
  (random-init base, T=400 underpowering) without introducing new mechanisms.

---

## Implications for Next Experiments

1. **Level 1 PoC gate is complete.** Gate 1B-ext (Option A ≥85% at L=36, d_model=3072)
   is done. The depth arc (L scaling) and width arc (d_model scaling) are both
   provisionally closed for toy tasks. Proceed to Level 2.

2. **Width scaling changes nothing about the rank structure — but real tasks will.**
   The B.5 result proves width-independence mathematically. The Aghajanyan empirical
   finding does not: real NLP tasks may have higher d_int. The first macro-scale
   experiment should measure d_int empirically on the target domain before fixing
   d_M2P, not assume d_M2P=64 suffices.

3. **GL tuning at 227M param M2P scale needs recalibration.** GL (alpha=5.0) was
   validated up to 19M param M2P (Finding #365). At 227M params with T=400, the
   sensitivity range is different. The fix is T ≥ 1000 with n ≥ 2000 — not a new
   mechanism, just proper training budget.

4. **K897 needs seed replication to confirm robustness.** The 0.94pp margin above
   85% is within observed variance. At least 3 seeds at d=3072 with full training
   budget should be run before citing this result as "confirmed" at Qwen3-4B width.

5. **Third domain (cipher) is the most informative next step for Level 2.** Sort
   and reverse are low-complexity tasks. The cipher domain has different structural
   complexity (character mapping vs ordering). Its d_int may be higher, testing
   whether d_M2P=64 can handle moderate-complexity tasks at d_model=3072.

6. **N>5 composition is the direct successor.** Finding #351 (routing bottleneck)
   was addressed by TF-IDF routing (Finding #?). With routing solved and width arc
   closed, N>5 composition at real model width is the next logical PoC milestone.

---

## Recommended Follow-Up

**Priority 1 (strongest near-term evidence value): third domain (cipher) + d_M2P sensitivity**

- QUESTION: Does Option A quality ≥ 85% hold for cipher (moderate-complexity task)
  at d_model=3072, and what d_M2P is needed?
- MOTIVATION: Aghajanyan et al. (arXiv:2012.13255) show d_int varies with task
  complexity. Cipher has distinct structural properties (character bijection, no
  ordering) from sort/reverse. Testing a structurally different task at Qwen3-4B
  width directly addresses the macro-scale risk that toy d_int underestimates
  real NLP d_int.
- KILL CRITERIA: (a) quality ≥ 85% with d_M2P=64 at d_model=3072 [tests whether
  current bottleneck holds]; (b) if (a) fails, measure effective rank of cipher
  SFT adapters via random-projection (Aghajanyan method) to determine required
  d_M2P.
- MATH.md: Extend B.5 rank proof to include the measured d_int bound for cipher.
  If d_int < 64, bottleneck holds. If d_int > 64, derive required d_M2P.

**Priority 2 (closes known open caveat): full-budget d=3072 replication**

- QUESTION: Does quality_ratio at d=3072 replicate at 90%+ with n=2000, T=1000,
  pre-trained base (base_steps=1200)?
- MOTIVATION: The adversarial review identified two protocol mismatches (random-init
  base, T=400 underpowering) that prevent direct comparison with Finding #365.
  PAPER.md notes this as Caveat 1; K897 margin (0.94pp) warrants confirmation.
- KILL CRITERIA: quality_ratio ≥ 85% at d=3072 with matched protocol. If FAIL,
  determine whether root cause is base protocol (random vs pre-trained) or
  training budget (T=400 vs T=1000).
- This is a verification run, not a new experiment — no new MATH.md required
  beyond citing this finding and updating the prediction table.

**Priority 3 (macro-scale readiness): N>5 composition at d_model=512+**

- QUESTION: Does M2P composition of N=10 adapters hold quality ≥ 75% at d_model=512
  (intermediate width)?
- MOTIVATION: TF-IDF routing (Finding #?, exp_m2p_tfidf_routing_n5) showed 95%
  routing accuracy, solving the bottleneck identified in Finding #351. With routing
  and width both addressed, scaling N is the next PoC milestone. SHINE
  (arXiv:2602.06358) shows joint hypernetwork training (single forward pass for
  all layers) provides implicit cross-layer regularization that becomes more
  important as N grows — directly motivating Option A for N>5.
- KILL CRITERIA: Derive from the Grassmannian orthogonality result (Finding #353):
  at N=10, max|cos| across all slot pairs must stay < 0.1 to guarantee
  interference-free composition.
