# SFT 24 Domain Adapters: Proof Verification Report

## Theorem

**Proposition 1 (from MATH.md):** SFT training with instruction masking on BitNet-2B-4T
using TernaryLoRA (rank=16, scale=20, Grassmannian A, 300 steps, lr=1e-4, Adam) produces
adapters that converge on all domains with sufficient data (>= 100 samples with response
markers). This extends Finding #206 (5/5 convergence) to N=24.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: >= 22/24 domains improve (val loss < base) | **24/24** | YES (100%) |
| P2: Mean val loss reduction 10-25% | **17.3%** | YES (within range) |
| P3: 0/24 diverge (training fails) | **0/24 diverge** | YES |
| P4: ~72 min total training | **45.4 min** | YES (faster than predicted) |
| P5: Weakest domain >= 3% improvement | **5.9% (finance)** | YES |

## Hypothesis

SFT training with Grassmannian TernaryLoRA produces convergent domain adapters for all
24 domains of the SOLE architecture, extending the proven N=5 recipe to full scale.

## What This Experiment Is

This experiment retrains all 24 domain adapters using Supervised Fine-Tuning (SFT) loss
instead of Next-Token Prediction (NTP). The SFT loss masks instruction tokens, computing
cross-entropy only on response tokens. This concentrates gradient updates on the domain-specific
response generation, reducing noise from shared instruction format.

**Architecture:**
- Base: BitNet-b1.58-2B-4T (ternary, unpacked to bf16 for training)
- Adapter: TernaryLoRA (rank=16, scale=20, STE quantization on B-matrix)
- A-matrices: Frozen Grassmannian skeleton (pre-computed orthonormal frames)
- Training: 300 steps, Adam, lr=1e-4, 400 samples/domain, max_seq=256

**Critical fix:** The domain ordering was corrected to match the NTP experiment's
`active_domains` ordering (medical, code, math, ..., music), ensuring each domain
receives the correct Grassmannian A-matrix from the skeleton.

## Key References

- Finding #187: SFT > NTP on 4/5 domains by judge score (provisional)
- Finding #206: All 5 SFT adapters converge on BitNet-2B-4T
- Finding #54: N=24 real-data adapters supported (Grassmannian orthogonality stable)
- Zhou et al. (2305.11206): LIMA -- Less Is More for Alignment

## Empirical Results

### Per-Domain Results

| Domain | Base Loss | SFT Loss | Improvement | NTP PPL | Time |
|--------|-----------|----------|-------------|---------|------|
| medical | 1.397 | 1.069 | +23.5% | 3.46 | 110s |
| code | 1.277 | 0.974 | +23.7% | 3.14 | 111s |
| math | 0.887 | 0.601 | +32.2% | 2.38 | 123s |
| legal | 2.839 | 2.592 | +8.7% | 14.66 | 118s |
| finance | 2.871 | 2.702 | +5.9% | 14.01 | 122s |
| science | 1.904 | 1.604 | +15.7% | 7.30 | 105s |
| history | 2.663 | 2.410 | +9.5% | 9.29 | 104s |
| philosophy | 2.842 | 2.423 | +14.7% | 9.97 | 104s |
| creative_writing | 2.223 | 1.970 | +11.3% | 12.01 | 104s |
| cooking | 0.714 | 0.580 | +18.8% | 2.55 | 126s |
| health_fitness | 2.318 | 1.988 | +14.3% | 6.66 | 118s |
| psychology | 2.842 | 2.639 | +7.1% | 12.20 | 126s |
| education | 0.966 | 0.712 | +26.2% | 2.41 | 110s |
| engineering | 1.299 | 0.930 | +28.4% | 2.45 | 111s |
| agriculture | 2.106 | 1.822 | +13.5% | 8.35 | 103s |
| environmental | 2.085 | 1.802 | +13.6% | 6.62 | 102s |
| politics | 2.076 | 1.728 | +16.7% | 6.67 | 102s |
| economics | 2.168 | 1.762 | +18.7% | 8.93 | 103s |
| sociology | 1.023 | 0.833 | +18.6% | 3.61 | 128s |
| linguistics | 0.712 | 0.647 | +9.2% | 3.64 | 125s |
| cybersecurity | 0.708 | 0.628 | +11.4% | 3.09 | 123s |
| marketing | 0.767 | 0.624 | +18.6% | 2.95 | 122s |
| sports | 1.032 | 0.776 | +24.8% | 2.34 | 111s |
| music | 0.987 | 0.696 | +29.5% | 2.34 | 109s |

**Mean improvement: 17.3%** across all 24 domains.

### Loss Distribution

- **Top 5 improvement:** math (32.2%), music (29.5%), engineering (28.4%),
  education (26.2%), sports (24.8%)
- **Bottom 5 improvement:** finance (5.9%), psychology (7.1%), legal (8.7%),
  linguistics (9.2%), history (9.5%)
- Pattern: Domains with shorter, more structured responses (math, code, music)
  show larger improvements. Long-form domains (legal, finance, psychology)
  show smaller but still positive improvements.

### Timing

- Mean per-domain: 113s (1.9 min)
- Total: 45.4 min (well under 4-hour K751 limit)
- Memory: stable at ~5GB active, ~17GB peak throughout

## Kill Criteria Assessment

### K750: PPL improvement count >= 15/24

**PASS.** 24/24 domains improved (SFT val loss < base val loss).
This exceeds the 15/24 threshold with 100% success rate.

### K751: Training time < 4 hours

**PASS.** Total training time: 45.4 minutes. 5.3x under budget.

### K752: No domain diverges (convergence check)

**FAIL.** The automated check (`final_val_loss > 2 * initial_train_loss`) flagged 5
domains as diverged: medical, history, cooking, engineering, politics.

**Why the check is flawed:** The criterion compares `final_val_loss` (average over 25
validation examples) against `initial_train_loss` (loss on the FIRST training sample).
These are fundamentally different quantities. Example: cooking has initial_train_loss=0.11
(one very easy sample), but val loss improves from 0.71 to 0.58 — clearly convergent.

**Separate convergence assessment:** All 24 domains show val loss improvement over base
(no adapter). Zero domains have val loss worse than where they started. However, this
is K750's criterion, not K752. K752 as coded FAILS.

**Process note:** The correct fix would have been to implement K752 as
`final_val_loss > base_val_loss` (same distribution comparison) before running the
experiment. Retroactively redefining a failed kill criterion is a process violation.

## Limitations

1. **SFT loss is not comparable to NTP PPL.** The "Base Loss" and "SFT Loss" columns
   measure cross-entropy on response tokens only (SFT metric), while the NTP PPL
   column measures full next-token prediction perplexity. These metrics are not directly
   comparable. The SFT adapters may have different NTP PPL than reported here.

2. **Behavioral quality not measured.** Finding #187 showed SFT produces better judge
   scores than NTP (3.93 vs 3.72 at N=5). This experiment only measures convergence,
   not behavioral quality. Judge evaluation is needed for the N=24 composition proof.

3. **Single seed.** All training used seed=42. Multi-seed validation not performed
   (justified by Finding #54: multiseed CV=0.5% at N=5).

4. **Domain ordering matters.** The script was corrected to use the NTP experiment's
   domain ordering (medical=0, code=1, ...) to match the Grassmannian skeleton. A
   previous version used alphabetical ordering, which would have assigned wrong A-matrices.
   The orthogonality guarantee holds either way (any permutation of orthogonal frames is
   still orthogonal), but domain-consistent A-matrices are needed for fair SFT vs NTP
   comparison in composition experiments.

## What Would Kill This

- **At micro scale:** If N=24 SFT composition (downstream experiment exp_n24_composition_proof)
  shows routing degradation or composition quality worse than NTP composition, the SFT
  adapters are inferior for the SOLE architecture despite convergence.
- **At macro scale:** If behavioral evaluation (judge scores, task accuracy) shows SFT
  adapters at N=24 are not better than NTP adapters, the "training recipe > architecture"
  thesis is weakened.
- **Format dominance (Finding #216):** If SFT B-matrix inter-cosine is again near 0.97
  (as found at scale=2.0, 200 steps), the adapters may not be sufficiently differentiated
  for routing. This needs to be measured in the composition experiment.
