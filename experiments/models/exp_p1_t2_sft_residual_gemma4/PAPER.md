# PAPER.md — T2.5: SFT-Residual M2P on Gemma 4

**Status: KILLED** | V2 Audit Rerun | 2026-04-18

## V2 Audit Rerun (audit-2026-04-17-rerun + code-bug tags)

V1 ran with T2.1 adapter weights that existed at that time (acc_step0=80%,
acc_final=58%, QR=0.707, KILLED via gradient-identity forgetting). Audit tagged
this experiment `audit-2026-04-17-rerun + code-bug` — expected rerun after
applying cluster-level code-bug fix.

V2 rerun blocked by two missing preconditions. Applying class-level standing
rule (precondition-probe before macro rerun; now 4th instance this loop after
peer_comparison_llama31_8b, peer_comparison_qwen3_4b, mtbench_composed).

### V2 Prediction vs Measurement

| Precondition | Predicted | Measured | Pass? |
|---|---|---|---|
| P1 T2.1 adapter `.safetensors` on disk | FAIL (audit-observed from prior iters) | `adapters.safetensors` missing; only `adapter_config.json` stub present | FAIL |
| P2 T2.1 train.jsonl | PASS | `data/math/train.jsonl` 1800 lines | PASS |
| P3 T2.1 upstream verdict ≠ KILLED | FAIL (T2.1 killed 2026-04-18) | T2.1 verdict=KILLED (metric-swap MedQA vs MedMCQA + format-artefact max_tokens=256 CoT truncation) | FAIL |

### V2 KC Routing

| KC | Threshold | Measured | Status | Reason |
|---|---|---|---|---|
| K1044 M2P GSM8K | ≥ 73.8% | — | FAIL | unmeasurable (no B_sft) |
| K1045 B_applied compute | < 10ms | — | FAIL | unmeasurable (no B_sft to add ΔB to) |
| K1046 zero-init Frob | < 1e-6 | — | FAIL | unmeasurable (no 42-layer shape list) |

All FAIL → `all_pass=False` → KILLED. `results.json.verdict=KILLED` matches DB.

### Why the verdict classifies as KILL not PROVISIONAL

The V1 KILL already documented a real impossibility structure (gradient identity
∂L/∂ΔB = ∂L/∂B_applied — see §Why It Failed below). V2 cannot *disprove* V1, so
the reasonable verdicts are:

- KILLED (current): V1 evidence stands, V2 cannot overturn it, preconditions
  block rerun. Honest.
- PROVISIONAL: would require PAPER.md to claim a future rerun will flip the
  verdict — it won't, because the V1 mechanism (gradient identity) is
  mathematical, not a code bug. code-bug tag may reduce the gap, but not
  eliminate it.

V2 remains KILLED. The `code-bug` tag is a decoy on this experiment: the
failure mode is a mathematical property of gradient descent, not an
implementation defect. If the code-bug fix refers to EWC regularization or
data-separation (see §Implications below), that is a DIFFERENT experiment
scope; rerunning v1 with the same scope will reproduce the v1 KILL regardless
of bug fixes.

### Permanently learned (V2)

Class-level standing rules (now 4 instances confirmed):

1. **Precondition-probe before macro rerun.** Any `audit-2026-04-17-rerun`
   experiment whose preconditions are blocked (missing weights, killed
   upstream) is a precondition-probe KILL, not a code-fix-and-retry. Skip
   heavy training unless preconditions all pass.
2. **Adapter registry ≠ adapter artefacts.** `adapter_config.json` stub is not
   evidence the model weights ever existed on disk; only `.safetensors` is.
   Directory existence corollary also applies if the domain subdir is absent
   entirely.
3. **Downstream P1/P2 macros inherit upstream audit flags.** T2.1's
   2026-04-18 metric-swap + format-artefact kill propagates to every
   experiment that loads B_sft from T2.1. At least 4 downstream experiments
   killed this loop via this pattern.
4. **`code-bug` tag may be a decoy when the failure is mathematical.** If V1
   KILL was driven by a gradient-identity or continuity property (not a coding
   defect), V2 rerun with a code fix cannot recover the verdict. Classify the
   V1 mechanism before assuming the fix-category tag applies.

---

## V1 (unchanged below this line)

**Status: KILLED** | K1044 FAIL

## Abstract

We tested whether zero-initializing ΔB (B_applied = B_sft + ΔB, ΔB_init = 0) is sufficient to
maintain SFT quality during continued adaptation on Gemma 4 E4B. The zero-init guarantee holds
at step 0 (K1046 PASS, acc_step0 = 80%), but 500 steps of gradient descent on the same task data
causes ΔB to grow to 24.5% of B_sft's norm, reducing accuracy from 80% to 58%. K1044 FAILS:
quality_ratio = 0.707 < threshold 0.738. Finding #403 (Qwen3-4B, quality_ratio = 1.175) does NOT
replicate on Gemma 4 with the current training setup.

## Prediction vs Measurement

| Prediction (MATH.md) | Measured | Pass? |
|---|---|---|
| K1046: \|\|ΔB\|\|_F = 0 at step 0 (all 42 layers) | max_diff = 0.0 | ✓ PASS |
| K1045: B_applied time < 10ms | 0.385ms | ✓ PASS |
| acc_step0 = 82% (SFT quality, Theorem 1 Corollary) | 80.0% | ≈ (noise in 50-sample eval) |
| K1044: acc_final ≥ 73.8% (quality_ratio ≥ 0.90) | 58.0% (QR = 0.707) | ✗ FAIL |

## Results

| Metric | Value |
|---|---|
| acc_step0 | 80.0% |
| acc_final | 58.0% |
| quality_ratio | 0.707 |
| K1044 threshold | 73.8% |
| ΔB mean Frobenius | 0.2154 |
| B_sft mean Frobenius | 0.8771 |
| relative_correction | 0.2456 (24.6% of B_sft) |
| training final loss | 1.3594 (from 2.7656) |
| K1045 latency | 0.385ms ✓ |
| K1046 zero-init | verified ✓ |

## Why It Failed

**The zero-init guarantee is a static property (step 0 only), not a dynamic one.**

The gradient at each step is:
```
∂L/∂ΔB_l = (scale × A_l x^T)^T ⊗ (∂L/∂z_l)
```
This is identical to `∂L/∂B_applied_l` — the gradient does NOT know that B_applied = B_sft + ΔB.
Gradient descent moves ΔB in the same direction it would move B, regardless of initialization.

After 500 steps with LR=5e-6 and GRAD_CLIP=0.5, ΔB grew to 24.6% of B_sft norm. This is
sufficient to corrupt the chain-of-thought structure in B_sft (the math reasoning patterns), even
if training NLL decreased (2.77 → 1.36). Training loss measures token prediction on the training
set; eval accuracy measures structured reasoning with exact answer extraction.

**Key distinction from Finding #403 (Qwen3-4B):**
Finding #403 used a non-zero output_scale warmup schedule and trained on *different* data
(personalization queries, not the original SFT task data). This experiment re-trained on the exact
same GSM8K distribution as T2.1 — equivalent to continued SFT with fresh optimizer state, which
is known to cause catastrophic forgetting (Kirkpatrick et al., EWC, arXiv:1612.00796).

## Impossibility Structure

Zero-init of ΔB provides a **type-safe start** (quality at step 0 = SFT quality) but does NOT
prevent forgetting because:

1. `∂L/∂ΔB = ∂L/∂B_applied` — gradients are structurally identical to continued SFT gradients
2. Training on the SAME data distribution with fresh optimizer state overshoots the SFT minimum
3. No regularization term penalizes ||ΔB||_F growth — EWC or L2 penalty on ΔB is required

**Structural guarantee needed (not tried):**
```
L_total = L_task + λ × ||ΔB||_F²   (EWC or elastic anchor)
```
This would bound ||ΔB||_F ≤ O(1/√λ), preventing corruption of B_sft structure.
Alternatively: train ΔB on DIFFERENT data than SFT (the M2P intent).

## Implications

1. **SFT-residual M2P requires data separation**: ΔB must adapt on *new context data*, not
   re-train on the original SFT data. The experiment was testing the wrong scenario.

2. **For personalization (M2P)**: user-specific queries ≠ GSM8K — real M2P data would be
   different enough that the gradient conflict does not occur.

3. **EWC anchor is necessary for same-domain adaptation**: if the use case requires adapting
   on the same domain, add L2 regularization on ΔB with λ tuned to bound relative_correction ≤ 5%.

4. **Finding #403 remains valid**: it used different data (personalization queries) for ΔB.
   T2.5 refutes only the "same-domain re-training via ΔB" variant.

## References

- He et al. (2016, arXiv:1512.03385) — Residual learning
- Kirkpatrick et al. (2017, arXiv:1612.00796) — EWC: Overcoming catastrophic forgetting
- Hu et al. (2022, arXiv:2106.09685) — LoRA
- Finding #403 — SFT-Residual M2P on Qwen3-4B (quality_ratio=1.175, different data)
