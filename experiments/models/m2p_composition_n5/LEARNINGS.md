# LEARNINGS: exp_m2p_composition_n5

**Status:** KILLED (K852 FAIL: routing 36.6% < 50%)
**Finding:** #344

---

## What Was Proven

### ✓ Theorem 1: Grassmannian Parameter Orthogonality (CONCLUSIVE)

QR decomposition produces A-matrices with orthonormal columns (Q^T Q = I), which
guarantees ⟨Δ_i, Δ_j⟩_F = 0 for all i≠j regardless of B-matrix content.

- Measured: max|cos| = 1e-08 (float32 zero)
- This is a mathematical identity, not an empirical result
- Holds for ANY B_i, B_j — independent of training method, scale, or domain

**This result is transferable to macro scale without new experiments.**

### ✓ Ensemble Effect: Orthogonal Adapters Improve General Quality

When routing is inaccurate but adapters are parameter-orthogonal, the soft routing
weights average across adapters. This produces an ensemble effect that improves
general (mixed-domain) quality: -14.4pp degradation means 14.4pp improvement.

- Base general loss: 9.52 → Composition loss: 8.15
- This is a positive side-effect of parameter orthogonality + inaccurate routing
- Interpretation: N orthogonal adapters act as a regularizer on the base model

### ✓ Independent Per-Domain M2P Works

Training each M2P independently on a single domain avoids the multi-domain bottleneck
(#341, #342, #343). Median M2P quality: 93.3% of SFT. This confirms Finding #339.

---

## What Failed

### ✗ K852: Routing Accuracy 36.6% (KILL)

**Claimed root cause in PAPER.md:** Router too weak, domain signals too subtle.

**Actual root cause (adversarial review):** Router train/test distribution mismatch.

The router was:
- **Trained on:** base model last-layer hidden states (no LoRA corrections applied)
- **Deployed on:** composed model intermediate hidden states (every layer has LoRA corrections)
- **Applied at:** every layer including layer 0 (embeddings), despite training on layer-1 output

This means **Theorem 2 was never tested.** The experiment measured a harder claim:
"Can a router trained on base model representations generalize to composed model
representations?" The answer is: not reliably at 36.6%.

Finding #310 (98.3% linear separability) remains valid — it measured separability on
a static single model, not a composition. The separability may still exist in the
composed model's representations, but the router was never given the right input.

### ✗ Per-Token Routing: Fundamental Vocabulary Ambiguity

Three of five domains (sort, reverse, repeat) share identical character vocabulary.
A single token "a" is ambiguous across these three domains. Per-token routing requires
per-token disambiguation — impossible without sequence context.

This is not a router capacity issue. It is a fundamental information-theoretic problem:
the router input (one token's hidden state) contains insufficient information to
identify the domain when vocabulary overlaps.

---

## What To Do Next

### Path A: Fix Router Distribution Mismatch

Train the router INSIDE the composition loop, on the composed model's actual hidden
states. This tests Theorem 2 correctly. Prediction: routing accuracy should approach
Finding #310's 98.3% baseline (after the fix).

### Path B: Sequence-Level Routing

Replace per-token MLP router with a sequence-level router (reads prefix or full context).
This addresses the vocabulary ambiguity problem. A lightweight attention-pool over the
sequence followed by MLP would capture domain context.

### Path C: Routing-Free Composition

Since parameter-orthogonal adapters already improve general quality via ensemble effect,
ask: is explicit routing necessary? An oracle routing (ground-truth domain labels at test
time) would tell us the ceiling. If oracle routing gives <20% improvement, routing is
not worth pursuing.

---

## Transferable Numbers

| Result | Value | Transferable? |
|--------|-------|---------------|
| Grassmannian max|cos| | 1e-08 | Yes — guaranteed by QR |
| Ensemble improvement | +14.4pp general quality | Likely — depends on adapter orthogonality |
| Per-domain M2P quality | 93.3% median of SFT | Yes — matches Finding #339 |
| Router train/test mismatch → failure | 36.6% | Warning for all future router experiments |

---

## Key Theorem Revision

**Theorem 2 as stated is incorrect.** It assumes the router has access to domain-labeled
data from the COMPOSED MODEL's representations. The proof cites Finding #310 (base model
separability) but applies it to the composed model — a hidden distribution shift.

Corrected Theorem 2: The router must be trained on representations from the SAME MODEL
it will be deployed on (i.e., the composed model at inference time). Base model
separability does not guarantee composed model separability.
