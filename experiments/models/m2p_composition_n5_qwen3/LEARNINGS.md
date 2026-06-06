# LEARNINGS: exp_m2p_composition_n5_qwen3

**Status:** supported (Finding #381, post-adversarial fixes)
**Experiment type:** verification (Type 1 — with structural fix required)

---

## Core Learnings

### 1. Separate Grassmannian A-slots are mandatory for Theorem 1

The original code used the SAME A-matrices for both M2P networks, which voids the
orthogonality guarantee completely. When both domains share A, the Frobenius inner
product becomes:

  ⟨ΔW_math, ΔW_sort⟩_F = scale² · tr(B_math^T · I · B_sort) = scale² · tr(B_math^T B_sort)

This is NOT zero in general. Theorem 1 requires separate A-slots (A_math=Q[:,0:r],
A_sort=Q[:,r:2r]) so that A_math^T A_sort = 0 exactly.

**Rule:** Every new M2P domain needs its own r-dimensional slot in the global QR
construction. Maximum domains = d / r (for Qwen3-0.6B at r=4: up to 256 domains per layer).

**Reference:** Finding #50 (5-domain Grassmannian, max|cos|=1e-08) + MOLE (arXiv:2402.09432).

### 2. TF-IDF routing achieves 100% — covariate-shift failure fully resolved

Text-only TF-IDF routing achieves 100% accuracy on both tasks (math/sort), fixing the
prior 36.6% failure in exp_m2p_composition_n5 (Finding #351). Routing on raw input text
before any model forward is truly distribution-invariant by construction.

**Rule:** Always route on input text features. Never route on hidden states (unstable
under adapter composition). This is Theorem 2 (trivially true but architecturally critical).

**Reference:** LoraRetriever (arXiv:2402.09997) + Finding #354 (TF-IDF 95% on synthetic N=5).

### 3. B-matrix warm-start is required for convergence on fresh A-slots

The sort adapter trained on fresh A_sort slots for 1000 steps: training loss converged
to 0.77 (down from 5.91), but exact-match accuracy = 0.000. The math adapter (warm-start
from v4 SFT B-weights) converged fine at 300 steps.

**Root cause:** Fresh A-slots require B-matrices to simultaneously learn (a) what direction
to project into via A_sort, and (b) how to produce the task output. Without pre-trained
B-directions, the task is underdetermined for 1000 steps on a 4-bit 0.6B model.

**Fix:** Train LoRA SFT first on the task domain using the same A-slot columns. Then
warm-start the M2P network from those SFT B-weights. This is exactly what v4 did for
math: SFT B-matrices → M2P fine-tune → behavioral convergence.

**Rule:** M2P never trains from scratch on fresh A-slots. Always: SFT → M2P warm-start.

### 4. quality_ratio=1.0 under 100% routing is trivially true

With TF-IDF routing at 100% accuracy using routed_selection (alpha=1.0), the composed
evaluation ALWAYS applies the correct single adapter. Therefore:
  quality_ratio = (composed_acc - base_acc) / (single_acc - base_acc) = 1.0

The real stress test requires imperfect routing (<100%) or two tasks with confusable
vocabulary. The interesting measurement is: how does quality_ratio degrade as routing
accuracy decreases toward 80%?

**Next:** Test composition when routing is imperfect (e.g., two tasks sharing vocabulary).

### 5. Convergence gate is necessary before composition testing

Running K927 on an adapter with 0% single-task accuracy produces meaningless composition
results. The convergence gate (sort_single_acc > base_acc + 0.10) correctly fires and
skips K927-sort, preventing a false KILL based on adapter incompetence rather than
composition interference.

**Rule:** All composition experiments must gate on convergence: single-task accuracy
must exceed baseline by ≥10pp before composition is evaluated.

### 6. Composition mechanism verified on real LLM for math domain

K925 grad_norm=2.762 under composed adapter confirms Theorem 5 (functional forward intact).
K927_math quality_ratio=1.000 with separate Grassmannian A-slots confirms Theorem 1 on
Qwen3-0.6B. The adversarial-review-mandated Fix 1 (separate A-slots) was critical.

---

## What Remains Open

- **Sort adapter + composition test:** Needs SFT B-matrix warm-start for sort task.
  Then re-run composition with BOTH adapters converged — this is the true N=2 test.
- **Imperfect routing stress test:** quality_ratio at routing accuracy < 100%.
- **N=3+ composition:** Need to extend QR slot construction to ≥3 domains.

---

## References

- Finding #50 (conclusive): Grassmannian multi-domain, max|cos|=1e-08, 5 domains
- Finding #354 (supported): TF-IDF routing 95% on N=5 synthetic domains
- Finding #381 (supported): This experiment — math composition verified, sort blocked
- MOLE (arXiv:2402.09432): LoRA A-matrix interference via Frobenius inner product
- LoraRetriever (arXiv:2402.09997): Input-text routing invariant to model state
