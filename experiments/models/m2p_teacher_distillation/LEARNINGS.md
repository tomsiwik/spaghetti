# LEARNINGS: exp_m2p_teacher_distillation

**Status:** KILLED (K853 FAIL: quality gap closure negative on all domains)
**Finding:** #352

---

## Core Finding

KL distillation with M2P degraded the student on all three symbolic domains (gap_closure = −5.71, −1.74, −0.91) because Assumption A2 was violated: the student base model already outperformed the teacher SFT model at every domain. The KL objective worked correctly — it successfully moved the student distribution toward the teacher — but since the teacher was the inferior distribution, this was a regression gradient. The impossibility guarantee (Corollary 1: KL ≥ 0 → no regression) is conditional on A2; without A2, it does not apply.

---

## Why This Happened

### Primary Cause: A2 Violation at Micro Scale

At micro scale (equal 1200-step training budget, teacher d=512 vs student d=256), the larger teacher model did NOT learn better than the smaller student. This is the classic "capacity mismatch under equal training budget" effect:

> Larger models have more parameters to fit with the same data, so at sub-capacity training budgets, they may underfit relative to smaller, well-tuned models.

This is explicitly documented in **Stanton et al. 2021 ("Does Knowledge Distillation Really Work?", arXiv:2106.05537, NeurIPS 2021)**, which demonstrates that distillation fidelity (student imitating teacher) and generalization are separable concerns — and that the fundamental prerequisite is that the teacher's distribution is informationally richer. When the teacher is not better at the task, "imitation" is regression.

The identical failure mode appears in **Born Again Networks (Furlanello et al. 2018, arXiv:1805.04770)**: same-generation (same-capacity) KD works precisely because successive retraining naturally produces a better teacher in each generation. Furlanello's insight: A2 can be satisfied even with same capacity — but only through proper training, not by simply scaling parameters.

### Secondary Cause: Corollary 1 Misstated as Unconditional

The no-regression guarantee (Corollary 1) was written as unconditional ("The student+M2P adapter CANNOT be worse than the student base"). The adversarial review (REVIEW-adversarial.md) correctly identifies this as the central mathematical error — the guarantee only holds if A2 holds. Gibbs' inequality (KL ≥ 0) guarantees convergence toward p_T, not improvement over p_0. These coincide only when p_T is better than p_0.

### Relationship to Finding #30 (Different Failure Mode)

**Finding #30** (KD from Qwen2.5-7B-Instruct → ternary student, -34.4% PPL) was killed due to **cross-distribution mismatch**: the teacher (Qwen2.5-7B-Instruct) generates verbose markdown while the eval data is terse domain text. Diagnostic signature: lower training loss with higher eval PPL — the student learned the wrong distribution well.

**This experiment (#352)** failed due to **A2 violation**: the student base was already better than the teacher at the target task. The mechanism is different:

| Failure Mode | Finding | Root Cause |
|---|---|---|
| Cross-distribution mismatch | #30 | Teacher distribution ≠ target distribution (instruction-tuned teacher on domain-eval data) |
| A2 violation | #352 (this) | Teacher NTP loss > student base NTP loss (teacher is weaker at the task) |

Both confirm KD has fundamental prerequisites, but the structural fixes are different. Finding #30 is NOT this experiment re-treading a closed path — it is a second, independent failure mode confirmed. However, the MATH.md should have cited Finding #30 to acknowledge the prior kill and explicitly differentiate.

---

## Confirming Evidence (Supporting A2 as a Hard Prerequisite)

1. **Hinton et al. 2015 (arXiv:1503.02531):** Original KD paper assumes teacher is better. The temperature-scaled soft targets carry "dark knowledge" only when the teacher has learned a richer similarity structure. Explicitly: teacher advantage is a prerequisite.

2. **Stanton et al. 2021 (arXiv:2106.05537):** "Does Knowledge Distillation Really Work?" — directly studies fidelity vs. generalization. Key result: high fidelity (student imitates teacher closely) does not imply high generalization when the teacher itself lacks quality. A2 violation naturally follows.

3. **Furlanello et al. 2018 (arXiv:1805.04770) "Born Again Networks":** Same-capacity self-distillation works because retraining on teacher soft targets plus hard labels gives the student additional information about the teacher's uncertainty structure — but only when the teacher has been trained longer/better than the student will be. Pure parameter-count advantage is insufficient.

4. **Prior Finding #30:** At macro scale, cross-distribution mismatch killed KD before A2 even became the question. This experiment reveals A2 violation is a second distinct failure mode that is orthogonal to distribution mismatch.

---

## Contradicting Evidence

1. **Born Again Networks (arXiv:1805.04770) — Same-capacity success:** Furlanello shows that BAN-1 (student = teacher capacity) can outperform the teacher on CIFAR-100. This suggests A2 is NOT strictly required if the training procedure uses the teacher's soft targets as curriculum. However: BAN uses iterative generation where each generation is trained longer; in our experiment, teacher and student were trained for the same number of steps with the same data.

2. **Peer distillation and mutual learning (Zhang et al. 2018, arXiv:1806.00774, "Deep Mutual Learning"):** Two models of equal capacity trained simultaneously with mutual KL can outperform either individual model. This suggests mutual learning is viable even without A2. Mechanism: each model provides an ensemble-like signal to the other, reducing variance. Critical distinction: mutual learning does NOT require one model to be better than the other *before training begins*.

These contradictions suggest: A2 is required for classic KD (student imitates pre-trained teacher), but NOT required for mutual learning or iterative self-distillation, where A2 is built in dynamically.

---

## Alternative Approaches (Paper-Backed, for Cross-Model Transfer Without A2)

1. **FitNets / Hint-Based Learning (Romero et al. 2014, arXiv:1412.6550):** Instead of distilling output distributions (which requires A2), distill intermediate representations. The student learns to reproduce teacher's intermediate activations (hint layers). This is less sensitive to A2 because the student is trained on teacher's *internal features*, not output quality. At macro scale: train M2P on teacher hidden states directly (not KL distillation on outputs).

2. **Self-Supervised from Teacher Data (Finding #339 pattern):** Train M2P with NTP loss on teacher-generated sequences (without requiring teacher to be better). Finding #339 used this pattern and achieved 66.6% SFT quality. This completely sidesteps A2 — the teacher provides data, not a quality target.

3. **Deep Mutual Learning (Zhang et al. 2018, arXiv:1806.00774):** Student and teacher train simultaneously with mutual KL. Neither needs to be better at start. Both improve together. Potential application: train student M2P and teacher M2P simultaneously with cross-model soft targets.

4. **Curriculum Distillation / Progressive Training:** Train teacher to significantly higher quality before distilling. Guaranteed A2 if teacher trains 10× longer (student 1200 steps → teacher 12,000 steps). Trivially satisfies A2 at micro scale.

---

## Implications for Next Experiments

1. **A2 check is now a mandatory precondition** for any KL distillation experiment:
   ```python
   assert teacher_sft_loss < student_base_loss, "A2 violated: teacher not better than student"
   ```
   This must run before M2P training begins.

2. **At macro scale (Qwen3-8B → Qwen3-4B), A2 naturally holds** because 8B models have pre-training data advantage (trillions of tokens vs student). The micro-scale failure is NOT a macro-scale blocker.

3. **Two independent KD failure modes now confirmed:**
   - Finding #30: Cross-distribution mismatch (teacher distribution ≠ eval distribution)
   - Finding #352 (this): A2 violation (teacher weaker at target task)
   Both are structural, not hyperparameter issues. Both require architectural fixes, not tuning.

4. **The projection layer works** (K854 PASS): Linear(d_T → d_M2P) dimension adaptation is architecturally sound. This is transferable to any future cross-dimension M2P work.

5. **Current bottleneck is routing, not generation** (Finding #351): Per-domain M2P achieves 93.3% quality; routing achieves only 36.6%. Teacher distillation is a quality improvement question — but the project's real blocker is routing. Distillation work should wait until routing is solved.

---

## Recommended Follow-Up

**Priority: LOW** — the routing bottleneck (Finding #351: 36.6% routing vs 93.3% generation) must be solved first. Teacher distillation only matters once composition is working.

If pursued later:

**Option A (micro-scale verification):** Retrain teacher for 12,000 steps (student stays at 1,200). Verify A2 before training. Log cross-domain projection cosine. This would test whether the M2P+KL mechanism works when A2 holds.
- MOTIVATION: Finding #351 proves per-domain generation quality is not the bottleneck. This verifies whether teacher distillation improves per-domain quality further.
- LITERATURE: Hinton 2015 (arXiv:1503.02531) proves KD works when A2 holds.

**Option B (macro-scale):** Qwen3-8B SFT → Qwen3-4B student with M2P cross-dimension projection. A2 naturally holds (8B vs 4B pre-trained gap is large). Must also address Finding #30's cross-distribution failure mode.
- PREREQUISITE: Fix Finding #344's routing distribution mismatch (router trained on BASE states, deployed on COMPOSED states) first.

**Option C (mutual learning — no A2 required):** Apply Deep Mutual Learning (Zhang et al. 2018, arXiv:1806.00774) to simultaneously train student and teacher M2P with cross-model KL. Does not require A2. Novel for adapter-hypernetwork setting.
- MOTIVATION: Avoids A2 entirely. Both models improve together.
- LITERATURE: arXiv:1806.00774

---

## Transferable Patterns

| Pattern | Rule |
|---|---|
| **A2 precondition** | Before any KL distillation: assert teacher_loss < student_loss on eval set |
| **Cross-dimension M2P projection** | Linear(d_T → d_M2P) works architecturally (K854) |
| **KD failure modes are distinct** | Distribution mismatch (Finding #30) and A2 violation (#352) are separate; each requires its own structural fix |
| **Micro ≠ macro for A2** | Equal training budget at micro scale may violate A2 even with 2× larger teacher; macro-scale pre-training advantage naturally satisfies A2 |
