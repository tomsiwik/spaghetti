# LEARNINGS: M2P Loss Normalization — V2 Rerun (KILLED, Finding #573)

## Core Finding

Per-domain loss normalization (`loss / base_loss[d]`) does NOT break M2P B-matrix
centroid collapse. K859 FAIL: `repeat_max_cos = 0.9979` (threshold 0.6; mean 0.9943).
K848 PASS (structural, |cos|=0.000000). K847 PASS (median 31.8% ≥ 25%) is a
median-robustness artefact — mean quality worsened to -67.6% (vs V1's -41.2%);
repeat-domain quality collapsed to -466% (vs -329%). MATH.md §F pre-registered
this FAIL and is confirmed, not falsified. No KC relaxation (K859 was added with
strict FAIL prediction per audit-2026-04-17). Verdict: **KILLED**, clean exit.

## Why

Loss normalization inflates the gradient weight of the lowest-loss domain
("repeat", base=1.1). Without domain conditioning in the M2P input, the model
still sees only mean-pooled hidden states which carry insufficient discriminative
signal on short synthetic sequences. Re-weighting the *magnitude* of per-domain
gradients cannot substitute for *conditioning on domain identity*. Three
mechanisms contribute to collapse (missing domain conditioning, heterogeneous
base losses, 1:1 M2P capacity); loss-norm addresses only the second — and
makes the first worse by amplifying the lowest-loss domain's pull.

## Implications for Next Experiment

1. Do NOT re-open `exp_m2p_loss_norm`. Question conclusively answered.
2. Sibling path: **domain conditioning** — concatenate learned `(N, D)` embedding
   to M2P input. Cheapest, most direct, MoE-gating precedent.
3. Alternative sibling: **per-domain M2P heads** gated by routing (more capacity,
   higher interpretability, larger cost).
4. Grassmannian A-slot theorem (K848) remains valid — carry it forward unchanged.
5. Do NOT promote loss-norm as a "helper" in composed experiments; its effect
   on repeat-domain quality was negative.

## Antipattern Note

No new `mem-antipattern-*` added. REVIEW adversarial checklist (a)-(s) clean —
no process bug (K859 was added with FAIL pre-registered, not post-hoc relaxed).
The centroid-collapse mechanism is already characterised in revision-1 §F and
this V2 rerun quantifies it more precisely; it is a structural finding, not a
recurring process antipattern.
