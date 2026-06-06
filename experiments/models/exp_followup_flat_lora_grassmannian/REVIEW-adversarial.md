# REVIEW-adversarial.md — exp_followup_flat_lora_grassmannian

## Verdict
**PROCEED (ratify preemptive KILL).**

## Adversarial checklist (a)–(t)

(a) **Consistency (results/DB/PAPER):** results.json verdict=KILLED,
preemptive=true, executed=false, all_pass=false, is_smoke=false.
PAPER verdict: "KILLED — preemptive, structural impossibility". DB
status will be `killed` after `experiment complete --status killed
--k 1554:fail`. CONSISTENT.

(b) **is_smoke:** false; preempt analytic, no run. PASS.

(c) **Verdict string:** "KILLED" exact match in both PAPER and results.json.

(d) **Adversarial label:** not SUPPORTED, not PROVISIONAL, not PARTIALLY.

(e) **KC drift:** K1554 text verbatim from DB (`With true-Grassmannian A
and 5/5 converged adapters, Flat-LoRA outperforms random-QR baseline by
>=3pp`). Pre-reg 2026-04-17 unchanged.

(f) **Tautology check:** K1554 compares Flat-LoRA-with-Grassmannian-A vs
random-QR/Kaiming baseline — separately measurable under the parent's
pipeline. Not tautological. The reason to preempt is that the parent
*already measured both the sharpness mechanism and the orthogonality
mechanism*; there is no remaining mechanism for the delta to exceed noise.

(g)–(m2) **Platform / code / eval checks:** N/A — no code executed.
`run_experiment.py` is a 14-line stub that `sys.exit(0)` after printing.

(n)–(q) **Evaluation / numerical bounds:** N/A — no eval.

(r) **Prediction-vs-measurement:** PAPER table shows predicted Δ_merge ≤
0.15pp vs "not measured (preempt)". Parent quantities (sharpness 0.02/0.07%,
cos 0.001) cited verbatim from F#35.

(s) **Proof soundness (∧-kill):**
- L1 (sharpness floor): F#35 data direct. Sound.
- L2 (A-init ⊥ sharpness): F#132/F#498 describe A-row / A-cluster
  properties, not Hessian. Flat-LoRA's SAM updates depend on gradient
  perturbation, not A-packing — uncoupled. Sound.
- L3 (cos-to-merge absent): cos(adapter) 0.001 already 50× below threshold;
  F#38 (killed) explicitly states Grassmannian orthogonality does not drive
  specialization. Sound.
- ∧-combination: Inverting any single lemma is insufficient; all three
  independently bound Δ_merge below noise. Compounded bound holds.
Minor caveat: "uncoupled" is a structural claim rather than an exact
derivation — but there is no published mechanism linking A-Grassmannian to
loss-sharpness-reduction, and the parent's empirical floor is the stronger
argument. Acceptable for preempt.

(t) **Target-vs-proxy:** K1554 IS the target quantity (merge-quality delta
in percentage points). Not a proxy. The preempt uses F#35's *direct* merge
measurements, not a proxy. PASS.

## New sub-axis registered
`grassmannian-A-init-uncoupled-from-loss-sharpness` — applies broadly to
any variant that proposes an A-init change as rescue for a sharpness-
or landscape-driven mechanism. The A-init property space and the
loss-landscape property space are structurally orthogonal.

## Reviewer ruling
KILL (ratify preempt). No new finding registered; pure F#35/F#132/F#498/F#38
reuse plus a structural orthogonality argument. Antipattern flags recorded
in results.json. DB `experiment complete --status killed --k 1554:fail`
authorized.
