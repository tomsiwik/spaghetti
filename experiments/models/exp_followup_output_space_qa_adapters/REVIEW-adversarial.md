# REVIEW-adversarial.md — exp_followup_output_space_qa_adapters

**Verdict:** PROCEED-WITH-KILL (preemptive).
**Rationale:** K1552 is either tautological, prerequisite-gate-unmet, or
base-beat-impossible. Running the KC as-written cannot advance the thesis.

## Adversarial checklist (a)–(t)

- (a) **results.json verdict consistency.** KILLED ↔ DB killed target ↔ PAPER
  "KILLED — preemptive" ↔ preemptive=true, executed=false. MATCH.
- (b) **all_pass field.** false ✓ (K1552 fail).
- (c) **PAPER verdict line.** "KILLED — preemptive, structurally uninformative KC"
  (no PROVISIONAL / PARTIAL / INCONCLUSIVE upgrade).
- (d) **is_smoke.** false — this is not a smoke-mode result. Preempt is a
  structural verdict, not a subsampled run.
- (e) **KC stability (pre-reg diff).** K1552 pre-registered 2026-04-17, unchanged.
  No MATH.md git diff showing KC drift.
- (f) **KC is real quantity, not tautology.** K1552 text IS tautological with
  respect to the claimed thesis (L1). This is the reason for preempt, not a
  silent substitution — PAPER.md and results.json both flag it explicitly.
- (g) **code measures what DB describes.** N/A — no code executed. Stub
  run_experiment.py exits 0 and writes a marker only.
- (h)–(m) MLX idioms / composition-bug / LORA_SCALE / tautological-routing /
  projection-scope. N/A — no code executed.
- (m2) skill-invocation. N/A — no platform code to review. (A v2 would require
  invoking `/mlx-dev` and `/fast-mlx` before writing Falcon-E-3B inference code.)
- (n) single-seed-over-interpretation. N/A — no measurements.
- (o)–(q) N/A — no eval run, no smoke-reported-as-full, no proxy substitution.
- (r) **prediction-vs-measurement table.** PAPER.md lines 9–15, explicit with
  "not measured / preempt" cells.
- (s) **mathematical soundness.** Three-lemma proof in MATH.md:
  - L1 uses F#165's measured −24% MMLU-on-Falcon and the construction of NTP
    output distribution. Sound.
  - L2 uses F#166's verbatim impossibility-structure statement. Sound.
  - L3 uses F#477's measured 2/5 base-beat rate on Gemma 4 and the weaker-prior
    assumption for Falcon-E-3B. The assumption is flagged in PAPER Assumptions;
    if anything, Falcon-E-3B's ternary quantization could weaken priors further,
    strengthening the preempt; but even inverting that assumption only flips
    L3, not L1 or L2, so the ∨-combined kill still holds.
- (t) **target-gated metric.** K1552 is the target quantity but it is not
  target-gated with respect to the thesis — it compares two adapter variants
  without a base-beat gate. This is the core pathology the preempt identifies.
  Preempt is correct; re-grading of the KC is documented.

## Family reuse vs new antipattern

F#166 (prerequisite-gate) and F#165 (OS-top2-adapters-degrade-base) family reuse.
**New sub-axis registered in results.json antipatterns:**
`tautological-inter-adapter-delta-ignores-base-baseline` — a format-of-KC
pathology distinct from the classical tautological-routing (where the router
uses its own output as the evidence). Here the KC is tautological because the
compared-against straw variant (NTP-format adapter) cannot, by construction,
produce the output format K1552 scores.

Analyst may register this as a new tripwire: **"inter-variant delta KCs that
lack an anchor vs base are structurally uninformative when one variant is
format-incompatible by construction."**

## No-run justification

A valid v2 requires re-pre-registering (i) base-beat gate, (ii) anchored KC
(composition vs base, not composition vs straw). The current experiment would
need to be redesigned, not re-run. Preempt-kill is the correct verdict.
