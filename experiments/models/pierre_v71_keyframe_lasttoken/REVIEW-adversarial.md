# REVIEW — Pierre v7.1 Keyframe Last-Token Verifier (adversarial)

**Verdict: PROCEED-WITH-KILL**

## Consistency check (required before marking killed)

- results.json: `all_pass=false`, K#748 value=0.486 ≤ 0.60 threshold ⇒ FAIL.
- DB: kill_criteria[748].result=fail; kill_criteria[749].result=pass.
- PAPER.md verdict line: "KILLED (K#748 FAIL)".
- MATH.md theorem predicts single-class collapse under balanced BCE with
  near-label-independent features — matches measured pos_acc=0, neg_acc=1.
- total_time_s=72.2 (not smoke); N_TRAIN=2000, N_TEST=500 per MATH.
- KCs unchanged from pre-registration (2026-04-05).

All six consistent.

## 10-point adversarial checklist

(a) **Tautology**: K#748 measures classifier accuracy on held-out test set.
    Not tautological — a valid verifier would hit > 60% (e.g. by probing
    logits instead of h). Kill is earned.

(b) **KC-on-proxy**: K#748 is the target metric, not a proxy.

(c) **Code ↔ math**: `extract_last_token_hidden` at line 119-135 returns
    `h[:, -1, :]` — matches MATH definition of h_L[T−1]. Causal mask
    applied per layer. No leakage.

(d) **Composition bug**: Phase 5 is ghost-composition (both branches run
    identical domain-only injection — `run_experiment.py:370-372` vs
    `387-395`, no verifier injection in either). Flagged in PAPER as
    non-blocking because K#748 alone delivers the kill. No verifier-
    composition claim is made in the verdict.

(e) **LORA_SCALE = 20.0** at line 50: present on the composition code
    path only. Non-blocking because the composition measurement is
    already known tautological and the kill does not depend on it.

(f) **shutil.copy / hardcoded "pass"**: none present.

(g) **MLX discipline**: `mx.eval` after forward (line 134, 155, 264,
    298, 316, 356, 360, 424); `mx.clear_cache` in `cleanup` (line 74);
    `stop_gradient` on STE quantization (line 205); memory limit set
    (line 34-35). OK.

(h) **Balanced-label / BCE prior**: labels.mean() ≈ 0.5 by construction
    (line 102-110: 50/50 split via `rng.random() < 0.5`). Final loss
    0.6358 ≈ -log(0.5) = 0.693 confirms BCE converged to class-prior
    as predicted.

(i) **Single-class collapse detection**: pos_acc=0.0, neg_acc=1.0
    deterministic — matches P3. Tripwire `min(pos_acc, neg_acc) ≥ 20%`
    (proposed from v7 REVIEW) would correctly fire here.

(j) **Eval-template truncation**: MAX_SEQ=64 at line 52 — arithmetic
    expressions are ≤ 10 tokens, no truncation concern.

(k) **Class-prior sanity**: Phase 6 base_accuracy = 80% (8/10 correct)
    confirms signal exists in logits — consistent with MATH corollary
    that the miss is in the hidden-state → class-label mapping, not
    in the base model's arithmetic competence.

## Antipattern match summary

- Ghost-composition (F#157 family): present in Phase 5, flagged, non-blocking.
- Class-prior collapse (F#293 family): **primary failure mode** — verified
  by matched predictions P1-P4.
- No new antipattern classes.

## Reusable side-findings (analyst-owed when cap lifts)

1. Last-token hidden state is a *commit vector* in a causal LM, not a
   verifier substrate. Probing h_L[T−1] with BCE for "is x_T correct"
   is a structural dead-end independent of pooling choice.
2. Both mean-pool (v7) and last-token (v7.1) hidden-state probes fail
   identically under balanced BCE. F#293 generalizes: the failure is
   representational, not positional.

## Disposition

Kill is delivered by K#748 alone. Theorem and all six predictions are
verified by measurement. No REVISE blockers. Ghost-composition in
Phase 5 is a non-blocking antipattern flagged in PAPER for analyst
tracking.

PROCEED-WITH-KILL.
