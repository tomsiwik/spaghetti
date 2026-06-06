# PAPER.md — exp_memento_kv_serialization_format

**Verdict: KILLED (preempt-structural, F#666-pure-standalone — 11th drain-window instance)**

## Summary

`exp_memento_kv_serialization_format` is preempt-killed before execution because both
kill criteria (K1860 round-trip latency > 100ms, K1861 serialized size > 5MB) are
pure infrastructure measurements without a paired behavioral target KC. Under canonical
guardrail #1007 / Finding #666, an experiment whose entire KC set is proxy-only cannot
produce an admissible KILL or SUPPORTED verdict regardless of measurement — PASS is
tautological support for arbitrary thresholds and FAIL is a finding about the thresholds,
not about the behavioral feasibility of cross-session persistence.

Secondary fire: hygiene-multi-defect (3 defects — `success_criteria=[]`, `platform=~`,
`references=[]`) — canonical F#703 threshold. F#702 hygiene-patch PROVISIONAL path is
structurally unavailable because it requires ≥ 1 target KC to patch; 0 target KCs here.

Double-fire (F#666-pure primary + hygiene secondary; no §5), 2nd double-fire precedent
after F#703 and 1st since F#714 triple-fire.

Novel: K1860 + K1861 are the 1st **infrastructure-benchmark bucket** instance in the
drain window (6th bucket after derived-geometric, detection/classification, routing, PPL,
content-based similarity).

## Prediction vs. measurement

| # | Prediction (derived from KC topology + antipattern canon) | Measurement |
|---|---|---|
| P1 | Both KCs measure infrastructure properties (latency ms, size bytes) with no causal instrumentation linking them to cross-session persistence behavioral outcomes | not_measured (structural) |
| P2 | Thresholds (100ms, 5MB) are not derivable from a behavioral benefit/cost curve; they are operator intuition lacking calibration data | not_measured (structural) |
| P3 | PASS on both KCs yields tautological-support (thresholds define pass); FAIL yields finding-about-thresholds (not about persistence feasibility) | not_measured (structural) |
| P4 | Standalone topology (depends_on=[]); not F#669, not template-regression, not proxy-only-lineage-inheritance, not cross-paper-combined-loss-tautology | confirmed by DB record |
| P5 | Hygiene defect count ≥ F#703 threshold; F#702 hygiene-patch unavailable because 0 target KCs exist to patch | confirmed by DB record |

## Unblock path (v2)

A behaviorally-grounded v2 would register:

- **K_target**: multi-turn conversation recall accuracy with persistence vs no-persistence baseline on a held-out dialogue benchmark (e.g. MT-Bench multi-turn subset), 2048-token context boundary.
- **K_proxy_size** (paired): serialized size per 2048 tokens, threshold CALIBRATED from accuracy/size Pareto curve under K_target.
- **K_proxy_latency** (paired): round-trip latency, threshold CALIBRATED similarly.

Re-claimable when target+proxy pairing specified AND `success_criteria`/`platform`/`references` populated.

## Assumptions

- Classified as F#666-pure-standalone primary + hygiene-multi-defect secondary per F#714
  pre-claim-checklist hierarchy (KC class > KC form > metadata). With 0 target KCs,
  F#666-pure dominates.
- Latency/size are proxy (not target): notes frame the benchmark as "prerequisite for
  cross-session persistence"; persistence is the behavioral outcome, latency/size are
  engineering proxies. Alternative interpretation (pure infra benchmark where latency/size
  ARE target) rejected — thresholds behaviorally uncalibrated, notes explicitly behavioral.
- `depends_on=[]` is authoritative. Sibling `exp_memento_compression_ratio_benchmark`
  (F#699) is on a different axis (compression ratio, not serialization format); no parent.

## References

- Finding #666 (proxy-only target-gating)
- Canonical guardrail #1007 (target-gated KILL)
- Finding #703 (hygiene-multi-defect 3+ defect threshold)
- Finding #702 (hygiene-patch requires ≥ 1 target KC)
- Finding #714 (multi-pattern fire hierarchy; F#702 unavailability under 0 target KCs)
- Finding #711 (bucket taxonomy curatorial, not gating)
- MEMENTO (Kontonis arxiv:2604.09852) — research-line context, no mechanism under test
