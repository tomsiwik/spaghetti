# REVIEW: exp_shine_gemma4_qa_benchmark (SHINE S4)

## Verdict: KILL

The experiment is well-executed. Kill is correct and well-justified.

## Checklist

| Check | Status |
|-------|--------|
| Prediction-vs-measurement table | PASS — 5 predictions, all measured |
| Kill criteria match evidence | PASS — K1261=0.006 (FAIL), K1262=0.029 (FAIL), K1263=0.133s (PASS) |
| Finding status appropriate | PASS — KILLED (2/3 kill criteria failed catastrophically) |
| Math errors | NONE — Thm 1 (info lower bound) and Thm 2 (CE⊥QA) are sound |
| Evidence fabrication | NONE — results.json matches all PAPER.md claims |

## Strengths

1. **Math predicted the outcome.** Theorems 1-2 predicted F1 < 10% from centroid
   trap (cos=0.988). Measured F1=0.6%. This is proof-first research done right.
2. **Impossibility structure is clear.** Root cause (no contrastive signal) is
   correctly identified. The fix (InfoNCE, arXiv:1807.03748) is grounded.
3. **Behavioral test killed a misleading metric.** CE ratio 0.0804 (92% reduction)
   looks phenomenal but produces gibberish. This finding alone justifies the experiment.

## Issues (non-blocking)

1. **P2 refutation is a confound, not a true refutation.** P2 predicted "factual F1
   ≈ ICL F1 (base knows facts)." Measured: no-adapter=0.002, ICL=0.196. But the
   no-adapter condition generates `<turn|>` tokens and Japanese — this is a prompt
   formatting issue with 4-bit Gemma, not evidence that the base model doesn't know
   Napoleon was born in Corsica. The PAPER correctly notes this but still marks P2
   as "REFUTED." More accurate: "P2 CONFOUNDED by prompt format."

2. **Doc count mismatch.** MATH.md says "10 documents with 3 questions each (30 QA
   pairs)." PAPER.md and results.json show 7 documents, 21 questions. Minor, but
   the experiment design drifted from the spec.

3. **ICL baseline is weak.** ICL F1=0.196 is low for extractive QA with document in
   prompt. Many ICL answers also produce `<turn|>` garbage. This means the "upper
   bound" comparison is itself noisy. However, this doesn't save SHINE — SHINE
   outputs are content-invariant gibberish ("4 BC, his adopted heir" for every
   document), which is categorically worse than a noisy but sometimes-correct ICL.

## SHINE Series Arc Assessment

Four stages. Two structural facts discovered:
- **Multi-projection (q+v+o) works:** 7.7x over q-only (S3). Keep this.
- **Centroid trap is persistent and fatal:** cos > 0.99 in every stage. Root cause:
  reconstruction loss has no contrastive signal; the global CE minimum IS the centroid.

The CE metric (86-92% reduction) was always a trap. S4 proved it behaviorally.
The SHINE architecture (M2P hypernetwork → LoRA) is sound in principle but requires
contrastive training to produce document-specific adapters. Without it, M2P is a
very expensive way to compute the average English LoRA.

## Recommendation

KILL confirmed. Record finding. The contrastive fix (InfoNCE + 100+ diverse docs)
is the natural next experiment if this line continues.
