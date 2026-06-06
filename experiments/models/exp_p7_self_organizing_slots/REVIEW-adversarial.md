# Adversarial Review: exp_p7_self_organizing_slots

## Verdict: PROCEED (as KILLED)

## Review

This experiment was correctly killed before implementation based on accumulated P7 findings. The logic chain is sound:

1. **Mathematical argument is valid.** null(W_v) ⊥ range(W_v) is a fundamental property of linear algebra. Domain adaptation ΔW = BA operates in range(W_v). Any function of null-space coordinates is orthogonal to domain features. The mutual information argument (I(f(A_i); domain) = 0) follows directly.

2. **Evidence chain is complete.** Three independent findings (#495, #498, #500) each attack a different aspect of null-space information content — routing accuracy, clustering structure, quality prediction — and all converge on the same conclusion: zero usable signal.

3. **Kill criteria assessment is correct.** K1309 (routing ≥ 85%) is killed by #495 showing 20% = chance. K1311 (auto slot-finding) is killed by #498 showing A-matrices are init-determined. K1310 (orthogonality) is correctly marked moot — Grassmannian QR is already the closed-form optimum.

4. **Impossibility structure is well-articulated.** The key insight — "you cannot learn information from a subspace orthogonal to where that information lives" — is precisely stated and properly grounded.

## P7 Line Closure Assessment

The six findings (#495–#501) form a complete impossibility proof for null-space-as-information-source:

| Finding | Claim Killed | Method |
|---|---|---|
| #495 | Null-space routing | Empirical: 20% accuracy |
| #496 | Null-space ≠ composition mechanism | Empirical: ensemble works without geometry |
| #497 | Null-space ≠ reasoning strategy | Empirical: prompting dominates |
| #498 | A-matrices carry domain info | Empirical: cluster by init, not domain |
| #500 | Null-space quality prediction | Empirical: AUC 0.43 |
| #501 | Self-organizing positions | Logical: orthogonality barrier |

**The P7 line is correctly closed.** Null space = isolation tool (Grassmannian QR packing for interference prevention). Not an information source.

## No Issues Found

No mathematical errors, unsupported claims, or fabricated evidence.
