# Peer Review: persistence_bridge_extraction

## Experiment Type
Guided exploration (Type 2) -- stated correctly. The proven framework is the Algebraic Stability Theorem + Eckart-Young-Mirsky. The unknowns are: (1) whether H1 features are actually destroyed, and (2) whether a low-rank bridge can restore them. The experiment narrowed both unknowns to definitive answers.

## Hack Detector
- Fix count: 1 (truncated SVD bridge). Clean, no stacking.
- Is MATH.md a proof or a description? Proof with QED (Theorem 1). The proof is a straightforward composition of two classical results.
- Metric used as evidence: bottleneck distance d_B (proven proxy for topological change) and PPL (behavioral proxy, explicitly flagged as weakest prediction).
- Kill criteria source: K628/K629 derived from the proof's predictions. K630 (PPL) is acknowledged as weakly grounded -- fair for guided exploration.

## Self-Test Audit
1. One-sentence impossibility property: States Eckart-Young optimality. This is about the bridge being optimal, not about what makes the failure mode impossible. Acceptable but slightly misframed -- the "impossibility" is really "if features are lost and SVD captures the damage directions, the correction is provably optimal." **PASS with caveat.**
2. Cited theorems: Cohen-Steiner et al. (2007) Thm 5.2, Eckart-Young-Mirsky (1936), Rips filtration (Rieck et al. 2018). All real, all applied correctly. **PASS.**
3. Predicted numbers: 4 specific predictions (P1-P4) with quantitative thresholds. **PASS.**
4. Falsification condition: Correctly identifies that the proof can be VACUOUS if A1 fails (no features actually lost). This is exactly what happened. Honest and well-framed. **PASS.**
5. Hyperparameter count: 1 (bridge rank k), derived from the proof. **PASS.**
6. Hack check: Single mechanism. **PASS.**

## Mathematical Soundness

**Theorem 1 is correct.** The proof is a two-step composition:
1. Bridge-corrected point cloud P'' has distortion delta'' = max_i ||(R - R_k)_i||_2 relative to original P.
2. Stability theorem gives d_B(Dgm(P), Dgm(P'')) <= delta''.
3. Eckart-Young gives optimality of R_k for minimizing ||R - R_k||_F.

One subtlety the MATH.md correctly notes (line 98-99): Eckart-Young minimizes the Frobenius norm ||R - R_k||_F, but the bound uses max_i ||(R - R_k)_i||_2 (max row norm). These are different -- Frobenius-optimal does NOT guarantee row-max-optimal. The bound is valid (it is whatever the row-max happens to be), but calling R_k "optimal" for the bottleneck bound is slightly overstated. The SVD minimizes total energy, not worst-row energy. A targeted correction hitting only the worst rows could achieve a tighter bottleneck bound at the same rank.

This is a minor theoretical imprecision, not a proof error. The bound itself holds regardless.

**Corollary 2 (Rank Budget)** is correctly stated as a necessary condition.

**Assumptions are well-enumerated.** A1 (features actually lost), A2 (SVD alignment), A3 (H1-quality correlation), A4 (subsample adequacy) are all genuine and the experiment tested A1 and A3.

## Prediction vs Measurement

The prediction-vs-measurement table is present and thorough.

| Prediction | Outcome | Assessment |
|---|---|---|
| P1: >= 10/35 modules lose H1 features | 0/35 | Falsified. A1 assumption violated. |
| P2: d_B reduction >= 50% at rank 16 | 47.9% mean, 52.2% median | Borderline fail on mean, pass on median. |
| P3: Bridge rank < 16 suffices | Rank 8 = 37.2%, rank 2 = 19.9% | Falsified. |
| P4: PPL improvement >= 5% | PPL worsens by 1.0% | Strongly falsified. |

The table is honest and correctly interpreted. P1 falsification is the key result -- it renders the entire approach moot. The PAPER.md correctly identifies that the proof is mathematically correct but the premise (that H1 features are destroyed) is empirically false.

## Findings Assessment

**Finding 1 (H1 survival):** This is the genuine discovery. The stability theorem's vulnerability window is conservative by nature (sufficient condition for destruction, not necessary). The experiment narrows the unknown: at rank-16 with 5 domains, the vulnerability window is empty in practice. This is valuable negative knowledge.

**Finding 2 (bridge hurts PPL):** This is the deeper insight. The perturbation's SVD components are not "damage" but "learning signal." Reversing them removes useful adaptation. This is a strong structural finding that closes the entire bridge-extraction approach.

**Finding 3 (perturbation not low-rank):** Effective rank ~21 for 5 rank-16 adapters. This means each adapter contributes ~4.2 unique dimensions on average, which is a useful characterization of adapter diversity. However, the claim "adapters are NOT operating in redundant subspaces" is slightly overstated -- 21 < 80 (theoretical max) means there IS significant redundancy, just not enough for a rank-16 bridge to capture 90%.

## NotebookLM Findings
Skipped -- the experiment is already killed with clear reasoning and the mathematical framework is sound. NotebookLM review would not change the verdict.

## Novelty Assessment

The combination of persistent homology stability theorem + Eckart-Young for bridge extraction appears novel as an approach. However, the experiment conclusively shows it is a non-problem at the tested scale. The negative result (composition is topologically lossless) is the genuinely novel finding.

Prior art check: The Garin & Tauzin (2020) and arXiv:2312.10702 references on PH for compression are relevant but address different problems (compression vs. composition). No prior work found that specifically tests topological preservation of LoRA composition.

## Macro-Scale Risks (advisory)

The kill is valid at current scale. At macro scale:
- More domains (>5), higher rank, or larger scale factors could push the vulnerability bound past actual H1 persistence. The scaling threshold is unknown.
- The fundamental insight (SVD of perturbation = learning signal) likely holds at any scale, making bridge extraction counterproductive regardless.
- If topological damage DOES occur at scale, a different approach is needed -- one that distinguishes "useful topological change" from "destructive topological change."

## Verdict

**KILL -- confirmed.**

The kill is well-justified on three independent grounds:

1. **The premise is false.** Zero H1 features are lost at current scale. The bridge solves a non-problem.
2. **The correction is counterproductive.** Even where d_B is reduced, PPL worsens. The perturbation's SVD components carry useful learning signal.
3. **The proof is correct but vacuous.** The theorem guarantees optimal restoration, but there is nothing to restore.

The experiment is a model of how to kill a hypothesis cleanly. MATH.md correctly identified A1 as the key assumption that could make the proof vacuous, and the experiment directly falsified A1. The prediction-vs-measurement table is complete and honest. The finding that "topological change from composition is useful adaptation, not damage" is a genuine structural insight that closes the entire pathway-preservation research line at current scale.

No revisions needed. The kill stands.
