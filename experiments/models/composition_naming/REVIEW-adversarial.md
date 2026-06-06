# Peer Review: Composition Naming

## NotebookLM Findings

Skipped. This is a naming/terminology survey, not a mathematical or experimental contribution. NotebookLM deep review would add no value beyond what a close reading provides.

---

## Mathematical Soundness

### MATH.md

MATH.md is not really "math for this experiment" -- it is a standalone formal notation document for the entire SOLE architecture. It was written as part of the naming task but functions as a reference specification. Reviewing it on its own merits:

**What holds:**

1. The LoRA delta formula `dW_i = (alpha/r) * B_i @ A_i` is standard and correct.
2. The additive composition `W_composed = W_s + sum dW_i` is correctly stated, with the important caveat about nonlinearity cross-terms properly acknowledged (Section 2.3).
3. The capacity bound `N_max ~ d_out * d_in / r^2` is the standard Johnson-Lindenstrauss-style packing bound for near-orthogonal subspaces in high dimensions. Correctly applied.
4. The interference bound in Section 4.1 is dimensionally consistent.

**Issues found:**

1. **Section 3.2, cross-term bound is imprecise.** The inequality
   `|<dW_i @ x, dW_j @ x>| <= ||dW_i|| * ||dW_j|| * |cos(dW_i, dW_j)| * ||x||^2`
   is not tight and mixes weight-space cosine with output-space inner product. The correct Cauchy-Schwarz bound is `|<dW_i x, dW_j x>| <= ||dW_i x|| * ||dW_j x||`, and relating this to weight-space cosine requires additional assumptions about x (e.g., x drawn from a distribution that makes operator norms relate to Frobenius norms). The bound as stated is *directionally correct* but would not survive a rigorous appendix. This is a notation document, not a proof, so this is minor.

2. **Section 4.1, the loss decomposition is stated without derivation.** The term `interference(S, x)` and its bound `C * k^2 * (r/sqrt(d))^2 * max_i ||dW_i||^2` contain an unspecified constant C. Without derivation, this is an assertion, not a theorem. Acceptable for a notation reference but should not be cited as a proof in future papers.

3. **The worked example (Section 5) is internally consistent** and correctly computed.

### PAPER.md

PAPER.md contains no mathematical claims -- it is a literature survey and naming recommendation. No mathematical issues.

---

## Novelty Assessment

This is a naming task. The question is not "is the name novel?" but "is the name accurate, distinctive, and unoccupied?"

**Name availability:** No existing paper or system named "Structurally Orthogonal Latent Experts" or "SOLE" was found in the ML literature. No conflict in ML/AI.

**Accuracy of the name:**
- "Orthogonal" -- accurately reflects the structural property (cos ~ 0.0002 at d=896). Justified.
- "Additive" -- accurately reflects the composition mechanism (literal weight addition). Justified.
- "Experts" -- standard MoE term. Appropriate.

**Prior art coverage:** The survey covers the major relevant work: MoE (Shazeer, Switch, Mixtral, DeepSeek-V3), PEFT (LoRA, InfLoRA), merging (TIES, Model Soups), modular DL (Pfeiffer survey), BTM (Li et al.), LoRAHub (Huang et al.), Union of Experts (Yang et al.). This is reasonably comprehensive.

**Missing references worth noting:**
- **LoRA Soups (Ostapenko et al., 2024)** -- runtime LoRA composition by weighted averaging, which is closer to this work than Model Soups. The paper specifically studies composing independently trained LoRA adapters at inference time.
- **AdapterFusion (Pfeiffer et al., 2021)** -- learned attention-based composition of adapters. Relevant as a "learned routing" counterpoint.
- **Polytropon (Ponti et al., 2022)** -- task-level routing over adapter inventory. Another relevant comparison.
- **MELoRA / MoSLoRA** -- various multi-expert LoRA papers from 2024 that compose LoRA experts with routing.

None of these invalidate the name choice, but the positioning table (Section 5) would be strengthened by including at least LoRA Soups, since it is the most directly comparable architecture (independently trained LoRA adapters composed at runtime).

---

## Experimental Design

This is not an experiment. It is a terminology survey and naming recommendation. There is no hypothesis to test, no data to collect, no controls to evaluate. The HYPOTHESES.yml entry correctly marks it with "(no kill -- this is a framing/naming task, not an experiment)."

**Assessment:** The survey methodology is sound for what it is. The comparison table structure (Section 1) systematically categorizes existing terms by conflict level. The candidate evaluation (Section 3) applies consistent criteria. The glossary (Section 4) is well-structured and internally consistent with MATH.md.

**One concern:** The node status in HYPOTHESES.yml is "proven," which is the wrong status for a non-experimental task. A naming convention cannot be "proven" -- it can be "adopted" or "completed." This is cosmetic but worth noting for hypothesis graph hygiene.

---

## Hypothesis Graph Consistency

The experiment matches its HYPOTHESES.yml node. It has no kill criteria (appropriate for a naming task), no blockers, and no dependencies. The evidence listed in the node accurately reflects the content of PAPER.md.

The node does not block any other experiments, which is correct -- naming is orthogonal to experimental progress.

---

## Integration Risk

The naming hierarchy (Living Composable Model > SOLE > Skeleton/Expert/Library/Routing/Evolution) is clean and maps directly to the architecture described in VISION.md. No conflicts with existing components.

One minor integration concern: the MATH.md document conflates two roles -- it is both "formal notation for the naming experiment" and "the canonical mathematical specification of the architecture." If it is meant to serve as the latter, it should be promoted to a top-level document (e.g., `SOLE-SPEC.md`) rather than living inside a micro experiment directory.

---

## Macro-Scale Risks (advisory)

1. **"Additive" may need qualification at scale.** The paper acknowledges this (Section "What Would Kill This"): if composition at scale requires learned weights rather than unit weights, the "Additive" descriptor becomes misleading. The 50-expert pilot (`exp_distillation_pilot_50`) will provide the first real test.

2. **"Orthogonal" claim strengthens at scale** (as the paper notes), so this is not a risk.

3. **Community adoption risk.** The name only has value if it is used consistently. If future papers or code use inconsistent terminology (e.g., reverting to "LoRA experts" or "adapters"), the naming exercise was wasted. Not a technical risk but a practical one.

---

## Verdict

**PROCEED**

This is a well-executed terminology survey that does what it claims: systematically reviews the naming landscape and recommends a clear, accurate, unoccupied name. The glossary is internally consistent and maps cleanly to the architecture.

Minor issues that do not block PROCEED:

1. MATH.md Section 3.2 cross-term bound mixes weight-space and output-space quantities without sufficient justification. Should be tightened if cited in a future paper (not blocking here since this is a notation reference, not a proof).
2. Missing comparison to LoRA Soups (Ostapenko et al., 2024), the most directly comparable runtime LoRA composition work. Should be added to the positioning table.
3. HYPOTHESES.yml status should be "completed" rather than "proven" for a non-experimental task (cosmetic).
4. Consider promoting MATH.md to a top-level specification document if it is intended as the canonical SOLE notation reference.
