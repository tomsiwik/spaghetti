# MATH.md: NTP vs SFT Adapter OOD Benchmark

## Type: Guided Exploration (Type 2)

**Proven framework:** Adapter composition at per-domain optimal scales (Finding #249). SFT adapters degrade OOD benchmarks (Finding #260/261). NTP adapters showed +10pp GSM8K (Finding #237).

**Unknown:** Does the training objective (NTP vs SFT) determine whether composed adapters degrade OOD benchmarks? The proven framework has this free parameter: the loss function used during adapter training.

## A. Failure Mode Identification

**Observed failure:** SFT adapters at per-domain optimal scales (s=20 for math/code/medical, s=4 legal, s=1 finance) degrade ALL out-of-distribution benchmarks:
- GSM8K: 30% -> 15% (-15pp)
- Code generation: 90% -> 80% (-10pp)
- Clinical NER: F1 0.32 -> 0.24 (-7.4pp)
- MMLU: 44% -> 39% (-5pp)

**Root cause hypothesis:** SFT response-only masking trains the adapter to map instruction distributions to response distributions. The adapter learns: "when I see instruction format X, produce response format Y." On OOD inputs (different instruction format), the adapter applies format Y inappropriately, corrupting the base model's output.

NTP training sees ALL tokens (instruction + response). The adapter learns the joint distribution P(token_t | token_{<t}) across the entire text. On OOD inputs, the adapter's perturbation is proportional to how similar the OOD text is to any part of the training distribution (instruction OR response), not just the response conditional.

## B. The Right Question

**Wrong question:** "How do we prevent OOD degradation from adapter composition?"

**Right question:** "Under what conditions does an adapter perturbation Delta_W preserve the base model's behavior on inputs outside the adapter's training distribution?"

**Answer from perturbation theory:** The base model computes f_base(x) = W_base * x at each linear layer. The composed model computes f_composed(x) = (W_base + s * Delta_W) * x = f_base(x) + s * Delta_W * x. The perturbation term is s * Delta_W * x.

For OOD input x_ood, the perturbation magnitude is ||s * Delta_W * x_ood||. This is small when:
1. Scale s is small (but we need s=20 for in-distribution capability)
2. ||Delta_W|| is small (adapter norm)
3. Delta_W * x_ood is small — the adapter's weight perturbation has low response to OOD inputs

Condition 3 is the key. It depends on the alignment between Delta_W's row/column space and x_ood.

## C. Mathematical Framework

**Proposition (SFT vs NTP perturbation structure):**

Let D_train = {(x_inst, x_resp)} be instruction-response pairs.

**SFT objective:** L_SFT = E[-log P(x_resp | x_inst; W + Delta_W)]

The gradient of L_SFT w.r.t. Delta_W only flows through response tokens. Therefore Delta_W_SFT is optimized to transform hidden states at response positions. The column space of Delta_W_SFT aligns with {h_t : t in response positions}, which are conditioned on the instruction prefix.

**NTP objective:** L_NTP = E[-log P(x_t | x_{<t}; W + Delta_W)] for ALL positions t.

The gradient flows through all positions. Delta_W_NTP is optimized to improve prediction at BOTH instruction and response positions. Its column space spans {h_t : t in ALL positions}.

**Key difference:** SFT adapter's Delta_W is specialized to the response subspace. NTP adapter's Delta_W spans a broader subspace that includes instruction-like text patterns.

**Hypothesis 1.** Let x_ood be an OOD input. If x_ood has hidden state h_ood that is more similar to instruction-position hidden states than to response-position hidden states (true for most benchmark prompts which ARE instructions), then:

||Delta_W_NTP * h_ood|| / ||Delta_W_NTP|| <= ||Delta_W_SFT * h_ood|| / ||Delta_W_SFT||

because Delta_W_NTP has already been regularized by the instruction-position gradient to produce small perturbations on instruction-like inputs, while Delta_W_SFT has only been optimized for response positions and may produce arbitrary perturbations on instruction-like inputs.

**Note:** This is a hypothesis, not a proven theorem. The cited Gunasekar et al. (2017, arXiv:1705.09280) implicit regularization result applies to linear models with squared loss, not to multi-layer transformers with cross-entropy loss. The argument is plausible but the conditions of the cited theorem do not hold in our setting. The experiment tests this hypothesis empirically.

## D. Predictions

The proof framework predicts:

| Prediction | NTP composed | SFT composed | Basis |
|-----------|-------------|-------------|-------|
| P1: GSM8K accuracy vs base | <= 2pp degradation | -15pp (measured) | NTP adapter regularized on instruction-like text |
| P2: Code gen syntax vs base | <= 2pp degradation | -10pp (measured) | Same mechanism |
| P3: MMLU accuracy vs base | <= 3pp degradation | -5pp (measured) | Same mechanism |
| P4: In-distribution math quality | >= 60% correctness | ~70% (measured) | NTP may sacrifice some in-dist performance for OOD preservation |
| P5: In-distribution code quality | >= 40% pass@1 | ~80% (measured) | Same tradeoff |
| P6: NTP training convergence | loss < 2x SFT final loss | - | NTP on same data should converge (already confirmed by existing adapters) |

**Kill criteria mapping:**
- K1 (#678): P1, P2, P3 — if NTP adapters ALSO degrade >= 5pp on 3+ of ALL OOD benchmarks (gsm8k, code_gen, mmlu_medical, mmlu_code, mmlu_math, mmlu_legal, mmlu_finance) -> KILL
- K2 (#679): P4, P5 — if NTP adapters lose in-distribution quality -> KILL
- K3 (#680): P6 — if NTP adapters didn't converge -> KILL (already resolved: adapters exist)

## E. Assumptions & Breaking Conditions

1. **Assumption: Hidden states of OOD benchmark prompts resemble instruction positions more than response positions.** If violated (e.g., benchmark format is completely novel), both NTP and SFT adapters would degrade equally.

2. **Assumption: The existing NTP adapters from real_data_domain_experts were trained with enough steps to converge.** If undertrained, the comparison is unfair.

3. **Assumption: Per-domain optimal scales (Finding #249) transfer from SFT to NTP adapters.** NTP adapters may have different optimal scales. We use the same scales for controlled comparison.

4. **Assumption: The composition method (pre-merge with Grassmannian skeleton) works identically for NTP and SFT adapters.** The skeleton was computed once and shared.

## F. Worked Example (d=4, r=2)

Consider a 4-dim model with rank-2 adapters.

Base weight W_base = I_4 (identity for simplicity).

Training data: instruction = "What is 2+2?" response = "The answer is 4."

SFT adapter: Delta_W_SFT optimized only on response tokens. The hidden states at response positions cluster around h_resp = [0.1, 0.9, 0.1, 0.1] (response-specific). Delta_W_SFT's columns align with this direction.

NTP adapter: Delta_W_NTP optimized on all tokens. Hidden states at instruction positions: h_inst = [0.8, 0.1, 0.5, 0.2]. Delta_W_NTP must handle both h_inst and h_resp.

OOD input (GSM8K question): h_ood = [0.7, 0.2, 0.4, 0.3] — closer to h_inst.

Perturbation: s * Delta_W_SFT * h_ood applies a large perturbation because h_ood is not in the subspace Delta_W_SFT was optimized for. s * Delta_W_NTP * h_ood applies a smaller perturbation because h_ood resembles instruction states that Delta_W_NTP was trained to handle.

## G. Complexity & Architecture Connection

No additional FLOPs or parameters. Both NTP and SFT adapters have identical architecture (rank-16 LoRA on 7 target keys per layer, ternary B matrices with Grassmannian A). The only difference is the training objective.

Composition method: pre-merge with per-domain optimal scales. Same as prior experiments.

Runtime: evaluation only (no training needed — adapters exist). ~30-45 min for full benchmark suite.

## Self-Test (MANDATORY)

1. **What is the ONE mathematical property that makes the failure mode impossible?**
   Hypothesis 1 posits that NTP training regularizes the adapter to produce small perturbations on instruction-like inputs (which OOD benchmarks resemble). However, this was only confirmed for reasoning tasks (GSM8K), not universally. The hypothesis is partially supported.

2. **Which existing theorem(s) does the proof build on?**
   Implicit regularization of gradient descent toward minimum-norm solutions (Gunasekar et al., 2017, arXiv:1705.09280). Note: this theorem applies to linear models with squared loss, not transformers with cross-entropy. It provides motivation, not proof. The hypothesis remains empirically tested, not formally derived.

3. **What specific numbers does the proof predict?**
   NTP: GSM8K <= 2pp degradation, code gen <= 2pp degradation, MMLU <= 3pp degradation. In-dist: math >= 60%, code >= 40%.

4. **What would FALSIFY the proof?**
   The proof is wrong if NTP adapters degrade OOD benchmarks by >= 5pp on 3+ domains (same as SFT), which would mean the OOD degradation is caused by the composition mechanism itself, not the training objective.

5. **How many hyperparameters does this approach add?**
   Count: 0. The only variable is the training objective (NTP vs SFT), which is a binary choice, not a hyperparameter.

6. **Hack check: Am I adding fix #N to an existing stack?**
   No. This is a diagnostic experiment to identify which variable (training objective vs composition method) causes OOD degradation. No fixes are being added.
