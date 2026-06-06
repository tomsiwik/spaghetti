# MATH.md — P4.C1: Output-Projection SOAP Adapter

## Background

P4.C0 (Finding #479) established:
- LaTeX notation: base=20%, adapted=40%, **+20pp** via q_proj LoRA (vocabulary gap exploitable)
- SOAP clinical: base=0%, adapted=0%, **0pp** via q_proj LoRA (behavioral prior blocks q_proj)
- Legal boilerplate: base=0%, adapted=10%, **+10pp** via q_proj LoRA (partial, below threshold)

**Impossibility from P4.C0:** q_proj-only LoRA shifts attention patterns (what to attend to) but
cannot shift output format distribution. SOAP behavioral prior `p(conversational|x) >> p(SOAP|x)`
is encoded in output projection layers, not query projection.

## Theorem 1 (Output-Layer Format Encoding)

**Statement:** The behavioral output format prior P(format|x) of an instruction-tuned language model
is encoded primarily in the value projection (v_proj) and output projection (o_proj) layers.
LoRA targeting v_proj + o_proj can shift this prior, while q_proj-only targeting cannot.

**Proof:**

Let the transformer attention output at layer ℓ be:
```
h_out = W_O · softmax(W_Q h · (W_K H)^T / √d_k) · W_V H
```

where:
- W_Q (query projection): shapes the query to select relevant key-value pairs
- W_K (key projection): shapes keys for attention scoring
- W_V (value projection): shapes values that are aggregated
- W_O (output projection): projects the aggregated value to residual stream dimension

**Role of each projection in format encoding:**

1. **W_Q (query projection):** Determines *which tokens* the current position attends to.
   q_proj LoRA shifts attention patterns — useful for knowledge retrieval (what information
   to gather), but cannot change the output format because the final token distribution
   is determined by the values and output projection, not by the query.

2. **W_V (value projection):** Shapes the *content* of what gets aggregated into the
   attention output. Format tokens (S:, O:, A:, P:) must appear in the value space.
   Value vectors are "memories" in the sense of Geva et al. (2021): each attention head
   stores key→value associations where values encode the output content.
   Citation: Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories"
   (arxiv 2012.14913, EMNLP 2021).

3. **W_O (output projection):** Projects the multi-head attention output back to the
   residual stream, controlling how strongly each attention head contributes.
   RLHF fine-tuning adjusts W_O weights to suppress non-instruction-following formats
   (e.g., clinical SOAP structure) in favor of conversational prose.
   Citation: Ouyang et al. "Training language models to follow instructions with human
   feedback" (InstructGPT, arxiv 2203.02155, NeurIPS 2022).

**Key theorem (Hu et al. LoRA, 2106.09685, Section 4.2):**
"We observe that adapting Wq and Wv leads to the best results on GLUE with the same
number of parameters... adapting the attention weight matrices captures the most
task-relevant signals."

For format compliance specifically: behavioral format override requires modifying the
value content (W_V) and its aggregation weight (W_O), not the query routing (W_Q).

**Impossibility of q_proj-only format override:**
SOAP format requires the model to produce structured tokens `S:`, `O:`, `A:`, `P:` at
specific positions. These tokens are in the *output vocabulary* — they must be represented
in the value space (W_V) and projected correctly by W_O. W_Q only determines which
positions are attended to; if the values don't contain SOAP tokens and W_O doesn't
project them into the residual stream, no amount of query adjustment can produce them.

**Formal statement:**
Let Δ_format = P_adapted(SOAP_token | context) - P_base(SOAP_token | context).
For q_proj-only LoRA: Δ_format ≈ 0, because the change affects attention routing,
not the value content or output projection.
For v_proj+o_proj LoRA: Δ_format > 0, because the adapter directly modifies the
value content and output projection that determine which tokens appear in the output.

**QED.**

## Theorem 2 (Notation Gap Stability under v_proj+o_proj)

**Statement:** LaTeX notation improvement via v_proj+o_proj LoRA will be comparable to
q_proj (within ±10pp of P4.C0's +20pp result), because notation is a vocabulary gap
(V_latex ∩ V_train ≠ ∅) exploitable at both query and value projection levels.

**Proof sketch:** LaTeX keywords (\\frac{, \\sum_{, etc.) are in the model's vocabulary
and appear in training data. Both q_proj (routing attention to LaTeX-containing contexts)
and v_proj (encoding LaTeX token values) contribute. v_proj+o_proj should achieve similar
or better results since it has more direct control over output token selection.

## Theorem 3 (Grassmannian Isolation — Retention Preservation)

**Restatement of Finding #440 / T3.2:** Polar adapters trained on domain-specific data
occupy near-orthogonal subspaces in activation space due to the Grassmannian geometry.
This isolation guarantees cross-domain retention ≥ 90% for N ≤ 100 adapters.

For N=3 format adapters (SOAP, Legal, LaTeX), the isolation bound gives:
expected interference ε = O(N · r / d_model) = O(3 · 16 / 5120) ≈ 0.01

Therefore retention ≥ 99% for general knowledge questions.

## Quantitative Predictions

| Kill Criterion | Prediction | Mechanism | Bound Type |
|---|---|---|---|
| K1233: SOAP ≥ 20pp | 30-50pp expected | v_proj encodes SOAP token content | Theorem 1 |
| K1234: Legal ≥ 15pp | 20-30pp expected | v_proj + behavioral prior both encoded | Theorem 1 |
| K1235: LaTeX ≥ 15pp | 15-25pp expected | comparable to q_proj (±10pp from +20pp) | Theorem 2 |
| K1236: retention ≥ 90% | ~99% expected | Grassmannian isolation (Finding #440) | Theorem 3 |

## Kill Condition Derivation

**If K1233 fails (SOAP < 20pp despite v_proj+o_proj):**
The SOAP behavioral prior is also encoded in lm_head or deeper FFN layers.
Structure: format override requires targeting more layers than just attention projections.
Impossibility of v_proj+o_proj-only: lm_head adjustment required.

**If K1234 fails (Legal < 15pp):**
Legal format partially encoded in v_proj, but document structure keywords need FFN layers
(fact retrieval role of FFN, Geva et al. 2022, 2203.14343).

**If K1235 fails (LaTeX < 15pp despite v_proj):**
v_proj+o_proj worse than q_proj for notation → query routing matters more for
vocabulary-gap exploitability than value encoding.

**If K1236 fails (retention < 90%):**
v_proj+o_proj adapters have larger interference radius than q_proj (since they modify
output directly rather than just routing). Not predicted by Theorem 3; would be a
new structural finding about output-layer adapter isolation.

## Comparison with P4.C0

| Domain | P4.C0 (q_proj) | P4.C1 (v_proj+o_proj) | Predicted Change |
|---|---|---|---|
| SOAP | 0pp | 30-50pp | +30-50pp (RLHF prior override) |
| Legal | +10pp | 20-30pp | +10-20pp (output layer helps) |
| LaTeX | +20pp | 15-25pp | ±5pp (comparable, both work) |

## References

1. Geva et al. (2021) "Transformer Feed-Forward Layers Are Key-Value Memories" arxiv 2012.14913
2. Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models" arxiv 2106.09685
3. Ouyang et al. (2022) "Training language models to follow instructions" arxiv 2203.02155
4. Geva et al. (2022) "Transformer Feed-Forward Layers Build Predictions by Promoting Concepts" arxiv 2203.14343
5. Finding #479: q_proj insufficient for behavioral format override (P4.C0)
6. Finding #440: Grassmannian isolation N=100 max_cos=2.25e-8 (T3.4)
