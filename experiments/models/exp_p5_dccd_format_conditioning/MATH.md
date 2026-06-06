# P5.A1: DCCD Format Conditioning — Mathematical Proof

## Reference: arXiv:2603.03305 (Draft-Conditioned Constrained Decoding)

## Motivation: Finding #483 (Cross-Projection Composition Kills Quality)

Finding #483 proved that composing q_proj domain adapters with v_proj+o_proj format
adapters via weight addition causes catastrophic model collapse. The q->o functional
chain creates activation-space interference despite zero parameter overlap.

**Root cause:** Weight-space composition changes the attention-to-output pipeline
(Q determines what to attend to, O determines how to project the result). Changing
both simultaneously with independently-trained adapters creates an inconsistent
pipeline — the O-projection expects activations from the BASE attention pattern,
but receives domain-ADAPTED attention output.

**DCCD solution:** Don't compose at weight level. Instead:
1. Domain adapter generates semantically correct but unformatted draft
2. Format constraint is applied at DECODING level, not WEIGHT level
3. The two concerns (content vs format) are separated in time, not in parameters

## Theorem 1: Draft-Conditioned Decoding Eliminates Projection Tax

**Setup.** Standard constrained decoding masks invalid tokens at each step:

  P_constrained(y_t | y_{<t}) = P_base(y_t | y_{<t}) * 1[y_t in V_valid(y_{<t})] / Z_t

where V_valid is the set of valid next tokens given the grammar state, and Z_t
is a normalization constant.

**Problem (Projection Tax).** The normalization Z_t != 1 in general. Over T steps:

  KL(P_constrained || P_base) = sum_{t=1}^T E[-log Z_t]

This cumulative KL divergence degrades output quality — the model is forced away
from its natural distribution at every step where invalid tokens have high probability.

**DCCD fix.** Generate an unconstrained draft D = (d_1, ..., d_T) first, then:

  P_dccd(y_t | y_{<t}, D) = P_base(y_t | y_{<t}) * 1[y_t in V_valid(y_{<t})]
                              * w(y_t, d_t) / Z'_t

where w(y_t, d_t) is a draft-conditioning weight that biases toward the draft's
semantic content while respecting format constraints.

**Guarantee.** The draft D provides a "semantic anchor" — the constrained decoder
stays close to the draft's meaning while enforcing structural validity. The
projection tax is amortized across the draft, not accumulated per-token.

**Proof sketch.** Let S_valid be the set of valid complete sequences under the
grammar. For any draft D:

  P_dccd(Y in S_valid) = 1  (all outputs satisfy format by construction)
  E[sim(Y, D)] >= E[sim(Y_constrained, D)]  (draft conditioning preserves semantics)

The first property holds because the grammar mask is applied at every step.
The second holds because the draft-conditioning weight w biases toward draft
tokens within the valid set, whereas unconditioned constrained decoding has no
semantic guidance.  QED.

## Theorem 2: Domain + Format Separation Prevents #483 Failure

**Statement.** Under DCCD, the domain adapter and format constraint operate on
DIFFERENT computation phases:

  Phase 1 (domain): y_draft = generate(model + domain_adapter, prompt)
  Phase 2 (format): y_final = constrained_decode(model, y_draft, grammar)

**The domain adapter is REMOVED during Phase 2.** The base model's Q and O
projections are unmodified during constrained decoding. There is no q→o chain
inconsistency because only ONE adapter is active at any time.

**Proof.** In Phase 1, the model operates as (W_q + delta_W_q^domain) for attention.
In Phase 2, the model operates as W_q (base) with token masking from the grammar.
The two phases are sequential, not simultaneous. Therefore:

  Interference(Phase 1, Phase 2) = 0  (by temporal separation)

This is stronger than Finding #483's failure mode, which required SIMULTANEOUS
activation of q_proj and o_proj adapters.  QED.

## Theorem 3: Grammar Specification for SOAP/Legal/JSON

For structured formats, the grammar can be specified as a context-free grammar (CFG)
or regular expression:

**SOAP format:**
  S → "S:" line+ "O:" line+ "A:" line+ "P:" line+
  line → [^:\n]+ "\n"

**Legal citation:**
  citation → case_name ", " volume " " reporter " " page " (" court " " year ")"

**JSON schema:**
  Derived from jsonschema specification (JSONSchemaBench, arXiv:2501.10868)

Each grammar defines V_valid(y_{<t}) — the set of valid next tokens given the
current parse state. This is computable in O(|grammar|) per token via a pushdown
automaton for CFGs or a finite automaton for regular expressions.

## Kill Criteria Derivation

| K | Criterion | Theorem | Prediction |
|---|-----------|---------|-----------|
| K1 | SOAP compliance >= 70% | Theorem 3 | >= 80% (grammar enforces structure) |
| K2 | Domain quality < 5pp degradation | Theorem 2 | ~0pp (domain adapter unmodified) |
| K3 | No catastrophic collapse | Theorem 2 | Guaranteed (temporal separation) |

## Implementation on MLX

Two approaches for Phase 2 (constrained decoding):

**Approach A: Token masking in generation loop**
```
for each token position:
    logits = model.forward(tokens)
    valid_mask = grammar.get_valid_tokens(parse_state)
    logits[~valid_mask] = -inf
    next_token = sample(softmax(logits))
    parse_state = grammar.advance(parse_state, next_token)
```

**Approach B: Draft-conditioned reranking**
```
draft = generate(model + domain_adapter, prompt, unconstrained=True)
candidates = beam_search(model, prompt, grammar_constraint, n_beams=4)
best = argmax([similarity(c, draft) for c in candidates])
```

Approach A is simpler and sufficient for our SOAP/legal formats. The grammar
parser runs on CPU (negligible overhead). Only the logit masking touches the
MLX computation graph.

Use outlines library (github.com/dottxt-ai/outlines) for grammar-guided generation
if available, or implement simple regex-based token masking for SOAP format.
