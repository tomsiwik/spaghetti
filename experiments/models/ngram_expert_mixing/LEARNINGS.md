# Learnings: exp_ngram_expert_mixing

## Core Finding

Mixing a 5-gram language model with a small neural model via fixed-weight interpolation
(alpha=0.7) yields 20.8% PPL reduction on character-level names data, but the true
complementarity signal is only 3.36 ppt (mixed > pure n-gram), and the result is heavily
confounded by the neural model being weaker than the n-gram model on this task.

## Why This Happened (Literature-Grounded)

The improvement is mechanistically straightforward: convex combination of two probability
distributions that make complementary errors yields a better combined distribution. This is
a well-established property of linear interpolation going back to Jelinek & Mercer (1980).

The reason the n-gram model dominates here is task-specific: character-level names with
V=27 have extremely strong local statistical regularity (phonotactic patterns, common
bigram/trigram sequences). A 202K-param neural model trained for only 1000 steps cannot
fully capture these patterns. The n-gram model's advantage comes from perfectly memorizing
the training distribution's local statistics, which is exactly what n-grams excel at.

The 15.84% padding asymmetry confound (zero-padding without attention masking) further
inflates the neural model's weakness on short sequences, making the n-gram's compensation
appear larger than it would be with proper evaluation.

**Infini-gram (Liu et al., 2024, arxiv 2401.17377)** provides the strongest confirming
evidence that this mechanism scales: interpolating an unbounded n-gram model (built on
suffix arrays over 5 trillion tokens) with neural LMs reduces perplexity by up to 73%,
*even for Llama-2-70B*. This is the critical finding for our project: n-gram mixing
does NOT have diminishing returns with model scale when the n-gram model is also scaled.

## Confirming Evidence

1. **Infini-gram (arxiv 2401.17377, COLM 2024):** Scaled n-gram (infinity-gram via suffix
   arrays) interpolated with neural LMs gives up to 73% PPL reduction. Works even at 70B
   scale. Uses instance-wise interpolation hyperparameters via Random Forest, yielding
   3-20% additional gains over fixed interpolation. This directly validates our finding
   that n-gram mixing helps, and suggests scaling the n-gram component (not just the
   neural component) is key.

2. **Parameter Golf (OpenAI, 2026):** The competition's biggest breakthrough was multi-order
   n-gram backoff (2-7 gram) + entropy-adaptive alpha, achieving sub-1.0 BPB (0.9674).
   The community found 0.10-0.16 BPB gains from n-gram mixing over neural-only models.
   Notably, entropy-adaptive mixing was the winning approach in this competition, contrary
   to our finding that fixed-weight wins -- but the Parameter Golf task is harder
   (diverse text, V=256 bytes), where entropy adaptation is more valuable because n-gram
   reliability varies more across contexts.

3. **Mikolov et al. (2011-2012):** RNN + Kneser-Ney 5-gram interpolation was the standard
   approach in the pre-Transformer era. The complementarity mechanism (n-grams for local
   patterns, neural for global structure) is well-established.

4. **Khandelwal et al. (2020, kNN-LM):** Showed that interpolating neural LMs with a
   retrieval-based model (nearest neighbor in embedding space) yields significant PPL
   reductions, demonstrating the broader principle that non-parametric memory complements
   parametric models.

## Contradicting Evidence

1. **Fixed vs adaptive mixing:** Our finding that fixed alpha=0.7 beats entropy-adaptive
   mixing (by 1.45 ppt) likely does NOT generalize. The Parameter Golf competition found
   entropy-adaptive mixing essential for real text. Our toy task (V=27, strong n-gram
   regularity) means the n-gram is reliable everywhere, making entropy adaptation
   unnecessary. On real text with V=32K, n-gram reliability varies dramatically across
   contexts, and entropy-adaptive mixing should dominate.

2. **Vocabulary scaling concern:** Our experiment uses V=27 (characters). At V=32K
   (BPE tokens), 5-gram tables become exponentially larger. The infini-gram paper
   addresses this with suffix arrays (7 bytes/token storage), but requires a massive
   index infrastructure. For our edge deployment (M5 Pro, 48GB), standard hash-table
   n-gram storage at V=32K would be impractical at higher orders. We may need to limit
   to 2-3 grams at token level, where our experiment showed marginal (0-5%) improvement.

3. **kNN-LM may be strictly better:** Khandelwal et al. showed that kNN retrieval in
   embedding space captures more useful context than n-gram matching, achieving ~55%
   of kNN-LM gains with a simple MLP replacement. For our architecture, a lightweight
   retrieval mechanism in the composition pipeline may outperform n-gram mixing.

## Alternative Approaches (What We Could Try Instead)

1. **Infini-gram with suffix arrays (arxiv 2401.17377):** Instead of hash-table n-gram
   storage, use suffix arrays for unbounded n-gram lookup. Storage is 7 bytes/token
   with millisecond latency. Would allow scaling to higher n-gram orders without the
   exponential memory explosion. However, the index infrastructure is heavy for edge.

2. **Instance-wise interpolation (Random Forest):** Infini-gram showed that learning
   per-instance interpolation weights via Random Forest yields 3-20% additional gains
   over fixed interpolation. This is a more principled approach than our entropy-adaptive
   method: instead of using n-gram entropy as a proxy for reliability, learn the
   reliability from features of the context.

3. **kNN-LM / retrieval-augmented generation:** Replace n-gram tables with a retrieval
   datastore in the neural model's embedding space. More semantically meaningful than
   string matching. Could integrate with our composition pipeline: each expert builds a
   small domain-specific retrieval index. The MLP replacement approach (55% of kNN-LM
   gains at 4% storage) is especially attractive for edge deployment.

4. **N-gram cache during TTT (Test-Time Training):** From Parameter Golf, the n-gram
   cache is most powerful when combined with TTT. During per-document adaptation, build
   a document-specific n-gram table and mix with the adapting model. This aligns with
   our P1 TTT priority.

5. **Multi-order backoff with learned weights:** Instead of stupid backoff's fixed 0.4
   multiplier, learn backoff weights per n-gram order on held-out data. This is
   essentially Modified Kneser-Ney, which is known to significantly outperform stupid
   backoff.

## Implications for Next Experiments

1. **The 3.36 ppt complementarity signal is real but modest.** At macro scale with a
   stronger neural model, the headline improvement will be much smaller. Focus on the
   complementarity mechanism, not the absolute numbers.

2. **Entropy-adaptive mixing will matter more at scale.** On real text (V=32K, diverse
   domains), fixed-weight mixing is unlikely to be optimal. The Parameter Golf evidence
   strongly supports entropy-adaptive approaches for production use.

3. **Vocabulary scaling is the real challenge.** V=27 -> V=32K is a 1000x increase.
   Standard n-gram tables become impractical at 4-5 gram orders. Either:
   - Use suffix arrays (infini-gram approach, but heavy for edge)
   - Limit to 2-3 grams (marginal benefit in our experiment)
   - Use character-level n-grams on subword-decoded text (novel, untested)
   - Switch to kNN-LM / embedding-based retrieval instead

4. **N-gram + TTT is the highest-leverage combination.** Parameter Golf showed n-gram
   cache is most powerful when combined with test-time training. This directly feeds
   into our P1 TTT priority (exp_ttt_expert_selection).

5. **Integration with composition pipeline:** The n-gram mixing technique works at the
   logit level (post-softmax mixing). This means it can be applied AFTER expert
   composition with zero interference to the Grassmannian/pre-merge mechanism. It's
   an orthogonal improvement -- which is exactly its value.
