# LEARNINGS: exp_p0_semantic_routing_n10

## Core Finding
Sentence-embedding features (all-MiniLM-L6-v2) solve the TF-IDF routing bottleneck
at N=10: combined logistic achieves 89.9% (+8.6pp over Ridge baseline), with no
domain below 78% and total routing latency under 1 second.

## Why
Feature quality, not classifier complexity, determines the routing ceiling.
Embedding space has 4.9x better Fisher separation (0.133 vs 0.027) than TF-IDF,
explaining the accuracy gap. Trained classifiers add incremental value on top
(+3.6pp in embedding space vs +8.4pp in TF-IDF space — the less separable the
features, the more training compensates). Ref: arXiv:2402.09997 (LoraRetriever).

## Theorem 3 Caveat
Trace-ratio Fisher J(combined) < J(embedding) — concatenating noisy TF-IDF
features dilutes the ratio even when the classifier extracts benefit. Theorem 3
holds for determinant-based Fisher or pre-whitened features, not trace-ratio.
The accuracy improvement (89.9% > 88.0%) is real; the theorem was overstated.

## Remaining Gap
Psychology peaks at 78% due to genuine semantic overlap with medical and
philosophy. To exceed 90%: (a) contrastive fine-tuning on domain labels, or
(b) hierarchical routing (cluster semantically similar domains → disambiguate
within cluster using domain-specific keywords).

## Implications for Next Experiment
Routing at 89.9% is likely behaviorally sufficient (Finding #298: misrouting
between semantically adjacent domains is PPL-benign). Next priority: validate
end-to-end — does correct adapter routing produce measurably better generation
quality vs random routing? Or: pursue hierarchical routing to clear the 90%
threshold cleanly.
