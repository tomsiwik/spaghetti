# LEARNINGS: exp_p0_embedding_routing_n25

## Core Finding
Combined TF-IDF+embedding logistic routing achieves 88.8% at N=25 with only 1.1pp
degradation from N=10, and no domain below 74.1%. Routing at N=25 is behaviorally solved.

## Why
Feature complementarity GROWS with N: embedding+TF-IDF fusion gains +4.7pp over
embedding-alone at N=25 vs +1.9pp at N=10. Individual MMLU subjects have more
distinctive domain-specific vocabulary than meta-groups, so TF-IDF becomes MORE
valuable at finer granularity. MiniLM sentence embeddings (arXiv:1908.10084) do NOT
collapse at N=25 (79.4%), definitively refuting Finding #256 which used base-model
mean-pool embeddings.

## Implications for Next Experiment
Path to 90%+: increase TF-IDF max_features from 5000 to 20000 (Finding #431 baseline)
and training data from 200 to 300 per domain. Latency is the open problem — 50ms from
MiniLM is 10x the 5ms target; tiered routing (TF-IDF first, embed-refine on low-confidence)
or ONNX quantization should be explored. With routing solved at N=25, the next bottleneck
is end-to-end pipeline: adapter loading, composition quality, and generation benchmarks.
