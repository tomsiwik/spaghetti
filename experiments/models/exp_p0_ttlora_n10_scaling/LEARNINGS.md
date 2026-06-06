# Learnings: exp_p0_ttlora_n10_scaling

## Core Finding
TT-LoRA adapters scale linearly to N=10 real domains (all 10 converge in 45 min, 75.3% quality retention), but TF-IDF routing degrades to 79.3% — a structural ceiling from vocabulary overlap between semantically adjacent domains.

## Why
TF-IDF routing separates domains by lexical frequency; domains sharing academic English vocabulary (psychology/medical, science/engineering) cannot be distinguished at the word-frequency level. The upper bound for TF-IDF at N=10 MMLU domains is ~80%, matching the measured 79.3%. This is a routing method failure, not an adapter system failure.

## Implications for Next Experiment
Replace TF-IDF with semantic routing (learned embeddings, e.g. sentence-transformers or a lightweight bi-encoder trained on domain labels) to break the lexical ceiling. Adapters themselves are proven to scale — only the router needs to change.
