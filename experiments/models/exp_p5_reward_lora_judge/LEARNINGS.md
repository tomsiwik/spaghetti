# LEARNINGS: exp_p5_reward_lora_judge

## Core Finding
Rank-16 LoRA reward adapters (5.01 MB, 83.2ms avg) achieve 100% format-level discrimination
across math/legal/SOAP on M5 Pro — confirming reward LoRA infrastructure viability, but the
task is trivially easy (loss → 0.0 by step 50, margins 20-47).

## Why
Domain format differences (LaTeX vs plain text, SOAP headers vs conversational) are so
salient they barely require learned features. Finding #474 (6-dim TF-IDF separation) predicted
this: rank-16 provides 512-dim capacity for a ≤6-dim problem — 85x overcapacity.
Ref: arXiv:2506.05748 (reward LoRA, 0.8% params → 96.2% RewardBench).

## Implications for Next Experiment
The real test is **intra-domain quality discrimination**: good LaTeX vs mediocre LaTeX,
correct SOAP note vs one with wrong ICD codes. This requires preference data within a domain,
not across domains. Also: reward score as inline router validator (did composition select
the right adapter?) is now feasible at 5 MB / 83ms per check.

## Key Constraints Confirmed
- v_proj excluded from LoRA target: layers 22+ use shared KV (num_kv_shared_layers=20)
- SOAP sequences ~800 tokens → 103.8ms individually (exceeds 100ms); truncation fixes this
- Early stopping at loss < 0.01 saves ~75% of training iterations (converges at step ~50)
- Same serving pipeline as domain adapters: hot-swap compatible
