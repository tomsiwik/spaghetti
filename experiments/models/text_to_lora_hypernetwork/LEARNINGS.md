# Learnings: Text-to-LoRA Hypernetwork

## Key Finding
Text-to-LoRA hypernetwork generation is killed for our setup (K2 FAIL: 0.45% retention
after orthogonal projection). But nearest-neighbor adapter retrieval in embedding space
is surprisingly effective (mean 1.28x PPL ratio to trained adapter).

## What Worked
- Domain description embeddings from BitNet-2B-4T cluster semantically (cos 0.64-0.87)
- NN retrieval gives 50-85% of trained adapter improvement over base, with zero training
- Memory trivially passes (7.5 GB total)

## What Failed
- Hypernetwork B-cosine 0.036 (essentially random) -- 24 training pairs catastrophically insufficient
- Convex-combination architecture cannot generate novel adapters (99.4% in span of training set)
- Post-processing destroys generated adapter (0.59% retention)

## Root Cause
T2L paper trains on thousands of diverse (task, adapter) pairs. We have 24 from single base.
The 10.9M output dimension vs 24 examples makes the inverse problem hopelessly underdetermined.
Even the simplified convex-combination architecture (23 coefficients) collapses to near-uniform.

## Implication for SOLE
NN retrieval is a free routing mechanism. No router training needed -- just embed the input
and use the nearest domain's adapter. The softmax router already does this equivalently
(and matches oracle quality), confirming the approach.

Adapter generation is dead for N=24. New domains require explicit fine-tuning (200 iters, ~65s).
This is acceptable given training cost.
