# Gradient Coding / Reed-Solomon for Distributed ML

## Source
- Tandon et al. (2017). "Gradient Coding: Avoiding Stragglers in Distributed Learning." https://arxiv.org/abs/1709.01505
- Lee et al. (2018). "Speeding Up Distributed Machine Learning Using Codes." IEEE Trans. Info Theory.

## Key Insight
Reed-Solomon codes applied to gradient computation in distributed training:
encode data partitions across workers so that the master can recover the full
gradient from any k of n workers (tolerating n-k stragglers). The encoding
uses Vandermonde matrices over the reals, which is algebraically identical
to our Lagrange interpolation approach for expert weight encoding.

## Relevance to Our Work
- Proves RS codes work over real-valued tensors (not just finite fields)
- Our application differs: we encode expert WEIGHTS, not gradients
- Their encoding is for training-time straggler tolerance; ours is for
  inference-time fault tolerance of the expert library
- Same mathematical primitive (Lagrange interpolation / Vandermonde matrices)
- Their work validates the numerical stability of real-RS for neural network
  scale tensors

## What We Reuse
- The insight that real-valued polynomial interpolation has negligible
  numerical error for neural network weight magnitudes
- Chebyshev node selection for minimizing Lebesgue constant
- The framing of k-redundancy for fault tolerance
