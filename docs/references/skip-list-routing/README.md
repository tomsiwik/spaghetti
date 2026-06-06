# Skip-List Multi-Resolution Routing

## Source
- **Skip Lists**: Pugh, W. (1990). "Skip lists: a probabilistic alternative to balanced trees." Communications of the ACM, 33(6), 668-676.
- **Wikipedia**: https://en.wikipedia.org/wiki/Skip_list

## Relevance to Project
Skip lists provide O(log N) probabilistic search via multi-level indexing with
geometric level spacing (each level has ~half the elements of the level below).
Applied to MoE routing: organize N experts at multiple resolution levels, route
top-down from coarse to fine, with adaptive depth per token.

## Key Properties
- Expected O(log N) search time (vs O(N) for flat scan)
- Probabilistic structure (no rebalancing needed, unlike balanced trees)
- Each element promoted to level k with probability p^k (typically p=0.5)
- "Express lanes" at higher levels for fast approximate lookup

## Connection to Existing Work
- hierarchical_tree: fixed-depth binary tree routing (always traverses all levels)
- huffman_tree: frequency-weighted depth (optimal expected depth)
- skip_list_routing: adaptive depth per token (some stop early, some go deep)

## Implementation
Custom implementation in micro/models/skip_list_routing/. No external library
needed -- the data structure concept is adapted to neural routing, not used as
an actual data structure.
