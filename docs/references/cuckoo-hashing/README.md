# Cuckoo Hashing

## Source

- Paper: Pagh & Rodler (2004), "Cuckoo Hashing", Journal of Algorithms
- Wikipedia: https://en.wikipedia.org/wiki/Cuckoo_hashing
- Cuckoo Filter extension: Fan et al. (2014), "Cuckoo Filter: Practically Better Than Bloom"

## Key Insight

Two hash functions provide two candidate slots for every key. If the primary
slot is occupied, the new key evicts the occupant, which moves to its
alternative slot. This creates eviction chains that resolve collisions
deterministically with O(1) worst-case lookup.

Load factor must stay below 50% (with 2 tables) for amortized O(1) insertion.
Beyond 50%, eviction cycles become likely and the table must be rehashed.

## Relevance to Our Work

Maps to MoE expert routing: each token has two candidate expert sets (via
two learned hash functions). If the primary expert is a poor fit (routing
"collision"), evict to the secondary expert set. This resolves the softmax
score-tie problem where multiple experts get similar scores (measured at
57.4% collision rate at micro scale).

## Applied In

- `micro/models/cuckoo_collision_free_routing/` -- dual-hash routing with
  soft eviction blending
- Result: +0.15% vs softmax (within noise), both kill criteria pass
