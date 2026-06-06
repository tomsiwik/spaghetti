# Consistent Hashing

**Papers:**
- Karger et al. 1997: "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web" (https://dl.acm.org/doi/10.1145/258533.258660)
- Lamping & Stepanov 2014: "A Fast, Minimal Memory, Consistent Hash Algorithm" (Jump Consistent Hash, https://arxiv.org/abs/1406.2294)

**Key property:** Adding/removing one node from N nodes displaces only 1/N of keys. This is the minimal displacement possible for any hash scheme.

**Implementations:**
- Jump consistent hash: 5-line algorithm in any language
- Ketama: libmemcached's consistent hashing with virtual nodes
- Python: `hashring` package, `uhashring` package
- Go: `consistent` package (github.com/stathat/consistent)

**Relevance to this project:**
Used as the routing mechanism in `consistent_hash_routing` experiment. Experts are nodes on the ring, tokens are keys. Adding a new expert displaces ~1/N of routing decisions without recalibration.

**How we adapted it:**
- Virtual nodes (150 per expert) for load balance
- FNV1a hash for ring position computation
- Fixed random projection to map d-dimensional hidden states to scalar ring positions
- Softmax over inverse ring distances for smooth routing weights
