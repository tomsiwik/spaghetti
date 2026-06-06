# semantic-router (aurelio-labs)

**Source:** https://github.com/aurelio-labs/semantic-router

**Key Insight:** Superfast decision-making layer for LLMs using vector-space
semantics instead of text generation. Defines routes as sets of example
utterances, encodes them with any embedding model, and matches incoming queries
via cosine similarity to the nearest utterance. Returns the route of the best
match if above a threshold, else None.

**Core Algorithm:**
1. Define Route objects with example utterances per category
2. Encode all utterances and queries with same embedding model
3. At query time: cosine similarity of query against all stored utterances
4. Return the route of the highest-similarity utterance (if above threshold)

**Relevance to our work:**
- The utterance-matching pattern was adapted for our micro experiment as
  the "utterance_1nn" and "utterance_agg" routing strategies
- At micro scale with character n-gram embeddings, achieves 97% cluster
  accuracy but only 22% domain accuracy (information-limited by synthetic data)
- At macro scale with pretrained sentence encoders, this pattern could
  achieve high domain accuracy for SOLE expert routing
- Integrates with Pinecone/Qdrant for scale; our production would use FAISS

**What to use:**
- The utterance-matching routing pattern for hierarchical expert routing
- Their threshold optimization for per-route confidence calibration
- Integration patterns for vector DB scaling at N=500+
