# LEARNINGS: exp_tfidf_routing_real_text

## Core Finding
TF-IDF nearest-centroid routing achieves 100% accuracy on real NLP (math/code/text),
confirming that Finding #354's toy-domain result (95%) transfers to — and is exceeded by — real NLP.

## Why
Longer real-domain text provides richer n-gram features (unigrams + bigrams) than toy data,
giving cleaner decision boundaries. Despite cos(math,text) = 0.504 (above predicted <0.30),
discriminating n-grams ("how many", "in python") maintain a clean boundary independently of
centroid proximity. Future theorems should predict routing accuracy directly, not centroid cosines.

## Implications for Next Experiment
Routing is NOT the bottleneck for a 3-domain real NLP system. The Q_wrong harm identified in
Finding #386 (-58% relative on language tasks) is fully addressable by routing. The next
bottleneck is M2P adapter quality: d_M2P=64 captures only 77–88% energy (Finding #387).
→ exp_m2p_vera_bottleneck: VeRA-style shared basis to expand effective capacity while
  reducing parameter count from 357M to 4.7M (76×).
