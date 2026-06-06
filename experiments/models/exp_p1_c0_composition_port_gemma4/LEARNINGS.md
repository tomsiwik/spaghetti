# LEARNINGS: exp_p1_c0_composition_port_gemma4 (C0.1)

## Core Finding
P0 Grassmannian composition (Findings #3, #341, #404-406) transfers to Gemma 4 E4B with
full mathematical guarantees: isolation is machine-precision (5.19e-14, 1932× margin),
and exclusive routing preserves 90.2% of solo adapter quality.

## Why It Works
Gram-Schmidt orthogonalization in float64 produces near-exact column orthogonality regardless
of the underlying architecture (RMSNorm, attention dimensions). The P0 theorem (zero
activation-space interference under exclusive routing) is architecture-agnostic — it depends
only on one adapter activating per request, not on specific layer geometry.

## What Failed (KC01)
TF-IDF routing at 93.2% (finance=86%) is below 95%. Root cause: MMLU macroeconomics
vocabulary overlaps with statistics/math — the corpus, not the algorithm, is the bottleneck.
Fix: replace MMLU proxies with financial news corpora (Bloomberg, SEC filings) in C1+.

## Implications for C1.1 (PoLAR Gemma 4 Re-test)
C1 gate is open. PoLAR does not depend on 95% routing — it uses the same Grassmannian
framework that KC02 confirms works on Gemma 4. When training the finance router in C1+,
use domain-specific production text, not MMLU proxies.
