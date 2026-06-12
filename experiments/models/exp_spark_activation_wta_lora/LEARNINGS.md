## Core finding

Per-token activation-L2 winner-takes-all (WTA) routing is decisively refuted: the loudest q_proj adapter per token is anti-correlated with correctness, not correlated — wta_full (0.4625) fell below the no-adapter base (0.625) and 26 pp below random-pick (0.725), while the uniform-1/N merge it aimed to beat was best of all (0.900).

## Why

Two separable causes: (1) full-scale single-delta injection destabilizes the frozen 4-bit residual stream regardless of which adapter is chosen — s/N dilution is what keeps the merge stable; (2) the geometric claim that an on-domain expert is loud on its own tokens is false for these trained q_proj adapters — distractor adapters are loud on off-domain tokens often enough that argmax-on-norm selects against the correct expert below chance.

## Implication for the next experiment

Hard per-token routing keeps losing (reinforces F#863); s/N dilution is the stable floor. Any new routing scheme must operate at diluted magnitude (never inject a single full-scale delta) and use a signal that is positively correlated with correctness — raw activation L2-norm is an excluded signal class.
