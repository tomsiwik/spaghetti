# TIES-Merging: Resolving Interference When Merging Models

**Source:** https://arxiv.org/abs/2306.01708 (NeurIPS 2023)

**Key Insight:** Three-step merge: (1) Trim small-magnitude deltas, (2) Elect
sign by majority vote per parameter, (3) Average only agreeing-sign deltas.
Resolves sign conflicts that degrade naive weight averaging.

**Relevance to our work:**
- Directly applicable to our weight averaging composition fallback (+1.5% vs joint)
- TIES could reduce that gap by resolving sign conflicts in LoRA deltas
- Our orthogonality diagnostic (cos ~ 0.000) suggests sign conflicts may be
  rare at micro scale, but at macro scale with real domains they could matter
- Relevant to `exp5_macro_match` and `exp_ortho_diagnostic`

**What to use:**
- The trim-elect-merge algorithm (drop-in replacement for naive averaging)
- Their analysis of interference patterns between task vectors
- Comparison methodology against naive averaging baseline
