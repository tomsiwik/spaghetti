# LEARNINGS — exp_spark_adapter_as_head_compass

## Core finding
Reading the math adapter's per-head B-column energy as a "compass" to select which base attention
heads to amplify carries zero selection signal: compass-selected heads (GSM8K EM +3.75pp vs base)
tied a completely disjoint random head set exactly (0pp gap, 0 overlapping heads), and the adapter
delta applied directly to logits dominated both at +23.75pp.

## Why
Per-head B-energy is nearly uniform across all heads (scores span ~1.3×, top 9.9e-5 to low 8.8e-5),
so the ranking is effectively noise — any 12 heads amplified by a small γ produce a similar generic
"slightly louder base" effect. The adapter's value is in its learned A→B transformation, not in
identifying structurally special base heads to boost without it.

## Implication for the next experiment
Do not pursue any read-only compass/selector scheme derived from adapter weight statistics to
modulate the base model; the adapter's information lives in applying its delta, not in inspecting
its columns. The no-thinking harness (base 46.25, adapter +23.75pp) is a high-headroom substrate
where the adapter supplies reasoning structure the base lacks — future composition experiments
should exploit this regime rather than the near-ceiling thinking harness where the same adapter
hurts (-10pp, F#866/873). "Adapter toxicity" is harness-relative: cross-experiment EM comparisons
are invalid unless the thinking/prompt/parser configuration matches exactly.
