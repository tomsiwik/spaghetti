# LEARNINGS.md — T3.5: N=100 Grassmannian Composition (Production Scale)

## Core Finding
N=100 domains compose interference-free on Gemma 4: max|cos|=2.25e-8 (4,450× margin),
99.8% routing accuracy, 257.63 MB memory — all production targets cleared.

## Why
Grassmannian orthogonality is N-independent: the bound is set by float32 downcast
precision (not by N), so max|cos| stays ~2e-8 from N=25 to N=100. Exclusive routing
eliminates all in-flight interference regardless of adapter geometry (T3.7).
Reference: HRA (arxiv 2405.17484), Finding #426 (T3.4 N=25).

## Implications for Next Experiment
- Pierre Pro production target confirmed: 100 domains, 258 MB, 0 interference.
  Real N=100 (all B nonzero) will use ~477 MB — still 8.6× under 4 GB limit.
- Routing validated only on synthetic keyword corpus (99.8%); production routing
  must be re-validated on real query distributions (T4.1 showed 86.1% at N=25 with real MMLU).
- Architecture capacity limit is Grassmannian bound N_max=426, not memory.
  C0.1 (composition port to Gemma 4 with real trained adapters) is the next critical gate.
