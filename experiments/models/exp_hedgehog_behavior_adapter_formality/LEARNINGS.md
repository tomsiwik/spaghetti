# LEARNINGS.md — exp_hedgehog_behavior_adapter_formality

## Core Finding
PROVISIONAL design-lock (F#724). **1st dual-target / zero-proxy KC design** in the
Hedgehog-framework super-family (9 designs / 0 measurements). 2nd behavior-axis
instance (cousin of F#683 politeness). K1879 = behavioral acquisition target
(formality auto-judge Δ ≥ +10 pp over 50 neutral prompts); K1880 = style/substance
orthogonality safety target (|Δ accuracy| ≤ 2 pp on MMLU-100 seed=42).

## Why
K1880 introduces a NEW KC class: **safety targets** (non-interference /
orthogonality). By construction such KCs belong in the target column — a
proxy version would be a tautology. F#666 kill matrix extends axis-invariantly:
paired-target→PROVISIONAL; pure-proxy→KILL; **dual-target/zero-proxy→PROVISIONAL**
(more conservative: lacks structural-proxy short-circuit, so the conservative
verdict is hold-for-`_impl`). Behavior-axis sub-cluster opening past domain-axis
super-saturation (closed at 5 post-F#718) is justified: behavior was under-
represented 1:5; formality ⊥ politeness on canonical register theory.

## Implications for Next Experiment
1. **Zero-proxy sub-pattern at 1 instance** — promote to standalone memory at
   3rd instance (analyst's 3-instance threshold). Watch for safety-target KCs
   on future novel-mechanism filings (non-interference, orthogonality, drift).
2. **Behavior-axis sub-cluster at 2 instances** — a 3rd behavior axis
   (conciseness adapter, already open P=2 in DB) triggers sub-cluster
   standalone-memory promotion. Claim order: conciseness next if drainage
   prioritizes super-family closure.
3. **26B teacher cache remains highest-leverage unblock** — blocks 9+
   Hedgehog-framework `_impl` dependents. Single standalone-prereq-task.
4. **F#683 `_impl` is not a blocker for this experiment** (different neutral
   prompt set, different teacher capture) but its Phase A/B/C pipeline
   pattern is reusable engineering. Land F#683 `_impl` first to establish
   the reusable template.
5. **KC-design taxonomy now 3-way bifurcated** — update KC-design classifier
   to include zero-proxy as a third verdict-path alongside paired-target
   and pure-proxy. Future preempt-KILL checks must exclude this branch.
6. **No antipattern** — this is a novel, well-designed filing. No memory
   update beyond the sub-pattern watchlist above.
