# LEARNINGS — exp_spark_interference_eos

**Core finding:** The first decode step where an off-domain (medical) LoRA delta-output magnitude
overtakes the on-domain (math) delta is NOT a content-exhaustion signal: the crossover fires at
T_cross in {0..4}, before the answer span, in 34/35 correctly-solved cases, collapsing accuracy
from 0.70 to 0.02 (Δacc = −0.68) when used as an early-stop trigger.

**Why:** The two r=6 q_proj adapters perturb the base logits by comparable magnitudes
(off/on L2-delta ratio median 0.961, range 0.66–1.41). When magnitudes are this close and
order-unstable, the first off>on crossing is near-immediate and content-blind — it carries zero
information about where answer content lies in the generation. The "on-domain delta stays large
while useful, then decays" premise is simply false.

**Implication for the next experiment:** Adapter-delta MAGNITUDE (L2 norm of logit perturbation)
carries no usable domain-relevance or answer-position signal. Any proposed mechanism that reads
delta magnitude as a proxy for relevance, content state, or termination timing is dead on arrival
— consistent with F#864 (logit-shift sign is not a domain signal) and F#867 (loudness is not
correctness). Future routing and early-exit work must be grounded in a signal that is structurally
tied to content, not adapter amplitude.
