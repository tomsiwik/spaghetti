# MATH.md: Value of Routing Ablation

## Type: Guided Exploration (Type 2)

The proven framework is that SFT adapters with Grassmannian A-matrices produce
domain-specialized models. The unknown is whether routing between them provides
measurable behavioral benefit over simply using the single best adapter.

## A. Failure Mode Identification

**The failure mode being tested:** Routing adds complexity without behavioral benefit.

If a single adapter k has quality q_k(d) >= alpha * max_i(q_i(d)) for all domains d,
then routing provides at most (1-alpha) relative improvement. When alpha is close
to 1, routing is wasted effort.

This is a real risk because:
- Finding #204: Code adapter selected for 42/50 queries by energy gap routing
- Finding #203: Wrong-adapter cost is only ~13% at PPL level
- TF-IDF routing (90% accuracy) produces WORSE scores than energy gap (36% accuracy)
  on legal and finance domains, suggesting domain-specific adapters hurt these domains

## B. The Right Question

NOT: "How do we improve routing accuracy?"
Instead: "What is the information-theoretic value of routing for SFT adapters?"

Formally: Given N adapters with quality functions q_i: D -> [0,1] where D is the set
of domains, define:
- q_best(d) = max_i q_i(d) (oracle per-domain selection)
- q_code(d) = quality of code adapter on domain d
- alpha(d) = q_code(d) / q_best(d) (coverage ratio)

The value of routing V = sum_d [q_best(d) - q_code(d)] / sum_d q_best(d)
                       = 1 - sum_d alpha(d) * q_best(d) / sum_d q_best(d)

If V < epsilon for some small epsilon, routing is not worth its complexity.

## C. Prior Mathematical Framework

This is an instance of the **explore-exploit tradeoff** in multi-armed bandits
(Robbins, 1952). When one arm dominates all others, the optimal policy is to
always pull that arm (no exploration needed).

More specifically, it relates to the **specialization vs. generalization** tradeoff
in mixture of experts (Shazeer et al., 2017): if expert specialization is weak
(all experts have similar quality across domains), gating adds overhead without
benefit. DeepSeek-V3 (2024) uses auxiliary-loss-free load balancing precisely
because over-specialization can hurt.

The DES-MoE literature (Finding referenced in experiment notes) shows 43-76% drops
from wrong routing at the TASK level. If our adapters show much smaller drops, they
are not sufficiently specialized.

## D. Predictions (derived from existing data)

From Findings #204 and #207, we can predict alpha(d) for each domain.

Using existing composite scores (which mix keyword density with execution metrics):
- medical: alpha = 0.4086/0.4746 = 0.86 (code nearly matches base, domain adapter no better)
- code: alpha = 1.0 (code adapter IS the best)
- math: alpha = 0.5102/0.5784 = 0.88 (code adapter captures 88% of math benefit)
- legal: alpha = 0.4321/0.4644 = 0.93 (code adapter BETTER than domain adapter)
- finance: alpha = 0.4414/0.4715 = 0.94 (code adapter BETTER than domain adapter)

**But these use keyword-density metrics for prose, which Finding #179 showed unreliable.**

### Predictions for this experiment (execution-based metrics):

**P1 (Structured domains - code, math):** Code adapter captures >= 80% of domain-specific
adapter quality, because code SFT training teaches general instruction-following.
- Code: syntax pass rate identical (both ~70%)
- Math: correctness within 10pp (code: 70%, math-specific: 80%)

**P2 (Knowledge domains - medical, legal, finance):** This is the key unknown.
Two competing hypotheses:
- H2a: Code adapter captures >= 80% because SFT mainly teaches format compliance,
  and domain knowledge comes from the base model. (alpha >= 0.8 everywhere)
- H2b: Code adapter captures < 50% because domain-specific SFT teaches domain
  vocabulary and reasoning patterns absent from code training. (alpha < 0.5 on prose)

**P3 (Routing value):** If H2a holds, V < 0.1 and routing is not justified.
If H2b holds, V > 0.3 and routing is essential.

### Kill criteria mapping:
- K608 (code >= 50% of routed total): Tests whether code adapter is a viable
  universal adapter. Threshold 50% is generous; if it fails, code adapter is
  clearly inadequate.
- K609 (2+ domains where domain-specific beats code): Tests whether specialization
  exists. If fewer than 2 domains benefit from specialization, routing has
  minimal value for this adapter set.
- K610 (execution-based metrics): Methodological requirement. Previous results
  used keyword density for prose, which is unreliable.

## E. Assumptions & Breaking Conditions

1. **Adapters are the same ones from v3.** If adapters are retrained, results change.
2. **Evaluation prompts are from validation set.** Different prompts could shift results.
3. **n=10 per domain.** Small sample; effects < 20pp may not be statistically significant.
4. **Execution-based eval for prose domains is approximate.** Without a gold-standard
   factual QA benchmark, we use structured answer extraction and factual consistency
   checks. This is better than keyword density but not perfect.
5. **lora_scale=20 is not ablated.** Different scales could change the relative
   performance of adapters.

If Assumption 3 fails (n too small), we cannot distinguish H2a from H2b.
If Assumption 4 fails (eval is still unreliable), findings remain provisional.

## F. Worked Example

For d=5 domains, suppose execution-based scores are:

| Domain   | Base | Code Adapter | Domain Adapter | alpha |
|----------|------|-------------|----------------|-------|
| medical  | 0.3  | 0.4         | 0.5            | 0.80  |
| code     | 0.5  | 0.7         | 0.7            | 1.00  |
| math     | 0.1  | 0.7         | 0.8            | 0.875 |
| legal    | 0.3  | 0.4         | 0.5            | 0.80  |
| finance  | 0.3  | 0.4         | 0.5            | 0.80  |

V = 1 - (0.4+0.7+0.7+0.4+0.4)/(0.5+0.7+0.8+0.5+0.5)
  = 1 - 2.6/3.0
  = 0.133

So routing provides 13.3% relative improvement. Whether this justifies the
complexity of routing depends on the deployment context.

K608: Code total = 2.6, Routed total = 3.0. 2.6/3.0 = 86.7% >= 50%. PASS.
K609: medical, legal, finance have domain > code. 3/5 >= 2/5. PASS.

## G. Complexity & Architecture Connection

This ablation adds zero parameters and zero FLOPs to inference. It simply
measures whether routing overhead (TF-IDF classifier or energy gap computation)
is justified by quality gains.

If routing is not justified, the architecture simplifies dramatically:
- No routing mechanism needed
- Single adapter applied to all queries
- Inference cost: base_model + one LoRA forward pass (no branching)

This would mean the "composable experts" thesis needs refinement: instead of
runtime composition of specialized experts, the value may be in training
specialized adapters whose best specimen generalizes.

## Self-Test

1. What is the ONE mathematical property that makes the failure mode impossible?
   There is no impossibility guarantee -- this experiment tests WHETHER the failure
   mode (routing is useless) actually obtains. If it does, the composition thesis
   needs revision.

2. Which existing theorem(s) does the proof build on?
   Multi-armed bandit optimality (Robbins 1952): when one arm dominates, always
   pull it. MoE specialization-generalization tradeoff (Shazeer et al., 2017).

3. What specific numbers does the proof predict?
   P1: Code adapter >= 80% of domain-specific on code and math (execution metrics).
   P2a/P2b: Code adapter coverage on prose domains is the key unknown (>= 80% or < 50%).
   P3: Routing value V = 1 - mean(alpha) across domains.

4. What would FALSIFY the proof?
   If domain-specific adapters dramatically outperform code adapter on prose domains
   (alpha < 0.3), routing is clearly essential and the "universal adapter" hypothesis
   is wrong.

5. How many hyperparameters does this approach add?
   0 -- this is an ablation, not a method.

6. Hack check: Am I adding fix #N to an existing stack?
   No. This is asking whether a core component (routing) is necessary at all.
   The answer directly informs architecture decisions.
