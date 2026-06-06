# LEARNINGS: Cross-Domain Scale Phase Transition (KILLED)

## Core Finding

**Math's sharp phase transition at s*=[4,6] is NOT a universal LoRA property — it is an
evaluation artifact created by binary exact-match scoring interacting with format activation.**
Code shows noisy, non-monotonic scaling. Medical shows near-zero adapter effect at all scales.
The two-regime model (FORMAT vs CAPABILITY) from Finding #249 does not generalize beyond
discrete-answer tasks.

## Why This Happened

The kill has three independent causes, each grounded in literature:

### 1. Evaluation Metric Artifact (Primary Cause)

**Schaeffer et al. (2023) "Are Emergent Abilities a Mirage?" (arXiv:2304.15004, NeurIPS
Outstanding Paper)** demonstrated that apparent emergent abilities (sharp phase transitions)
are created by researchers' choice of nonlinear/discontinuous metrics. Binary exact-match
(used in GSM8K) is precisely such a metric: it maps continuous model improvement to
step-function output. Our math evaluation uses exact numerical match — the adapter at s>=6
activates GSM8K training format ("<<3*26=78>>, #### 78") that the regex can parse, creating
an apparent 0.60 jump. Code and medical use continuous metrics (syntax+recall, factual overlap)
that smooth over the same underlying change, producing no apparent phase transition.

This is **exactly** the Schaeffer mechanism applied to LoRA scale instead of model scale.
Finding #250's "phase transition" is a metric artifact, not a capability transition.

### 2. Task-Dependent Perturbation Requirements

**FlexMoRE (Flexible Mixture of Rank-heterogeneous Experts)** showed empirically across 120
tasks that optimal LoRA rank is substantially higher for reasoning-heavy benchmarks than
knowledge-heavy ones. **DR-LoRA (Dynamic Rank LoRA)** confirmed that uniform adapter scaling
causes resource mismatch — task-relevant experts are under-provisioned.

Our three domains confirm this pattern:
- Math: reasoning-heavy, benefits from high perturbation (s=20)
- Code: mixed (some prompts already strong at base, others need adaptation)
- Medical: knowledge-heavy, adapter learned nothing useful (max delta +0.028)

### 3. Bimodal Code Behavior

Code prompts split into Group A (base strong, adapter disrupts) and Group B (base weak,
adapter helps). This matches the "Standing Committee" finding — the base model already has
concentrated code expertise for common patterns, making adapter perturbation destructive
on those prompts while activating capability on novel patterns.

## Confirming Evidence

- **Schaeffer et al. 2023 (arXiv:2304.15004):** Metric choice creates apparent emergent
  abilities. Directly confirms our finding that binary math scoring creates apparent phase
  transitions that continuous scoring does not show.
- **FlexMoRE:** Task-dependent optimal rank confirms that no uniform scale works across
  domains.
- **DR-LoRA:** Dynamic rank allocation based on task saliency — the architectural solution
  to our observed problem.
- **Finding #249 (this project):** Two behavioral regimes (FORMAT vs CAPABILITY) — confirmed
  for math, but this experiment shows the model is math-specific, not universal.
- **Finding #250 (this project):** Math phase transition at s*=[4,6] — DOWNGRADED. The
  transition is real for math's binary evaluation but is an artifact of the metric, not
  a fundamental LoRA property. Should be annotated as evaluation-method-dependent.

## Contradicting Evidence

- **Cui et al. 2024 "Phase Transition in Dot-Product Attention":** Mathematical proof that
  phase transitions exist in attention mechanisms. However, this proves transitions exist
  during TRAINING (loss dynamics), not that inference-time scale sweeps produce universal
  transitions. Our finding is compatible: attention has phase transition capacity, but
  whether it manifests depends on the evaluation metric's resolution.
- **Chen et al. 2024 "Sudden Drops in Loss":** Shows "knees" in training loss corresponding
  to syntax acquisition. Again, training dynamics — not inference-time LoRA scale sweeps.
  These papers prove attention CAN exhibit sharp transitions but do not contradict that
  our observed transition is evaluation-metric-dependent.
- **"Standing Committee" in MoE (COMMITTEEAUDIT):** Found domain-invariant expert usage
  (a universal core, not domain-specialized). This contradicts the premise that different
  domains need different specialized adapters. However, this applies to co-trained MoE
  experts, not post-hoc LoRA adapters which ARE domain-specialized by construction.

## Alternative Approaches (with paper references)

1. **FlexMoRE (rank-heterogeneous experts):** Instead of sweeping scale, use different
   adapter RANKS per domain — high rank for reasoning (math), low rank for knowledge
   (legal/finance). Avoids the scale problem entirely by building capacity into the
   adapter architecture.

2. **DR-LoRA (dynamic rank allocation):** Grow adapter rank during training based on
   task saliency scoring. Automatically tailors perturbation magnitude to task demand.
   Could replace our static per-domain scale lookup table.

3. **LoRAuter (arXiv:2602.21222, similarity-weighted fusion):** At inference time, use
   vector retrieval to compute task-aware fusion weights. Each adapter's contribution is
   scaled by input similarity, not a fixed per-domain constant. Handles the bimodal code
   behavior naturally — strong-base prompts get low adapter weight.

4. **RDLC (Router-Driven LoRA Compaction):** Continuous HyperRouter outputs unconstrained
   token-dependent scaling coefficients. Learns the exact perturbation magnitude dynamically.
   Most architecturally aligned with our routing system.

## Implications for Next Experiments

### Eight-Level Proxy Chain (Complete)

1. PPL → MMLU (r=0.08, Finding #236)
2. MMLU → behavioral (Finding #238)
3. PPL improvement → specialization (Finding #240)
4. Cosine → functional disagreement (Finding #240)
5. Domain classification → composition quality (Finding #243)
6. Adapter orthogonality → contrastive value (Finding #245)
7. PPL-optimal scale → behavioral-optimal scale (Finding #249)
8. **Math phase transition → universal phase transition (Finding #252)**

### Architectural Implications

- **Per-domain scale lookup table is insufficient.** Code's non-monotonic behavior means
  no single scale value is optimal — it depends on the specific prompt.
- **The router needs input-dependent scaling**, not domain-dependent scaling. LoRAuter or
  RDLC approaches output per-token scaling coefficients, which would handle the bimodal
  code behavior.
- **Medical adapter needs retraining or replacement.** Max delta +0.028 means the adapter
  learned nothing useful. Before any routing experiment on medical, the adapter must be
  validated on held-out data.

### What Finding #250 Should Become

Finding #250 should be annotated: "Phase transition at s*=[4,6] is evaluation-method-dependent
(binary exact-match + format activation). Not a fundamental LoRA perturbation property.
See Finding #252."

## Recommended Follow-Up (Priority Order)

1. **Input-dependent scaling via RDLC-style router** — Motivation: Finding #252 shows
   domain-level scaling is too coarse (code bimodal, medical null). RDLC's continuous
   HyperRouter produces per-token scaling, handling within-domain variation. Literature:
   RDLC architecture, LoRAuter (arXiv:2602.21222).

2. **Medical adapter quality validation** — Motivation: Finding #252 shows medical
   adapter has near-zero effect (max +0.028). Before any composition experiment involving
   medical, verify the adapter learned something on held-out data. If not, retrain with
   different data/hyperparameters. This is a prerequisite, not an experiment.

3. **Execution-based code evaluation** — Motivation: Finding #252's code data is noisy
   (syntax+recall is crude). Use HumanEval-style execution testing to get cleaner signal.
   Would resolve whether code's non-monotonic scaling is real or evaluation noise.

## References

- Schaeffer et al. 2023, Are Emergent Abilities a Mirage? (arXiv:2304.15004)
- Cui et al. 2024, Phase Transition in Dot-Product Attention
- Chen et al. 2024, Sudden Drops in Loss
- FlexMoRE (Flexible Mixture of Rank-heterogeneous Experts)
- DR-LoRA (Dynamic Rank LoRA)
- LoRAuter (arXiv:2602.21222)
- RDLC (Router-Driven LoRA Compaction)
- Finding #236, #238, #240, #243, #245, #249, #250, #252 (this project)
