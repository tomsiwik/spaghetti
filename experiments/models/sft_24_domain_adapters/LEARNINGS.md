# LEARNINGS: exp_sft_24_domain_adapters

## Core Finding

SFT training with a fixed recipe (rank=16, scale=20, 300 steps, lr=1e-4) converges on
all 24 domains without per-domain tuning, confirming that Grassmannian TernaryLoRA +
instruction masking is a universal adapter factory: zero hyperparameter search per domain.

## Why This Happened

**The LIMA hypothesis operates at two timescales.** Finding #216 showed that at
scale=2.0/200 steps, SFT adapters converge to a shared format direction (cos=0.97).
At scale=20.0/300 steps, the adapters have enough capacity and iterations to separate
the domain-specific gradient signal from the shared format component. This is consistent
with Zhou et al. (2305.11206): SFT teaches format first, then domain content with
diminishing marginal returns on data quantity. Our recipe hits the sweet spot where
format is already captured and the remaining gradient signal is domain-specific.

**Response structure predicts improvement magnitude.** The top improvers (math 32.2%,
music 29.5%, engineering 28.4%) have short, structured responses where response-only
masking removes proportionally more noise (instruction tokens). Bottom improvers
(finance 5.9%, psychology 7.1%, legal 8.7%) have long-form, less structured responses
where the instruction-to-response ratio is lower, so masking removes less noise.

**Independent training makes scaling trivial.** Each domain trains independently with
its own frozen Grassmannian A-matrix. No shared optimizer state, no inter-domain
gradients, no catastrophic forgetting. This means N=5→N=24 scaling is purely a matter
of data coverage, not architectural complexity. Lin et al. (2509.20758) confirm that
SFT with controlled learning rate preserves general capabilities — our frozen base +
low-rank adapter is an extreme case of this principle.

## Confirming Evidence

- **LIMA (2305.11206):** 1,000 curated examples sufficient for alignment. Our 400
  samples/domain is in the same regime. SFT data quality > quantity.
- **SFT Doesn't Always Hurt (2509.20758):** Domain-specific SFT with small LR preserves
  general capabilities. Our frozen-base + LoRA is the limiting case — general capabilities
  are fully preserved by construction (base weights never change).
- **Brainstacks (2604.01152):** Frozen MoE-LoRA stacks with null-space projection for
  continual multi-domain learning. Validates the frozen-adapter-stacking paradigm at up
  to 10 stacks on Gemma 3 12B. Our approach uses Grassmannian orthogonality instead of
  null-space SVD, but the same principle: freeze trained adapters, ensure orthogonality.
- **Finding #206:** 5/5 SFT converge with identical recipe (9-32% improvement range
  matches our 5.9-32.2% range at N=24).
- **Finding #54:** Grassmannian skeleton stable at N=24, mean inter-adapter cos=0.024.

## Contradicting Evidence

- **Finding #216:** At scale=2.0/200 steps, SFT adapters have cos=0.97 (format
  dominance). Our scale=20.0/300 steps recipe may still have residual format correlation.
  **B-matrix inter-cosine at the current recipe was not measured in this experiment.**
  This is the critical open question for routing: if B-matrices are still highly
  correlated, ridge routing on SFT adapters may fail for the same reason energy-gap
  routing failed (Finding #187).
- **Finding #262:** NTP preserves GSM8K (+10pp) while SFT degrades (-20pp). SFT
  adapters may trade reasoning capability for format quality. Not measured at N=24.
- **Finding #260/261:** SFT at scale=20 degrades all OOD benchmarks. Our adapters are
  trained at the same scale — OOD degradation risk is real and unmeasured.

## Alternative Approaches

- **Brainstacks residual boosting (2604.01152):** Train multiple stacks per domain,
  each learning what previous stacks missed. Could address the weak-improvement domains
  (finance 5.9%) without changing recipe for strong domains. Adds complexity.
- **TALR — Token-Adaptive Loss Reweighting (2509.20758):** Reweight loss per token to
  reduce general-capability degradation. Could address the OOD degradation risk from
  Finding #260/261 without fully switching to NTP. Would require custom loss function.
- **LIMO (2502.03387):** "Less is More for Reasoning" — curated minimal examples for
  reasoning tasks specifically. Could be applied to the weak math correctness (Finding
  #187: K3 FAIL at 10%) by using higher-quality math examples rather than more data.

## Implications for Next Experiments

1. **B-matrix inter-cosine is the gating measurement.** Before running N=24 composition
   with SFT adapters, measure inter-cosine at scale=20/300 steps. If cos > 0.5, format
   dominance is still present and ridge routing will fail. If cos < 0.2, the adapters
   are sufficiently differentiated. This is a 5-minute measurement that gates a 2-hour
   composition experiment.

2. **The weak-domain pattern is informative, not concerning.** Finance (5.9%) and legal
   (8.7%) also improved least at N=5 (Finding #206: 6% and 9%). These domains have
   longer, more complex responses where instruction masking provides less relative
   benefit. The consistency across N=5 and N=24 suggests this is a structural property
   of the data, not a failure of scaling.

3. **K752 code fix is trivial but important.** The false-positive K752 check
   (val vs first-train-sample) must be fixed to `final_val_loss > base_val_loss` before
   any future training experiment. This is a 1-line code change.

## Recommended Follow-Up

### P0: N=24 Composition with SFT Adapters (exp_n24_composition_proof)
- **Motivation:** This experiment proves the adapters converge. The next question is
  whether SFT adapters compose better than NTP adapters at N=24 (Finding #296 showed
  NTP composition is robust but routing fails on slice domains).
- **Literature:** Brainstacks (2604.01152) validates frozen adapter composition at scale.
  Finding #296's A-orthogonality hypothesis predicts SFT adapters compose if
  Grassmannian skeleton is shared (it is).
- **Gate:** Measure B-matrix inter-cosine first. If format dominance persists, routing
  will fail regardless of composition quality.

### P1: Null-Space Ablation (P=I test)
- **Motivation:** Finding #296 showed null-space preservation fails (36.1%) but
  composition works anyway. If P=I (no null-space projection) produces equivalent
  composition quality, null-space can be dropped — simplifying the pipeline.
- **Literature:** Brainstacks (2604.01152) uses null-space projection but notes it's
  only needed when domains are evaluated "in isolation." Our runtime composition
  may not need it if A-orthogonality is sufficient.
