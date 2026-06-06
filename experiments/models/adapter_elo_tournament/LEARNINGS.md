# LEARNINGS: Adapter ELO Tournament

## Core Finding

ELO tournament ranking over composition PPL does **not** recover individual adapter quality
(min Kendall tau = 0.333 < 0.5). The tournament instead measures **composition compatibility**
with the fixed context adapter set — identical rankings across all 3 domains (medical, math, code)
confirm this is a systematic bias, not a domain-specific signal.

## Why This Happened

The core failure is **non-monotone composition quality**: an adapter's contribution to composed
PPL depends on pair-specific B-matrix interactions with context adapters, not on its standalone quality.

1. **Context adapter homogeneity creates systematic bias.** All context adapters were "baseline" variants
   (seed=42, lr=1e-4). The baseline test variant shares initialization characteristics with context
   adapters, producing more compatible interference patterns. This explains why baseline won every
   match in every domain despite not being individually best in medical or math.

2. **Pair-specific interference dominates individual quality.** The Grassmannian A-skeleton guarantees
   low mean |cos| (~0.00125), preventing catastrophic interference. But it does NOT guarantee monotone
   composition quality ordering. Adapters with similar hyperparameters produce B-matrices with more
   compatible interference patterns — a pair-specific effect that ELO captures faithfully.

3. **Quality spread too narrow to overcome bias.** Math (2.6%) and code (1.9%) domains have PPL spreads
   near the noise floor. Only medical (11.9%) had sufficient spread to potentially distinguish quality
   from compatibility — and it still failed (tau = 0.333).

4. **Deterministic PPL makes repeated rounds redundant.** Same weights + same eval data = identical
   outcomes. The "36 matches" are actually 18 unique comparisons. ELO dynamics (update from surprise)
   add nothing when outcomes are perfectly predictable.

## Confirming Evidence

- **Unchosen Experts Can Contribute Too (Self-Contrast MoE):** Increasing activated experts does not
  monotonically improve output quality — different experts don't natively act synergistically. Confirms
  composition quality is not additive in individual quality.

- **Model merging with SVD (Stoica et al., 2025, ZipIt/KnotIt):** Independently trained LoRA adapters
  exhibit "limited alignment" that fundamentally hinders parameter merging. Pair-specific alignment
  matters more than individual quality.

- **Our own exp_softmax_router_scaling:** Confirmed that within-cluster adapter misrouting is
  "quality-benign" (<1.2% oracle gap). Different adapters in the same semantic cluster produce
  similar PPL — individual quality differences are small relative to cluster membership effects.

- **Our own SOLE review finding:** Math-medical adapter pair hits cosine similarity 0.703 (3,500x
  worse than theoretical 0.0002). Interference is pair-specific and domain-dependent, not predictable
  from individual quality alone.

## Contradicting Evidence

- **LMSYS Chatbot Arena (arxiv 2403.04132):** ELO successfully ranks whole models via pairwise human
  preference. But Arena compares independent models on the same task, not compositions of adapters.
  The key difference: Arena items don't interact (no composition), so individual quality IS the
  measured quantity. Our setting introduces pair-specific interaction effects absent in Arena.

- **Code domain tau = 1.0:** In the one domain where baseline happens to also be individually best,
  ELO and quality agree perfectly. This suggests that with diverse enough context adapters (breaking
  the homogeneity bias), ELO might recover quality. However, the reviewer correctly flags this as
  likely coincidental (baseline favored by context, AND happens to be best for code).

## Alternative Approaches (Literature-Backed)

1. **Direct standalone evaluation (trivial baseline).** Sort by standalone PPL — already computed,
   zero additional cost. The experiment showed this trivially outperforms ELO for individual quality
   ranking. For standalone quality, just measure standalone quality.

2. **Canary queries for quality monitoring (SOLE project).** Per-expert curated test queries achieve
   2.0% FNR for detecting degradation within composition. Manual curation required but proven reliable
   where automated proxies (cosine gating: 33.8% FNR, KL divergence: rho=-0.7) all failed.

3. **PPL-probe weighting (SOLE project).** K+1 forward passes on 10 probe examples per query measures
   answer-conditioned perplexity, achieving r=0.990 correlation with oracle selection. More expensive
   than ELO but measures the right construct.

4. **LoRAuter (arxiv 2602.21222).** Dataset-centric retrieval — embeds adapter training examples in
   vector DB, retrieves nearest neighbors via nucleus sampling. Bypasses parameter-space assessment
   entirely by reasoning about training data similarity.

5. **Hybrid score.** alpha * standalone_PPL + (1-alpha) * composition_ELO. Paper's own recommendation.
   Would require diverse context adapters (not all baseline variants) to deconfound the composition
   signal. Not yet tested.

## Implications for Next Experiments

1. **Individual adapter quality: just measure it directly.** No tournament needed. Standalone PPL or
   domain-specific task metrics (MMLU, HumanEval, GSM8K) are the correct instruments.

2. **Composition quality IS pair-specific.** This aligns with three prior findings:
   (a) softmax_router_scaling showing semantic clustering matters more than exact selection,
   (b) generation_quality_test showing PPL != generation quality,
   (c) the SOLE review finding 3,500x cosine violations for semantically related domains.
   Any future composition evaluation must account for the specific adapter ensemble.

3. **Evolve track needs a different selection mechanism.** The original goal was "ELO selects best
   variant for evolution." Since composition quality != individual quality, selection must be
   multi-criteria: standalone quality first (retrain + evaluate), then composition verification
   as secondary check.

4. **The "composition compatibility" signal is real but confounded.** Future work could deconfound
   by using diverse/randomized context adapters. But the practical value is unclear — if the
   router already achieves oracle quality (softmax_router_scaling), composition compatibility
   differences may be in the noise.

## Recommended Follow-Up

No new experiment recommended from this kill. The Evolve track's selection problem is better
addressed by direct quality measurement (standalone PPL + task metrics) than by tournament
mechanisms. The composition quality signal, while real, requires deconfounding work (diverse
context adapters) that is lower priority than the P0 deployment track experiments.

The most relevant next step is already on the roadmap: exp_task_accuracy_real_benchmarks
will establish whether standalone quality differences translate to meaningful task performance
differences — which is the question ELO was ultimately trying to answer indirectly.
