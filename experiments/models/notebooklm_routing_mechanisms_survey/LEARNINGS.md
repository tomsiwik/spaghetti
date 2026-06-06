# LEARNINGS: Routing Mechanisms Survey

## Core Finding

The survey's top recommendation (SIPS low-rank routing) solves a problem it creates: angular concentration only matters if you first project to low-rank space. At our native d=2560, N=100 poses no packing problem. The real value of R1-R3 is cost reduction and parameter efficiency, not overcoming a fundamental scaling barrier. Meanwhile, Finding #118 (routing moot without specialization) sets an upper bound on how much routing optimization can help.

## Why This Happened

### 1. The SIPS Circularity Is The Key Insight

The reviewer correctly identified that the Welch bound sqrt((N-d)/(d(N-1))) is vacuous when N < d. At d=2560 and N=100, the bound doesn't even apply (N < d means perfect orthogonal packing exists). Angular concentration only emerges when routing is projected to r_route=16, which is SIPS's own architectural choice (arXiv:2601.21349).

**Honest framing:** "Low-rank routing is 6x cheaper in FLOPs; SIPS prevents accuracy degradation from the dimensionality reduction." This is still valuable, but it's an efficiency optimization, not a scaling necessity. Our current full-rank Gumbel-sigmoid router should work fine at N=100 without SIPS.

**Practical implication:** Before implementing SIPS (R1), first test whether full-rank Gumbel-sigmoid maintains oracle-matching quality at N=100. If it does, SIPS is purely a FLOP-saving optimization (nice to have), not a scaling fix (must have).

### 2. Expert Quality Bounds Routing Optimization Upside

Finding #118 (oracle routing = random routing on NTP loss) and Finding #145 (best quality capture 41.5% < 90% threshold) both point to the same conclusion: routing accuracy has bounded upside when expert specialization is insufficient. Hash routing (arXiv:2106.04426) being competitive with learned routing in large-scale MoE confirms this is not unique to our setup.

MoLoRA (arXiv:2603.15965) shows "specialization dramatically beats scale" — a 1.7B with 4 specialized adapters beats 8B. The value is in the adapters, not the router. PHATGOOSE (arXiv:2402.05859) achieves SOTA zero-shot generalization with post-hoc routing over independently trained experts, further confirming expert quality > routing sophistication.

**This does NOT mean routing is useless.** It means: fix expert quality first (deployment track P0), then optimize routing. The survey's recommendations become actionable AFTER exp_task_accuracy_real_benchmarks proves the adapters work.

### 3. Hierarchical Routing (R2) Is The Strongest Recommendation

Cluster-level routing at 96% accuracy (Finding #116) is already proven. Grassmannian MoE (arXiv:2602.17798) provides mathematical collapse guarantees via Bingham distribution concentration. The two-stage approach (coarse cluster + fine Gumbel-sigmoid) reduces the effective routing problem from N=100 to N/C=10-20, staying well within our proven N<=50 range.

This is the only recommendation that builds on TWO proven foundations (our cluster accuracy + published Bingham bounds) rather than introducing new untested mechanisms.

### 4. Three Missing Alternatives Deserve Acknowledgment

The reviewer flagged:
- **Expert Choice routing** (arXiv:2202.09368, Zhou et al.) — experts choose tokens, guaranteeing perfect load balance by construction. Incompatible with our per-sequence granularity (requires per-token), but the concept of inverting the routing direction is worth noting.
- **Mixture-of-Depths** (arXiv:2404.02258) — tokens skip entire layers for dynamic compute. Related to our entropy gating (R3) but operates at layer granularity, which we've killed.
- **Soft MoE** (arXiv:2308.00951, Puigcerver et al.) — fully differentiable soft assignment eliminates discrete selection. Relevant as theoretical contrast but requires all experts active simultaneously (memory-hostile at N>25).

All three are correctly excluded from our architecture (per-layer killed, per-sequence required, memory-constrained), but should be acknowledged as surveyed-and-eliminated rather than simply missing.

## Confirming Evidence

- **L2R** (arXiv:2601.21349): SIPS + low-rank projection reduces routing FLOPs. Confirmed as technically sound, but motivation is efficiency, not necessity.
- **Grassmannian MoE** (arXiv:2602.17798): Bingham distribution provides exponential collapse bounds. Aligns with our Grassmannian skeleton.
- **LD-MoLE** (arXiv:2509.25684): Dynamic-k expert selection is a principled upgrade over binary entropy gating. Per-sequence adaptation untested but theoretically sound.
- **CoMoL** (arXiv:2603.00573): Core-space merging reduces per-expert memory. Incompatible with distinct A_i matrices but cluster-hybrid is plausible.
- **MoLoRA** (arXiv:2603.15965): Per-token routing, 1.7B beats 8B. Confirms specialization > scale.
- **Finding #116**: Cluster-level routing 96% accuracy — hierarchical routing is proven at coarse level.
- **Finding #72**: N=50 Gumbel routing works (86.33% accuracy) — current mechanism scales at least to N=50.
- **Finding #28**: Softmax matches oracle at N=24 — routing quality is already good at current scale.

## Contradicting Evidence

- **Finding #118**: Routing moot without expert specialization — oracle = random on NTP loss. Sets an upper bound on routing optimization value. The survey's R1-R3 recommendations have bounded upside until expert quality is proven at scale.
- **Finding #145**: Best quality capture 41.5% < 90% threshold — routing quality may be fundamentally bounded by expert overlap, not router architecture.
- **Hash Layers** (arXiv:2106.04426): Zero-parameter random hashing competitive with learned routing. Suggests learned routing's advantage is smaller than assumed.
- **SpectR** (arXiv:2504.03454): Training-free spectral routing competitive — router training may not be necessary if experts are high-quality.
- **Finding #115**: Content-aware routing killed at micro scale (26.5% accuracy). Sophisticated routing can fail if the signal isn't in the representation.

## Alternative Approaches

1. **Do nothing until N>50** — current Gumbel-sigmoid is proven to N=50 (Finding #72). Only invest in routing upgrades when we actually hit scaling limits, not preemptively.
2. **Full-rank Gumbel-sigmoid at N=100** — test whether the existing router works at N=100 before adding SIPS complexity. Cost: 256K FLOPs (still 0.16% of inference).
3. **Expert Choice** (arXiv:2202.09368) — inverted routing where experts claim tokens. Perfect load balance but requires per-token granularity.
4. **Soft MoE** (arXiv:2308.00951) — differentiable soft assignment. Elegant but memory-hostile (all experts active).

## Implications for Next Experiments

1. **Routing optimization is PREMATURE.** The deployment track (P0) must prove expert quality first. exp_task_accuracy_real_benchmarks and exp_generation_quality_test are higher priority than any routing experiment.

2. **When routing work resumes, test baseline first.** Before implementing SIPS, hierarchical routing, or dynamic-k, establish that full-rank Gumbel-sigmoid fails at N=100. If it doesn't fail, routing optimization drops to P3.

3. **R2 (hierarchical) is the safest bet.** It builds on two proven foundations and keeps within-cluster N in the proven range (N<=50). R1 (SIPS) and R3 (dynamic-k) introduce new mechanisms.

4. **Memory table needs correction.** The 45.2 MB "runtime buffer" per adapter likely conflates all-loaded vs top-k-active. If only k=2 are active at once, runtime memory is ~90 MB regardless of N. This changes the scaling story entirely.

## Recommended Follow-Up

**No new routing experiment recommended at this time.** Priority is the deployment track:
1. exp_generation_quality_test — does composition produce better text?
2. exp_task_accuracy_real_benchmarks — MMLU/GSM8K/HumanEval with composition
3. exp_real_data_25_domain_adapters — scale to 25 real adapters

Routing optimization becomes relevant only after these prove expert quality. The survey's recommendations (R1-R3) are filed for when N>50 scaling is actually needed.
