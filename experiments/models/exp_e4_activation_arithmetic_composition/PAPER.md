# E4: Activation Arithmetic Composition — Results

## Verdict: PROVISIONAL (smoke) — Strong KILL signal

## Prediction vs Measurement

| Prediction | Measured | Match |
|---|---|---|
| Inter-strategy cos < 0.5 (contrastive fixes E1) | cos=0.76 mean (range 0.19–0.92) | PARTIAL — early layers discriminate, later don't |
| Best injection layer in middle layers [14,28] | Best sweep: L=7 (early) at α=8.0 | WRONG — early layers best, middle/late degrade |
| >2pp GSM8K improvement with ActAdd | -5pp (ActAdd worse than base) | FAIL — steering changes behavior but wrong direction |
| Contrastive extraction separates strategies | Layers 3-9 separate (cos 0.19-0.63), layers 24+ don't (cos>0.87) | PARTIAL — position-dependent |

## Kill Criteria Results

| KC | Threshold | Value | Result |
|---|---|---|---|
| K_struct (proxy) | cos < 0.7 | 0.76 | FAIL |
| K2024 (target) | \|Δ\| > 2pp | 5.0pp (negative direction) | PASS (change detected, but degradation) |
| K2025 (target) | hybrid > NRE by 2pp | not tested | FAIL (deferred) |

## Data

### Phase B: Contrastive Vector Extraction (N=8)
- Decompose vs analogy contrastive vectors extracted at 14 layers
- Early layers show strategy discrimination: L3 cos=0.19, L6 cos=0.53, L9 cos=0.63
- Later layers converge: L24 cos=0.91, L30 cos=0.92
- Improvement over E1: mean cos drops from 0.99 (mean-diff) to 0.76 (contrastive)
- But still above 0.7 threshold — strategies only partially discriminated

### Phase C: Layer × Alpha Sweep (N=5 per condition)
- Base accuracy: 2/5 = 40%
- Best conditions: α=8.0 at L=7, L=14, L=28 all show +20pp (3/5 correct)
- L=35 best at α=4.0 (+20pp)
- Middle layers (L=21) degrade at mid-alpha: -20pp at α=2.0, α=4.0
- N=5 too noisy for reliable signal

### Phase D: Full Eval (N=20)
- Base: 25% (5/20)
- ActAdd (L=7, α=8.0): 20% (4/20) → **-5pp vs base**
- Explicit decompose prompt: 10% (2/20) → **-15pp vs base**

## Mechanism Analysis

### Finding 1: Contrastive extraction partially fixes E1
Contrastive subtraction (strategy - direct) reduces inter-strategy cosine from 0.99 (E1 mean-diff) to 0.76. This confirms Finding #801's prescription: contrastive baselines cancel format signal. However, residual format signal still dominates in later layers (cos > 0.87 at L24+).

**Layer-dependent discrimination**: Strategy-specific information exists in early processing (L3-9) but gets washed out by later layers where representations converge to a shared high-level format regardless of strategy instruction. This is consistent with the residual stream accumulation hypothesis — early layers encode instruction-specific features, later layers converge to task-solving mode.

### Finding 2: Strategy forcing is counterproductive on Gemma 4 E4B GSM8K
The most striking result: **explicit decomposition prompting degrades accuracy by -15pp** (10% vs 25% base). This is NOT an injection artifact — it's the prompt itself. The model performs WORSE when told to decompose step-by-step.

This matches E6 Finding #804: strategy-forcing is antagonistic for reasoning on this model. The model's default reasoning (likely already includes internal decomposition via thinking) is disrupted by explicit strategy instructions that constrain its generation pattern.

### Finding 3: ActAdd changes behavior but degrades accuracy
Activation injection at α=8.0 does change model outputs (60-80% of outputs differ from base). The behavioral effect is real. But it reduces accuracy from 25% to 20%. The steering vector captures "decomposition-like" processing direction, but pushing the model in that direction is counterproductive — same mechanism as Finding 2 but via activation-space.

### Finding 4: Early vs late layer divergence
Strategy discrimination is layer-dependent:
- **Early layers (3-9)**: Encode strategy-specific features (cos 0.19-0.63). Represent the instruction-processing phase.
- **Late layers (24+)**: Converge to shared problem-solving mode (cos > 0.87). Strategy instruction has been "compiled away" into a common reasoning pathway.

This implies ActAdd-style steering must target early layers to leverage strategy-specific information, but early-layer injection has outsized and unpredictable effects on downstream processing (as shown by the noisy sweep results).

## Implications

1. **Strategy adapter path remains blocked**: Both activation-space (ActAdd/E4) and weight-space (Hedgehog/E6) approaches fail for reasoning strategies. The model's default reasoning is superior to any forced strategy.

2. **E11 inherits partial fix**: Contrastive extraction works better than mean-diff (cos 0.76 vs 0.99) but still insufficient. E11 should focus on early layers (3-9) and consider second-order contrastive (strategy_A - strategy_B, not strategy - direct).

3. **Strategy adapters may be impossible for reasoning**: Both prompting and injection degrade accuracy. The model's latent reasoning strategy may be optimized during pretraining in a way that external forcing cannot improve. Strategy adapters may only work for surface behaviors (F#804 precedent: politeness works, reasoning doesn't).

4. **Finding #497 precedent**: Direct prompting dominates all 5 domains — no strategy differentiation at this complexity level. E4 reconfirms for GSM8K specifically.

## Status
PROVISIONAL (smoke, N=20). Method-level failure (strategy forcing degrades accuracy regardless of injection mechanism), but N is small and results are noisy. Strong kill signal.
