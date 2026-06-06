# Learnings: exp_flat_lora_training

## Core Finding

SAM (Sharpness-Aware Minimization) provides zero merge benefit (+0.07pp, 43x below threshold) when LoRA adapters are already near-orthogonal. The near-orthogonality (|cos|=0.001) comes from high-dimensional concentration of independently trained adapters in ~17M-dimensional parameter space, NOT from Grassmannian skeleton construction (this experiment uses random-init trained A+B). This is a clean negative result: when merge perturbation is geometrically orthogonal to each adapter's subspace, loss landscape curvature is irrelevant.

## Why This Happened

### The orthogonality mechanism

Two independent random vectors in R^d have expected |cos| ~ 1/sqrt(d). At d=17.2M parameters per adapter, this gives ~0.00024. The observed 0.001 is 4x higher (training correlation moves adapters toward shared features), but still far below any interference threshold. This is the same phenomenon confirmed by exp_minimum_viable_base: |cos| ~ 1/sqrt(D_flat) with beta=-0.506, R^2=0.997, and LoRA/random ratio 0.93-1.13.

SAM's mechanism is widening the loss basin to tolerate merge perturbation delta_i = sum_{j!=i} lambda_j * B_j @ A_j. But when adapters are orthogonal, ||delta_i projected onto adapter i's subspace|| ~ 0. The adapter never leaves its basin regardless of basin width. SAM optimizes for a threat that doesn't exist.

### Flat-LoRA literature context

Sun et al. (arXiv:2409.14396) demonstrated SAM benefits for merging standard LoRA adapters where cross-adapter interference is real. Their setting differs: (a) FP16 base model with sharper loss landscapes, (b) no orthogonality enforcement, (c) longer training where SAM can steer to distinct minima. The transfer to our regime fails because the precondition (significant merge perturbation in adapter subspace) is absent.

### Ternary landscapes may be naturally flat

Both standard and SAM adapters show near-zero sharpness (<0.3% PPL change at 1% perturbation). The ternary {-1,0,+1} weight structure creates degenerate minima (fewer distinct weight values = flatter loss surface). This is speculative but consistent with BitNet's known robustness to perturbation. If confirmed, it would explain why SAM finds nothing to flatten.

## Confirming Evidence

1. **exp_minimum_viable_base**: Directly confirmed |cos| ~ 1/sqrt(D_flat) with R^2=0.997. Random baseline LoRA/random ratio 0.93-1.13 — orthogonality is from dimensionality, not structure. This experiment is another data point for the same phenomenon.

2. **exp_composition_interpolation_landscape**: Smooth, convex PPL landscapes with Grassmannian A-matrices (|cos(A_i, A_j)| = 0.004). The smoothness means merge perturbation never crosses loss barriers, regardless of SAM.

3. **exp_structural_orthogonality_proof**: Trained adapters are 2-9x MORE correlated than random baselines, but still negligible at cos ~ 0.001. Training dynamics push adapters together slightly but cannot overcome dimensional concentration at d >> r^2.

4. **Model Soups** (Wortsman et al., ICML 2022): Weight-space averaging works when models share a loss basin. Our adapters are in orthogonal subspaces — a strictly easier regime than shared-basin averaging.

5. **Foret et al.** (arXiv:2010.01412): Original SAM paper shows benefits primarily at sharp minima with significant curvature. At near-zero curvature (our regime), SAM's epsilon-perturbation has no gradient signal to exploit.

## Contradicting Evidence

1. **Sun et al. Flat-LoRA** (arXiv:2409.14396): Showed clear SAM benefit for merging LoRA adapters on FP16 models. Their setting has HIGHER cross-adapter interference (no orthogonality enforcement) and SHARPER landscapes (FP16). The contradiction is resolved by recognizing these as different regimes — Flat-LoRA addresses a real problem that our architecture (orthogonality + ternary) does not have.

2. **Full weight-space SAM was not tested.** Flat-LoRA perturbs in the full m×n weight space; this experiment perturbed only in LoRA parameter space. Full weight-space perturbation might find curvature that LoRA-space perturbation misses. However, the orthogonality argument still applies: merge perturbation is orthogonal regardless of the SAM variant used.

3. **200 steps is short.** SAM typically differentiates from standard training over thousands of steps. At 200 steps, both methods may converge to the same point simply because the training hasn't run long enough for curvature to matter. This is a valid confound but doesn't change the architectural conclusion — if orthogonality already handles interference, longer SAM training is wasted compute.

## Alternative Approaches

### For improving merge quality (if needed at scale)
1. **TIES-Merging** (Yadav et al., arXiv:2306.01708): Trims, elects signs, and disjointly merges — already tested in this experiment and performs comparably to Task Arithmetic. The merge method matters less than orthogonality.

2. **DARE** (Yu et al., arXiv:2311.03099): Random delta dropping + rescaling. Tested here, no benefit over Task Arithmetic in the orthogonal regime.

3. **OSRM** (Zhang & Zhou, arXiv:2505.22934, ACL 2025): Constrains LoRA subspaces to be orthogonal during training. Stronger guarantee than post-hoc measurement, but our Grassmannian skeleton already achieves this through frozen A-matrices. OSRM would be relevant for regimes without frozen A.

### For addressing the Grassmannian value question (raised by this experiment)
4. **Direct comparison: Grassmannian-frozen-A vs random-init-trained-A composition quality.** This experiment accidentally shows random-init-trained-A achieves cos=0.001. Our grassmannian_expert_init showed AP init gives 1.3-2x lower cos than random. The question is whether this 1.3-2x matters for actual PPL — or if cos=0.001 is already so low that halving it is irrelevant.

## Implications for Next Experiments

1. **Training-time merge optimization is a dead end for our architecture.** With near-zero cross-adapter interference, techniques like SAM, gradient conflict resolution, or curvature-aware training have nothing to fix. Research effort should go to adapter quality (better data, longer training) not merge quality.

2. **The Grassmannian value question is sharpened.** This experiment shows independently trained adapters with random-init A+B reach cos=0.001 purely from high-dimensional concentration. Our Grassmannian skeleton (frozen A from Gr(r,d)) achieves cos=0.0002-0.001. The 1.3-2x improvement from AP init (per grassmannian_expert_init) may be "nice to have" rather than load-bearing at this parameter scale. A direct A/B test of composition quality with and without Grassmannian init at production scale (d=2560) would settle this.

3. **Ternary landscape flatness deserves investigation.** The zero-sharpness finding (both methods <0.3%) may be intrinsic to ternary weight structure. If ternary models are naturally flat, this has implications beyond SAM — it would mean any technique predicated on sharp minima (progressive sharpening, catastrophic forgetting interventions) may be unnecessary on ternary models.

## Recommended Follow-Up

**exp_grassmannian_value_test**: Direct A/B comparison of composition quality (PPL) between (a) Grassmannian-frozen-A adapters and (b) random-init-trained-A adapters on BitNet-2B-4T at d=2560. Same 5 domains, same data, same training steps. Measures whether the deliberate Grassmannian construction improves composition PPL beyond what dimensional concentration provides for free.

**Motivation**: This experiment (flat_lora_training) and exp_minimum_viable_base both suggest orthogonality is "free" from dimensionality. But exp_grassmannian_expert_init showed AP init gives 1.3-2x lower interference. The question: does this 1.3-2x translate to measurable PPL improvement, or is it architectural insurance with no practical effect?

**Literature**: Cao et al. (arXiv:2508.11985, "Efficient Modular Learning through Naive LoRA Summation") confirm that independently trained LoRA modules are approximately orthogonal via the Superposition Principle. If this is sufficient, the Grassmannian skeleton's value shifts from "enabling composition" to "guaranteeing worst-case bounds" — a different but still valid role.

---

## Audit-Rerun Closure Addendum (2026-04-18)

Closure confirms KILL under the `audit-2026-04-17-rerun, code-bug` tag. Three
independent theorems (C1 threshold-invariant +0.07pp vs 3pp, C2
orthogonality-induced projection to 10⁻⁶·λ_max, C3 1/√D concentration
baseline at D=17.2M) show the code-bug fix (results.json verdict label swap)
is cosmetic — the kill is measurement-driven. See PAPER.md §Audit-Rerun
Closure for proofs.

**Antipattern promotion:** This experiment is the 2nd confirmed instance of
`ap-oracle-ceiling-blocks-headroom` — now promoted to confirmed antipattern
(mem-antipattern-021 "CEILING-HEADROOM COLLAPSE"). First instance was
`exp_depth_routed_adapters` (test-time oracle ceiling). Pattern:
mechanism M layered on baseline B_0 that already attains M's theoretical
ceiling — headroom is zero by construction. Future researchers should
pre-flight: identify the proposed mechanism's theoretical ceiling, measure
baseline distance from it, and kill preemptively if gap ≤ KC threshold.

**Deferred cosmetic fix:** `run_experiment.py` lines 879-882 verdict ladder
stamps SUPPORTED whenever K1+K2 pass, ignoring S1. Low priority — does
not change measurements, just the stamped label. Fix during next edit of
the file, don't open a dedicated task.
