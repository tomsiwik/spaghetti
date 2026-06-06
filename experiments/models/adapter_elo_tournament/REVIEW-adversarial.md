# Peer Review: Adapter ELO Tournament

## Mathematical Soundness

**ELO/Bradley-Terry formulation (MATH.md sections 1-2): Correct.**
The logistic model, update rule, and zero-sum conservation are standard and accurately stated. The worked example in section 6 checks out numerically (verified the E_v1 calculation in the v1-vs-v3 match: 10^((1500-1516)/400) = 10^(-0.04) ~ 0.912, so E_v1 = 1/(1+0.912) = 0.523 -- correct).

**Kendall tau implementation (run_experiment.py lines 360-390): Correct.**
Standard O(n^2) pairwise concordance/discordance counting. No ties handling needed (PPL is continuous, as noted).

**Key assumption failure correctly predicted (MATH.md section 3): Yes.**
The paper explicitly identifies that composition PPL can be non-monotone in individual quality when "pair-specific effects rather than individual quality effects" dominate. This is exactly what happened. The MATH.md deserves credit for predicting its own failure mode.

**One minor issue in MATH.md section 1:** The ELO_SCALE parameter comment says "Standard: 400/ln(10) ~ 173.7 for logistic model" (line 98 in code), but the code uses ELO_SCALE = 400 directly in the expected_score formula (line 338: `10.0 ** ((rating_b - rating_a) / self.scale)`). This is the standard chess ELO formula, not the Bradley-Terry parameterization with s = 400/ln(10). The MATH.md conflates the two: it writes "P(i beats j) = sigma((R_i - R_j) / s) where s = 400/ln(10)" but then uses the 10^(delta/400) formula in the worked example. Both formulations are equivalent (one uses natural log sigmoid, the other base-10), but the notation is sloppy. Not a functional error -- both converge to the same MLE.

## Experimental Design

**Critical flaw: Context adapter bias (correctly identified by the paper).**
All context adapters are "baseline" variants (seed=42, lr=1e-4). The experiment measures how well each variant composes with baseline-family adapters, not composition quality in general. The paper correctly diagnoses this on lines 83-88 of PAPER.md. This is a confound that the MATH.md section 4, assumption 2 should have flagged more prominently as a design risk before running.

**Deterministic matches undermine ELO dynamics.**
The results show IDENTICAL PPL values across round 0 and round 1 for every single match (e.g., medical baseline-vs-low_lr: PPL 10.82 vs 11.63 in both rounds). This means the "2 rounds" add zero information -- every match is deterministic given the same model weights and eval data. The ELO update from round 2 is strictly redundant. The paper claims "2 full rounds for more stable ratings" but the ratings would be identical after 1 round with deterministic outcomes. This is not a bug, but it means the experiment effectively ran 18 matches (6 per domain), not 36. The computational overhead claim of "36 matches in 70s" overstates by 2x.

**Quality spread too narrow to test the hypothesis.**
Math domain: PPL range is 3.07-3.15 (2.6% spread). Code domain: 2.06-2.10 (1.9% spread). With 15 eval batches, the noise floor on PPL estimation could easily be 1-2%. The experiment can only meaningfully test the hypothesis on medical (11.9% spread), where it fails. The code domain tau=1.0 is likely an artifact of the ELO systematically favoring baseline (which happens to also be individually best for code), not evidence of mechanism validity.

**The ELO rankings are IDENTICAL across all 3 domains.**
Medical, math, and code all produce: baseline > high_lr > alt_seed > low_lr with the exact same ELO ratings (1583.8, 1530.4, 1471.9, 1413.9). This is a smoking gun for the context-adapter bias. The ranking reflects "compatibility with baseline-family context adapters" as a universal property, independent of domain. The paper correctly identifies this but undersells its severity -- it means the tournament provides zero domain-specific information.

## Kill Criteria Assessment

**K1 correctly applied.** min(tau) = 0.333 < 0.5. The kill is valid.

**The paper's "reframing" section weakens the kill inappropriately.** PAPER.md lines 112-126 argue that composition quality is "arguably the correct selection criterion" and suggest the experiment should be considered sound for a different purpose. This is post-hoc goal shifting. The hypothesis (line 5 of PAPER.md) explicitly states "rank adapters consistently with their individual (standalone) quality." The experiment failed to do what it set out to do. The finding that "composition quality differs from individual quality" is genuinely useful, but it does not redeem the mechanism for its stated purpose.

**Missing control: standalone PPL ranking as a trivial baseline.**
The experiment should have compared: "just sort by standalone PPL" vs "run ELO tournament." Since standalone PPL is already computed (for ground truth), this is free. The tournament adds 70s of compute to get a WORSE ranking than the trivial approach. This comparison is implicit in the Kendall tau analysis but should have been made explicit.

## Novelty Assessment

ELO for model comparison is well-established (LMSYS Chatbot Arena, arxiv 2403.04132). Applying it to adapter variant selection within a composition framework is a reasonable but incremental extension. The novel contribution would have been demonstrating that composition-based pairwise comparison recovers individual quality rankings -- which it failed to do.

The finding that composition quality diverges from individual quality is not novel in the MoE/adapter literature. It is well-known that adapter interference patterns depend on the specific combination, not just individual quality. The Grassmannian orthogonality guarantee (low mean |cos|) controls catastrophic interference but does not guarantee monotone composition quality ordering.

## Macro-Scale Risks (advisory)

Not applicable given the kill, but noted for context:
- At N=25+ domains with V=4 variants each, the context-adapter bias problem compounds: which variant to use as context for other domains is itself a chicken-and-egg problem.
- Tournament cost O(N * V^2) is manageable but the deterministic-match observation means it could be replaced by a single round-robin, halving cost.

## Verdict

**PROCEED** (confirming the kill as an informative negative result)

The kill is correctly applied. K1 FAIL is genuine and the mechanism does not achieve its stated goal of ranking individual adapter quality via composition comparison. The experiment is well-executed code, the math is sound (minor notation issue only), and the failure mode was correctly predicted by MATH.md section 3.

Key learnings to preserve:
1. Composition PPL is not monotone in individual quality -- a fundamental finding for the Evolve track.
2. Context-adapter homogeneity creates systematic bias -- any future composition-based selection must use diverse or randomized context sets.
3. Deterministic PPL evaluation means repeated tournament rounds are wasted compute.
4. The ELO mechanism itself works correctly (consistent, fast) -- it just measures the wrong construct for the stated goal.

The paper's suggestion of a hybrid score (alpha * standalone + (1-alpha) * composition_elo) or diverse context adapters are reasonable follow-ups, but these would be new experiments, not revisions of this one.
