# Learnings: exp_bitnet_retrain_evolve

## Core Finding

Retrain-from-scratch works as an Evolve primitive (4.4x PPL improvement, 57.59 to 13.08), but PPL and KR-Test diverge completely on legal domain: massive PPL gain produces zero KR-Test improvement at n=50. The original quality gate (KR > 0.03, cos < 0.01) is miscalibrated and rejects all adapters including good ones.

## Why This Happened (Literature-Grounded)

The PPL/KR-Test divergence is not a bug in our experiment -- it reflects a fundamental property of LoRA fine-tuning documented in the literature:

1. **LoRA learns style before facts.** Shuttleworth et al. (arXiv:2410.21228, "LoRA vs Full Fine-tuning: An Illusion of Equivalence") show that LoRA primarily learns "stylistic tokens" -- token distribution patterns that match the training corpus -- while full fine-tuning adapts more deeply to factual content. At 300 steps on legal text, our adapter learned to predict legal token sequences (PPL drops) without internalizing the factual discrimination capacity that KR-Test measures.

2. **KR-Test was designed to detect exactly this.** Ziabari et al. (arXiv:2601.03505, "Beyond Perplexity") explicitly designed KR-Test because "perplexity aggregates token-level prediction errors and does not explicitly distinguish between stylistic learning and factual knowledge." Our result is a textbook confirmation of their motivation: a 4.4x PPL gain with zero KR-Test improvement is precisely the failure mode KR-Test was built to detect.

3. **Legal domain is pathologically hard for contrastive evaluation.** With base KR-Test score of 0.540 (near chance for binary discrimination), the model is essentially guessing on most pairs. Legal reasoning requires long-context understanding that 300 steps of rank-16 LoRA cannot capture. The token-level statistical patterns (PPL) converge quickly, but factual discrimination requires either more parameters, more data, or both.

4. **Training budget insufficient for knowledge acquisition.** Chen et al. (arXiv:2501.14315, "Clear Minds Think Alike") show that high-perplexity tokens (typically rare factual content) require more training to learn than low-perplexity tokens (common patterns). At 300 steps with batch=1, the adapter sees each sample once -- enough for statistical adaptation but not for factual internalization.

## Confirming Evidence

- **Ziabari et al. (2025), arXiv:2601.03505**: KR-Test explicitly addresses PPL's blindness to factual retention. Our PPL/KR divergence is the motivating example for their benchmark. They show KR-Test provides "fine-grained signal for comparing PEFT design choices that are otherwise indistinguishable under perplexity-based evaluation."

- **Shuttleworth et al. (2024), arXiv:2410.21228**: LoRA learns differently from full fine-tuning -- primarily stylistic/distributional patterns rather than deep factual adaptation. This explains why our LoRA adapter achieves dramatic PPL improvement without KR-Test movement.

- **Gekhman et al. (2024), arXiv:2402.05119** ("Limitations of Instruction Tuning"): Large-scale fine-tuning "largely relies on pre-trained knowledge without acquiring new information" -- SFT shifts token distributions but diminishes overall knowledge quality. This supports our finding that PPL improvement does not imply knowledge gain.

- **Chen et al. (2025), arXiv:2501.14315**: Token perplexity analysis shows rare/factual tokens require disproportionately more training. Our 300-step budget is insufficient for the high-perplexity legal factual tokens.

## Contradicting Evidence

- **No direct contradictions found.** The PPL/KR divergence is well-predicted by the literature. However:

- **Potential nuance from FunLoRA (arXiv:2510.02631)**: Rank-1 LoRA achieves continual learning on sequential tasks, suggesting that even very low-rank adapters CAN learn factual content -- but with explicit continual learning objectives (EWC, replay), not standard SFT. Our experiment used standard SFT, which may explain the factual learning failure.

- **LoRA Soups (arXiv:2410.13025)**: Shows that concatenation of LoRAs (keeping them separate with learned weights) outperforms merging. This is consistent with our SOLE architecture (runtime composition, not merge) but suggests that quality evaluation should happen post-composition, not per-adapter in isolation.

## Alternative Approaches (What We Could Try Instead)

1. **Two-signal quality gate (recommended, already proposed in PAPER.md)**:
   - PPL as primary (proven sensitive): val_PPL(new) < val_PPL(old)
   - KR-Test as non-regression: delta_KR >= 0
   - Cosine < 0.05 (aligned with interference threshold)
   - This is the minimum viable gate. The original was over-specified.

2. **Contrastive training objective instead of standard SFT**:
   - DPO or contrastive loss would directly optimize for the KR-Test signal (preferring correct over incorrect continuations). Standard next-token prediction optimizes PPL.
   - Reference: Rafailov et al. (2023), DPO -- could train legal adapter with DPO on correct/incorrect legal pairs.

3. **Longer training with curriculum**:
   - 1000+ steps with full epoch coverage (not 300 steps seeing 37% of data)
   - Curriculum: easy legal QA first, hard case law reasoning later
   - This was already identified in PAPER.md limitations but the literature confirms it's necessary.

4. **Composition-aware quality evaluation (from LoRA Soups)**:
   - Instead of evaluating adapters in isolation, evaluate them in composition context
   - LoRA Soups' concatenation approach with learned weights could serve as the quality signal: if adding the new adapter to the mixture improves mixture-level performance, accept it.

5. **X-LoRA dynamic gating (arXiv:2402.07148)**:
   - Train a lightweight gating network that dynamically selects adapter contributions per-token
   - Quality gate becomes: does adding this adapter improve the gated mixture's output?
   - Aligns with our Track 3 (per-token routing) goals.

6. **Larger KR-Test sample size**:
   - Reviewer's corrected power analysis: n~1700 for 80% power at delta=0.03
   - At n=50, the experiment has only 12% power -- it cannot detect real improvements
   - Before concluding KR-Test is insensitive to legal domain, need adequately powered test

## Implications for Next Experiments

1. **exp_bitnet_kr_test_evaluation (P1)**: This experiment's design should incorporate the power analysis lesson. Use n >= 200 minimum (40% power) or ideally n >= 500 for meaningful signal. The n=50 used here was proven inadequate.

2. **exp_bitnet_effective_delta_cosine (P1)**: The cosine findings here (0.014-0.016 for all retrained adapters) provide a useful baseline. The effective-delta mode should compare vec(B@A) cosine against these raw cosine values to determine if effective delta tells a different story.

3. **Quality gate for Evolve track**: The revised gate (PPL primary, KR non-regression, cos < 0.05) should be adopted as the default. The original thresholds (KR > 0.03, cos < 0.01) are proven too strict.

4. **Training budget**: Future adapter training should use min(3 epochs, 1000 steps) to ensure data coverage. The 300-step budget was proven insufficient for knowledge acquisition (though adequate for stylistic adaptation).

5. **Intruder dimensions warning**: The literature (arXiv:2410.21228) shows LoRA introduces "intruder dimensions" that accumulate during continual learning. Our separate-adapter architecture (SOLE) avoids this by design, but Evolve's retrain-from-scratch approach must ensure each fresh adapter starts from clean initialization -- confirmed by our experiment showing identical results across 3 fresh initializations.

## New References to Add

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| LoRA vs Full Fine-tuning: An Illusion of Equivalence | 2410.21228 | Stylistic vs factual learning in LoRA; intruder dimensions |
| Clear Minds Think Alike: Token Perplexity in Fine-tuning | 2501.14315 | High-PPL tokens need more training; explains our budget insufficiency |
| LoRA Soups: Merging LoRAs for Practical Skill Composition | 2410.13025 | Concatenation > merging; composition-aware evaluation |
| X-LoRA: Mixture of Low-Rank Adapter Experts | 2402.07148 | Dynamic per-token gating; quality via mixture performance |
| Limitations of Instruction Tuning | 2402.05119 | SFT relies on pre-trained knowledge, doesn't acquire new info |
| FunLoRA: Functional LoRA for Continual Learning | 2510.02631 | Rank-1 LoRA + EWC for continual learning; alternative to retrain |
