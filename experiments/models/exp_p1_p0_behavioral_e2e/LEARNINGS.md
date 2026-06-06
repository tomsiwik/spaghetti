# LEARNINGS.md — P0: Behavioral E2E Quality

**Status: KILLED — Finding #457**

## Core Finding

Domain adapters (rank-4 LoRA, 1000 steps, MMLU MCQ data) do not improve behavioral vocabulary
for open-ended explanatory queries on a capable base. Three of five domains (code, legal, finance)
showed adapted performance WORSE than base.

## Why

Two compounding causes:

1. **MCQ format-register mismatch.** MMLU training teaches "select A/B/C/D" (concise answer
   style). Open-ended explanatory evaluation rewards vocabulary density. MCQ fine-tuning actively
   reduces response verbosity → vocabulary scores DROP.

2. **δ_D requires a base capability gap.** Q_pipeline = ρ_D × δ_D. For personal adapters
   (Finding #436), δ = 0.76 because base had ZERO style compliance (100% gap). For domain
   vocabulary, Gemma 4 already covers the domain in pretraining → δ_D ≈ 0. PPL reduction
   (26.3%) does NOT imply behavioral vocabulary improvement (r=0.08, established).

Medical domain exception: base mean was lowest (1.4 terms/response), adapted rose to 2.1.
This is exactly the δ_D > 0 condition — a real base gap existed.

## Impossibility Structure

For MCQ-trained adapters on a capable base: δ_D ≤ 0 possible. Routing accuracy ρ_D is
irrelevant when the adapter itself provides no behavioral benefit. Fix requires either:
- (a) Adapters trained on open-ended explanation data (format-matched to evaluation)
- (b) Target domains where the base model demonstrably fails baseline tasks (δ_D > 0 guaranteed)

**Testable prediction:** δ_D ≥ 0.5 iff baseline_accuracy(domain, base) < 50%.

## Implications for Next Experiment

Next behavioral experiments must select domains/tasks where Gemma 4 fails WITHOUT adaptation.
Vocabulary rubric on capable-base domains will always show δ_D ≈ 0. Medical is the one confirmed
domain with a real base gap — a targeted experiment with open-ended medical data (not MMLU MCQ)
would test whether the behavioral gap can be closed.
