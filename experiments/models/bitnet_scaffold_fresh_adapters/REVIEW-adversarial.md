# Peer Review: Scaffold Fresh Adapters

## NotebookLM Findings

Skipped -- experiment is a clean negative result with straightforward math. The mechanism under test (rank-16 LoRA on random base) has well-understood information-theoretic limits. No deep-dive needed beyond manual verification.

## Mathematical Soundness

### Adapter capacity calculation -- CORRECT

The per-layer parameter count is verified:
- 4 attention projections: 4 * (2560*16 + 16*2560) = 327,680. Correct.
- gate_proj + up_proj: 2 * (2560*16 + 16*6912) = 2 * (40,960 + 110,592) = 2 * 151,552 = 303,104. Correct.
- down_proj: 6912*16 + 16*2560 = 110,592 + 40,960 = 151,552. Correct.
- Total per layer: 782,336. Total: 23,470,080. Correct.
- 23.5M / 2.4B = 0.98%. Correct.

### Information-theoretic bound -- APPROXIMATELY CORRECT with caveats

The bound in MATH.md (lines 106-122) is heuristic, not a formal information-theoretic bound. Specifically:

1. The formula `H_scaffold >= H_pretrained + log(P_random) - I_adapter_bits / N_tokens` is presented as a "lower bound" but is not formally derived. It mixes entropy (nats) with adapter information capacity (bits) without consistent units, though the final calculation converts to nats (12.8 nats/token from 9000 bits/token -- this is 9000/ln(2) = 12,984 nats, so 12.8 is close but not exact; correct value is ~12.98).

2. The capacity formula `I_adapter = r * (d_in + d_out) * log2(levels)` assumes each adapter parameter is an independent degree of freedom that can encode arbitrary information. In practice, STE ternary quantization constrains the gradient landscape, so effective capacity is likely lower than this upper bound.

3. The bound gives PPL ~ exp(5.6) ~ 270. Observed values range 186-2887. The lower end (math=215, creative=186) is near or below the bound, which should be impossible if it were a true lower bound. This confirms it is a heuristic estimate, not a rigorous bound. The paper should not call it a "lower bound."

**Verdict:** The directional conclusion is sound -- rank-16 on random base is fundamentally capacity-limited. The quantitative estimate is in the right ballpark. The claim "adapters are operating near their information-theoretic capacity limit" is reasonable for math/creative but overstated for medical/legal (641x and 113x worse than pretrained, well above the ~60x the bound predicts).

### Convergence criterion -- MINOR ISSUE

The convergence check is `last_50 < first_50 * 0.95` (5% reduction). This is a low bar. More importantly, note that the creative adapter on the pretrained base has `converged: false` (loss INCREASED from 1.24 to 1.64), but PAPER.md reports "All 4 scaffold domains converge" for scaffold while omitting that the pretrained creative adapter diverged. This is mentioned in FINDINGS.md but not in PAPER.md. The creative pretrained-vs-scaffold ratio (36.3x) may be understated because the pretrained creative adapter did not converge -- the gap would likely be larger with a properly converged pretrained adapter.

### Cosine orthogonality claim -- CORRECT but trivially expected

Mean |cos| 0.0021 (scaffold) vs 0.0029 (pretrained). The paper interprets this as "scaffold adapters are MORE orthogonal." This is trivially expected: random base provides no shared structure for adapters to align with, while pretrained base provides shared linguistic features that induce weak correlation. The finding is not surprising and does not provide independent evidence for or against the scaffold approach.

## Novelty Assessment

### Prior art alignment

The experiment correctly references FreezeNet (arXiv:2011.14087) and TernaryLM (arXiv:2602.07374). The FreezeNet paper showed that random frozen weights support gradient flow for classification tasks (CIFAR, ImageNet). This experiment extends the finding to language modeling at 2B scale with ternary quantization -- a legitimate novel data point.

### Missing prior art

- **Linear probing literature** (Alain & Bengio 2017, "Understanding intermediate layers using linear classifier probes"): Random features + linear probe is a well-studied setup. The observation that a thin trainable layer on random features learns but plateaus is the standard result. The experiment is essentially rank-16 LoRA probing on random features, which is a modest generalization.

- **Random feature literature** (Rahimi & Recht 2007, 2008): Random projections preserve useful structure for kernel methods but with known capacity limits. The information-theoretic gap is the expected outcome.

Neither of these invalidates the experiment, but the framing could acknowledge that "thin trainable parameters on random base hit capacity limits" is a well-known phenomenon, not a novel discovery. The novel contribution is quantifying the gap at 2B ternary scale specifically.

## Experimental Design

### Does it test the stated hypothesis? YES

The hypothesis is: "Training fresh LoRA adapters directly on a random ternary scaffold will produce usable domain experts." The experiment trains fresh adapters on both pretrained and random scaffold with identical hyperparameters and measures PPL. The kill criterion (5x gap) is pre-registered and clearly evaluated. Clean design.

### Controls -- ADEQUATE

Pretrained base with identical training is the correct control. Same seeds per domain, same data, same optimizer, same steps. The only difference is the base weights. This is a well-controlled experiment.

### Could a positive result be explained by a simpler mechanism? N/A (KILLED)

Since the result is negative, this question applies in reverse: could the negative result be an artifact? Potential confounds:

1. **Training budget (400 steps):** The scaffold starts from much higher loss (10-15 vs 1-3). With the same 400 steps, it covers less of its optimization trajectory. However, PAPER.md correctly notes that even with more steps, closing a 36-642x gap is implausible. The scaffold loss curves show 43-58% reduction in 400 steps -- extrapolating, even 4000 steps would not close the gap to 5x.

2. **Learning rate:** Same lr=1e-4 for both conditions. The scaffold has much larger gradients initially (loss ~15 vs ~3). A scaffold-specific lr schedule (warmup, higher initial lr) might help, but the capacity argument makes this irrelevant -- the ceiling is set by rank-16 capacity, not optimization speed.

3. **Norm matching:** The scaffold uses Frobenius norm matching to the pretrained weights. This is the correct approach for gradient magnitude comparability. Alternative scaffold distributions (structured random, Gaussian instead of ternary) are acknowledged as untested but irrelevant to the capacity argument.

### Composition test on scaffold -- PROBLEMATIC INTERPRETATION

The compose_adapters function (line 348-356) applies `scale_per_adapter = 1/N` to the SUM of all adapter parameters. This means each adapter contributes at 1/N strength. On a random base, the individual adapters already achieve PPL ~200-2900. Under 1/N scaling (N=4, so each at 25% strength), the adapters lose 75% of their correction capacity. The composed scaffold PPL of millions is therefore expected mathematically, not a separate finding. The PAPER.md presents it as if composition is separately catastrophic, but it is simply 1/N dilution of already-weak adapters on a random base. This should be stated more clearly.

## Hypothesis Graph Consistency

The HYPOTHESES.yml node `exp_bitnet_scaffold_fresh_adapters` has:
- K1: scaffold PPL > 5x pretrained PPL per domain. **Tested and KILLED** (36-642x). Correct.
- K2: adapters fail to converge on scaffold. **Tested and PASSED** (all 4 converge). Correct.
- Depends on `exp_bitnet_basefree_exploration` (killed). Correct chaining.
- Blocks nothing (terminal node for this path). Appropriate.
- Status: killed. Correct.

The kill criteria match what was tested. The evidence in HYPOTHESES.yml accurately reflects the results.json data. No discrepancies.

## Macro-Scale Risks (advisory)

Not applicable -- experiment is killed. The base-free scaffold path is dead at two levels. The only macro-scale advisory is:

1. The GaLore scaffold path (`exp_bitnet_galore_scaffold`, supported) is the surviving alternative. If that path proceeds to macro, the key question becomes whether GaLore-grown scaffolds at 2B+ scale maintain adapter composition quality comparable to pretrained bases. The gap quantified here (36-642x for random scaffold) sets the baseline that GaLore must beat.

2. ReLoRA-style iterative training on scaffold remains theoretically viable but untested. The information-theoretic analysis here correctly identifies that the limitation is rank, not trainability -- progressive rank accumulation (merge + retrain) could close the gap. This should be tested if the GaLore path also kills.

## Minor Issues (non-blocking)

1. **PAPER.md line 58:** Creative pretrained adapter has `converged: false` in results.json (loss 1.24 -> 1.64) but the convergence table in PAPER.md only lists the scaffold convergence status for K2, not the pretrained creative divergence. The creative pretrained PPL (5.12) is still better than base (6.99), so the adapter helps despite loss increase -- likely overfitting on this small dataset. This should be noted.

2. **MATH.md line 97:** Reports "Loss reduction 41.4%" for pretrained medical. Results.json shows `ppl_improvement_pct: 35.4%`. These are different metrics (loss reduction from first_50 to last_50 = (2.9249-1.7132)/2.9249 = 41.4% vs PPL improvement). Both are correct but should be labeled clearly. The table mixes them confusingly.

3. **K2 threshold discrepancy:** HYPOTHESES.yml says "loss does not decrease over 1000 steps" but the experiment ran 400 steps. The kill criterion text is inconsistent with the actual training budget. Since K2 passed (adapters converged at 400 steps), this is moot, but the criterion should have been updated to match the actual design.

## Verdict

**PROCEED** (kill is valid and well-documented)

The experiment is a clean, well-controlled negative result. The kill on K1 is unambiguous (36-642x vs 5x threshold). The positive finding (FreezeNet principle at 2B ternary scale, K2 PASS) is a genuine contribution. The information-theoretic capacity analysis is directionally sound even if the "lower bound" label is imprecise. The experimental design is appropriate for the micro scale.

No revisions required. The kill closes the base-free-via-random-scaffold path definitively. The surviving alternative (GaLore scaffold) is correctly identified as the next step.

Non-blocking notes for record-keeping:
1. MATH.md should relabel the capacity estimate as "heuristic estimate" rather than "lower bound" (two observed PPLs fall below it).
2. PAPER.md should note the creative pretrained adapter divergence.
3. K2 threshold in HYPOTHESES.yml says "1000 steps" but experiment used 400.
