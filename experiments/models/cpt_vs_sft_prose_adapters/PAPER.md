# CPT vs SFT Adapters: Knowledge Injection via Continued Pre-Training

## Theorem
CPT (causal LM on raw domain text) allocates full adapter capacity to domain
knowledge I_knowledge, while SFT splits capacity between I_format and I_knowledge.
For prose domains where format training overwrites base model capability, CPT
should produce better behavioral quality.

## Predictions vs Measurements

| Prediction (from MATH.md) | Measured | Match? |
|---------------------------|----------|--------|
| P1: CPT improves legal factual recall >= 15% vs SFT | CPT 0.097 vs SFT 0.056 = +74.2% | YES (but see note) |
| P2: CPT improves medical factual recall >= 15% vs SFT | CPT 0.290 vs SFT 0.291 = -0.2% | NO (tie, not improvement) |
| P3: CPT training converges | Medical YES, Legal NO (loss flat 3.09->3.10) | PARTIAL |
| P4: CPT coherence >= 80% | 100% on both domains | YES |
| P5: Training < 2 hours | 127 seconds total | YES |

## Hypothesis
CPT adapters trained with causal language modeling on raw domain text will inject
domain knowledge and improve prose domain behavioral quality by >= 15% vs SFT
adapters on legal and medical domains.

**Verdict: KILLED on K3 (convergence), but findings are informative.**

## What This Experiment Is
A controlled comparison of two LoRA training objectives on the same model
(BitNet-2B-4T), same adapter architecture (rank-16 TernaryLoRA with Grassmannian
init), and same domain data (legal, medical). The ONLY variable changed is the
training data format:
- **SFT:** `### Instruction:\n...\n### Response:\n...` (instruction-response CLM)
- **CPT:** Raw response text only (unconditional CLM)

## Key References
- Gururangan et al. 2020 (2004.10964): Domain-Adaptive Pre-Training
- Zhou et al. 2023 (2305.11206): LIMA -- Less Is More for Alignment
- Ke et al. 2023 (2301.09515): Continual Pre-Training of Language Models

## Empirical Results

### Training
| Domain | Time | First 50 Loss | Last 50 Loss | Converged? |
|--------|------|---------------|-------------|-----------|
| Legal  | 68s  | 3.087         | 3.095       | NO (0.3% increase) |
| Medical| 59s  | 2.271         | 1.846       | YES (18.7% decrease) |

Legal CPT shows loss oscillation around a floor (~3.0), not divergence. The
convergence criterion (last_50 < first_50 * 0.95) is too strict for data with
high per-sample variance. The legal training data has heterogeneous complexity
(simple Q&A to multi-paragraph legal analysis), causing high loss variance.

### Behavioral Evaluation (Factual Recall)
| Domain  | Base  | SFT   | CPT   | CPT vs SFT | CPT vs Base |
|---------|-------|-------|-------|-----------|------------|
| Legal   | 0.098 | 0.056 | 0.097 | +74.2%    | -0.5%      |
| Medical | 0.263 | 0.291 | 0.290 | -0.2%     | +10.5%     |

### Key Finding: CPT Preserves, SFT Destroys (for Legal)
The critical result is NOT that CPT adds knowledge -- it is that CPT does NOT
destroy the base model's prose capability on legal text, while SFT DOES.

- **Legal SFT:** -42.9% vs base (0.056 vs 0.098). SFT format training overwrites
  the base model's legal reasoning with response format patterns.
- **Legal CPT:** -0.5% vs base (0.097 vs 0.098). CPT at scale=20 is essentially
  a no-op for legal -- the adapter learned so little that the base model dominates.
- **Medical:** Both SFT (+10.7%) and CPT (+10.5%) improve identically over base.
  Medical text is more structured (clinical facts), so both objectives capture
  similar information.

### Coherence
100% coherent for both CPT domains. No degenerate output observed.

## Interpretation

### Why Legal CPT "Won" (It Did Not Actually Win)
The legal CPT adapter did not inject knowledge. It did not converge, and its
factual recall matches the base model exactly (0.097 vs 0.098). It "beats" SFT
only because SFT actively degrades legal prose quality.

The real finding is: **SFT on legal text is harmful at scale=20**, confirming
Finding #209 with the causal mechanism now clear. SFT teaches the adapter to
produce legal-formatted responses, overwriting the base model's broader legal
knowledge with narrow format patterns.

### Why Medical Is a Tie
Medical SFT text is more structured (factual Q&A: "What type of delusion...?
-> Non-bizarre delusion"). The instruction-response format closely mirrors the
domain's actual format. For medical, format IS a reasonable proxy for knowledge,
so SFT and CPT learn similar information.

### The Capacity Allocation Theory Partially Confirmed
- For legal (high format-knowledge divergence): CPT preserves base, SFT destroys
- For medical (low format-knowledge divergence): CPT = SFT

This is consistent with the theory: when format and knowledge are different
(legal prose vs legal Q&A format), allocating capacity to format wastes it.
When format and knowledge align (medical Q&A), both objectives are equivalent.

## Kill Criteria Assessment
- **K1 (#672):** PASS. CPT worse on 1/2 domains (medical, by 0.2% -- effectively tied)
- **K2 (#673):** PASS. 0% incoherent output
- **K3 (#674):** FAIL. Legal CPT did not converge (loss oscillating, not diverging)
- **OVERALL:** KILLED on K3

## Limitations
1. **n=10 prompts per domain.** Small sample, high variance.
2. **Convergence criterion too strict** for heterogeneous data. Legal loss was
   stable, not diverging -- a looser criterion (e.g., last_50 < first_50 * 1.05)
   would pass.
3. **CPT used the SAME data as SFT** (just stripped format). A proper CPT
   experiment would use a LARGER corpus of raw domain text (Gururangan used
   millions of tokens; we used ~80K).
4. **Scale=20 for all.** Finding #249 says legal should use scale=4. At scale=4,
   the SFT degradation would be smaller, narrowing CPT's advantage.

## What Would Kill This
1. At scale=4 (format regime), if SFT and CPT are equivalent on legal, then
   the finding reduces to "don't use high scale on legal" (already known).
2. With 10x more CPT data, if legal CPT still doesn't converge, the approach
   is fundamentally limited by data quantity at this adapter rank.

## Findings for the Project

**Finding:** CPT does not inject domain knowledge at 200 iters / 80K tokens /
rank-16. Instead, it acts as a near-identity adapter, preserving base model
capability. SFT actively damages legal prose at scale=20 (confirming Finding #209
with causal mechanism identified).

**Implication for SOLE architecture:** The problem with legal adapters is not
the training objective (CPT vs SFT) -- it is that legal text at this data scale
cannot be meaningfully compressed into a rank-16 LoRA adapter. The two-regime
solution (scale=4 for legal, scale=20 for math/code) remains the correct
approach from Finding #249.

**New question:** Would CPT with 10-100x more data (raw legal corpus from
HuggingFace) produce adapters that genuinely inject knowledge? This is a
data-scale question, not a training-objective question.
