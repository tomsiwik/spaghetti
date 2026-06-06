# Peer Review: bitnet_instruction_task_eval (RE-REVIEW)

## Context

This is a re-review after 4 REVISE fixes were applied. The original review identified: (1) degenerate legal domain, (2) confounded NTP comparison table, (3) missing K2 confidence interval, (4) overstated "reverses NTP KILL" framing. All 4 fixes have been verified in PAPER.md.

## NotebookLM Findings

NotebookLM deep review was not performed for this re-review. The prior review's mathematical and experimental analysis remains applicable; this re-review focuses on verifying the 4 fixes and checking for residual issues.

## Fix Verification

### Fix 1: Legal flagged as degenerate, excluded from K1 -- VERIFIED

PAPER.md line 35 marks legal with a warning symbol. Lines 38-41 explain the degenerate behavior (identical outputs across all 15 samples). Lines 50, 55, 57 explicitly exclude legal from K1 counts, changing the denominator from 5 to 4. The K1 result (1/4 = 25%, PASS at 40% threshold) is actually more conservative than the code's 1/5 = 20% because the excluded legal point would have counted as "better" (11.9% > 7.8% base). This is the right call.

### Fix 2: NTP comparison table caveated -- VERIFIED

PAPER.md lines 43-46 add a prompt format confound caveat directly below the summary table. Lines 68-69 add a second warning above the NTP-vs-instruction comparison table. Line 79 adds asterisks marking cross-column deltas as confounded. The legal row is retained in both tables (appropriate -- the data is shown, just flagged) but excluded from K1 claims. The confound is now visible to any reader.

### Fix 3: K2 reports 95% CI and "statistically inconclusive" -- VERIFIED

PAPER.md lines 61-62: "PASS by point estimate, statistically inconclusive" with the 95% binomial CI [0.2%, 32%] explicitly stated. MATH.md lines 127-132 discuss the CI and its implications. The framing is honest.

### Fix 4: "What This Proves" rewritten with honest effect sizes -- VERIFIED

PAPER.md lines 101-105 now state: "effect sizes are small (medical +0.1pp vs instruction base, math = 1 problem out of 15, code = 1 additional valid sample out of 10) and the evaluation is underpowered." Lines 104-105: "reflects a directional shift, not a large-magnitude effect." This is appropriately calibrated.

## Residual Issues

### Non-blocking: HYPOTHESES.yml and FINDINGS.md not updated

The HYPOTHESES.yml evidence entry (line 3170-3178) still says "1/5 metrics (20%)" instead of "1/4 excluding degenerate legal (25%)", still cites "Legal adapter +18.7pp F1 over base under routing" as a headline number, and says "K2 PASS" without the "statistically inconclusive" qualifier. Similarly, FINDINGS.md line 79 uses the pre-fix framing. These are bookkeeping inconsistencies, not errors in the experiment itself, but should be corrected to maintain consistency across the project's documentation.

### Non-blocking: Code still counts legal in K1

The run_experiment.py K1 logic (lines 1053, 1074) includes legal in the domain loop. The paper's interpretation (excluding legal) is layered on top of the raw code output. This is acceptable -- the code preserves the raw data and the paper applies the degenerate exclusion in analysis. No code change needed.

### Non-blocking: K1 OR logic (composed OR routed)

The original review noted that the K1 criterion text says "composed model" but the code passes K1 if EITHER composed or routed passes (line 1143). The paper now reports both conditions separately (composed: 1/4 PASS, routed: 0/4 PASS), which is transparent. The OR logic is stated but readers should note that composed-only K1 would also pass at 25% < 40%.

## Mathematical Soundness

No changes from prior review. The math is correct:
- 1/N composition formula properly implemented
- Orthogonality measurement consistent with project conventions
- Binomial CI correctly computed and reported
- The mechanistic explanation for why instruction adapters survive 1/N (signal concentration, Li et al. 2023) remains hand-waved but is not load-bearing -- the empirical results stand on their own

## Novelty Assessment

No changes from prior review. This is a confirmation of known findings (instruction tuning > NTP for task skills; LoRA Soups; Biderman et al.) in the BitNet-2B ternary + 1/N composition setting. The contribution is replication on a new architecture, not a novel mechanism. This is appropriate for a micro-scale experiment that serves the SOLE pipeline.

## Experimental Design

The 4 fixes address all critical issues from the prior review. Remaining design limitations are acknowledged in the paper's Limitations section:
- Small eval sets (15 math, 10 code, 15 QA) -- acknowledged
- MATH-500 inappropriate for 2B -- acknowledged, GSM8K recommended
- Prompt format confound with NTP comparison -- acknowledged, caveated
- Single seed -- acknowledged, mitigated by prior CV=0.5% finding
- FP16 LoRA not ternary QAT+STE -- acknowledged as conservative

These are micro-scale constraints, not experimental design flaws.

## Macro-Scale Risks (advisory)

1. Keyword F1 will not scale. Use MMLU, HumanEval, GSM8K at macro.
2. Medical adapter improvement over instruction-formatted base is +0.1pp -- near zero. Macro must demonstrate meaningful per-adapter improvement over the prompted base, not just over the NTP-formatted base.
3. Legal-style degenerate behavior (template collapse on short-answer tasks) will recur. Detection and exclusion logic needed in the evaluation pipeline.
4. The code syntax validity metric (ast.parse with truncation fallback) is too lenient. Use pass@k at macro.

## Verdict

**PROCEED**

All 4 required fixes from the REVISE have been applied correctly. The paper now honestly represents:
- Legal domain as degenerate and excluded from claims
- K2 as statistically inconclusive despite passing by point estimate
- NTP comparison as confounded by prompt format
- Effect sizes as small and directional

The core finding -- instruction-tuned adapters are directionally better than NTP adapters for task performance on BitNet-2B with 1/N composition -- is supported at micro scale with appropriate caveats. The experiment answers the prior task_eval KILL by changing the training objective, which was the P0 recommendation. The orthogonality result (|cos|=0.00084) is clean and unaffected by the task evaluation caveats.

**Action item (non-blocking):** Update HYPOTHESES.yml evidence and FINDINGS.md to match the caveated framing in PAPER.md (exclude legal from headline numbers, add "statistically inconclusive" to K2, note prompt format confound).
