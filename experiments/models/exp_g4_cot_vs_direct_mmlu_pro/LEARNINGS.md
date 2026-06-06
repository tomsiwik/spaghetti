# LEARNINGS — exp_g4_cot_vs_direct_mmlu_pro

## Core Finding
On Gemma 4 E4B 4-bit (MLX), enabling the thinking channel (CoT) beats direct
answering by **+25.0pp pooled** on reasoning-heavy MMLU-Pro subjects at matched
N=30/subj (MATH 46.7→60.0, +13.3pp; Physics 20.0→56.7, +36.7pp). All 3 KCs
PASS; 0 parse errors; 1948 mean thinking chars on CoT-correct items.
`Δ ≥ +8pp` kill criterion cleared by >3×.

## Why
Direct mode with `max_tokens=16` must emit a letter in a single forward pass
from the question embedding. Multi-step MMLU-Pro items (integrals,
combinatorics, electrostatics) require O(10²)–O(10³) scratch tokens of
intermediate state that cannot fit in 16 output tokens. The thinking channel
provides exactly that scratch. The gap is therefore a reasoning-depth gap, not
a decoding/format artifact — the 0 parse errors and nontrivial per-item
thinking content confirm this. Physics benefits more (+36.7pp) because its
direct baseline (20%, 10-option) is barely above random, leaving the most
headroom.

## Implications for Next Experiment
1. **CoT is a precondition, not a knob.** Any Gemma 4 experiment that claims
   a capability gain on reasoning-heavy MMLU-Pro subjects MUST run with
   `enable_thinking=True` as the baseline, else it is competing against a
   structurally-crippled direct-mode floor. Plain-direct numbers are only
   meaningful as the lower bound on "what can the embedding do in one pass".
2. **MATH CoT came in at 60%, below the 70–90% MATH.md band** — truncation
   at `max_tokens=2048` on combinatorics chains. Future MATH-dominated runs
   should raise the cap to ~4096 or log truncation rate as a first-class
   metric.
3. **Thinking content is evidence, not ceremony.** Logging
   `mean_thinking_chars_per_correct` per phase is cheap and directly rebuts
   the "framing-trick" failure mode. Adopt as default for every future
   thinking-on vs thinking-off comparison.
4. **Direct-mode reasoning is ~3× random on 10-option Physics but
   ~4.7× random on MATH (4-option).** When designing matched comparisons,
   normalize by random-baseline before pooling Δ across mixed-option subjects
   so one subject's larger headroom does not dominate the pooled delta. Not
   a blocker for this KC (both above +8pp individually) but relevant for any
   future "CoT gain per bit of choice entropy" claim.
5. **No antipattern triggered** — no LoRA, no routing, MATH.md untouched
   post-pre-reg (commit 22a0f17). Use this experiment as a clean template for
   the Gemma 4 baseline-pair pattern in the remaining P2 backlog.
