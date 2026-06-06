# LEARNINGS: exp_p7_quality_feedback_loop

**Status: KILLED** | Finding #500

## Core Finding

Null-space projection magnitude carries zero quality information (AUC=0.4293, 0.00pp feedback improvement). All 3 kill criteria failed as predicted, confirming the impossibility theorems in MATH.md.

## Why

Projection magnitude decomposes into adapter norm × null-space input energy — neither factor is input-domain-sensitive. The null-space Q^T strips domain features by construction (null(W_v) ⊥ range(W_v^T)), so no function mapping projection magnitude to quality can achieve AUC > 0.5. Confirmed: projection magnitude is constant per adapter across all inputs (47.5–49.0), zero input-dependent variation.

## Implications for Next Experiment

Null-space geometry is an isolation tool, not an information source. Routing and quality signals must come from range(W_v) features (contain domain info) or external signals (TF-IDF, LLM-as-judge). The P7 null-space line is fully closed. LoRAHub (2310.13699) correctly uses task loss, not geometric features — now we know why.
