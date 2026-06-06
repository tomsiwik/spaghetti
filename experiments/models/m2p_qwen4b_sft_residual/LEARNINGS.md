# LEARNINGS.md: SFT-Residual M2P at 4B

## Core Finding
SFT-residual connection (B_applied = B_sft + output_scale * head(z)) fixes the 4B M2P scaling failure. After three consecutive collapses (v1: -0.125, v5: -0.187), quality_ratio=1.175 at 4B — matching SFT quality (74.4% vs 73.0%, difference not statistically significant at p=0.62, n=500).

## Why It Works
Zero-init heads guarantee B_applied = B_sft exactly at step 0 (Theorem 1, VERIFIED: init_qr=1.00 to 4 decimal places). This is a structural guarantee: failure is impossible at init because the M2P adapter IS the SFT adapter. Training perturbs from a known-good starting point. Mechanism: residual learning (He et al., 2016) applied in weight space. The intrinsic dimensionality of the correction ΔB is lower than the full B (Aghajanyan et al., arXiv:2012.13255), making it learnable from 1000 steps.

## Key Caveat
"Exceeds SFT" framing is unsupported. 74.4% vs 73.0% = z≈0.50, p≈0.62. Correct framing: M2P matches SFT quality while previous architecture actively degraded. The structural claim (quality_ratio > 0) is proven; the superiority claim requires larger n.

## Parameter Note
808M parameter M2P encoder for a 4B base model. Not a problem for this experiment, but relevant context: VeRA-style reduction (Finding #380, killed at rank-4) needs to be re-approached at 4B if parameter overhead becomes a deployment constraint.

## Implications for Next Experiment
SFT-residual is now the proven baseline for M2P at any scale. The composition experiments (exp_m2p_composition_n5_qwen3) can now be scaled to 4B using this pattern. The 0.6B v4 warm-start precedent (quality_ratio=1.433) suggests longer training or more data may close the significance gap. Priority: run composition verification at 4B scale using SFT-residual.
