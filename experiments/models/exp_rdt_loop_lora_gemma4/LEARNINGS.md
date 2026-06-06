# LEARNINGS — exp_rdt_loop_lora_gemma4

## Summary

Smoke-mode build of RDT (Relaxed Recursive Transformer, Bae 2024) loop-indexed
LoRA + LTI-injection on frozen Gemma 4 E4B-it-4bit, layers 12–20, N_loops=6,
r=16. **Verdict: PROVISIONAL** — K1743 (partition-QR orthogonality at init)
and K1744 (ρ(A)<1 across 50 Adam steps) pass; K1740/K1741/K1742 (target-metric
claims on GSM8K-Hard/MMLU/saturating-exp fit) deferred to full-scale follow-up.

## Reusable observations

1. **Partition-QR Grassmannian init scales to Gemma 4 per-loop-index families.**
   max |cos| = 3.75e-8 across 18 projection family instances (9 layers × {v_proj,
   o_proj}). Within Higham float32 bound; 7 orders below the 0.1 kill threshold.
   Extends F#562 from single-projection Pierre defaults to N-loop per-projection
   families. Reusable rule: **any N-adapter orthogonality claim at r=16 on
   Gemma 4 native in-dims is structurally safe under partition-QR**.

2. **LTI + LoRA bundle training via `nn.value_and_grad` works cleanly.**
   Bundle contained 108 LoRADelta + 6 LTIInjection modules; Adam update at 50
   steps + batch 2 + seqlen 32 runs in 1.95s total including 4-bit Gemma 4 E4B
   load. No OOM, no mx.eval discipline issues.

3. **Fixed-A convention via `_`-prefixed attr keeps A out of
   `trainable_parameters()`.** `self._A_fixed = A_init` + `@property A` + only
   `self.B = mx.zeros(...)` as trainable means Grassmannian init is preserved
   during training (not drifted by Adam). This is the MLX-idiomatic way to
   freeze a sub-parameter without calling `module.freeze(keys=[...])` on each
   LoRADelta.

4. **Smoke-loss didn't exercise LTI dynamics.** max ρ(A) stayed at
   exp(-exp(0)) ≈ 0.3679 through 50 steps. LTI param gradient flows through
   the bundle but the effective Adam step on log_A/log_dt was below float32
   resolution for changing ρ at s=0. **Implication for full-scale**: use a
   real task loss (GSM8K+MATH) and confirm LTI params move; don't assume
   smoke-tested dynamics.

## Follow-up ticket (for P=1 or P=2)

`exp_rdt_loop_lora_gemma4_full`: same architecture, full-scale training on
GSM8K+MATH, eval on GSM8K-Hard (T=1..6) + MMLU (T=3), fit saturating-exp.
Depends on current experiment's architecture wiring (now validated).

## Antipatterns checked and clear

- No composition-bug (single-loop deltas, no ΣA·ΣB path).
- No tautological routing (loop index is a schedule, not a learned decision).
- No unsafe LORA_SCALE (α=2, scale=0.125).
- No shutil.copy / hardcoded pass.
- is_smoke=true honored; verdict stays PROVISIONAL.
- F#452/F#453/F#1564 family check: K1744 is NOT a reproduces-or-refutes of
  F#667; it EXTENDS F#667's primitive-only claim to the loop+LoRA composition
  context, which is novel scope.

## Acknowledged limitations

- Surrogate forward pass is slice-padded LoRA-v (not full DecoderLayer block);
  only the adapter-wiring surface is smoke-verified. Full-scale must
  monkey-patch or subclass DecoderLayer to plumb per-loop LoRA into
  `self_attn.v_proj` and `self_attn.o_proj` call sites.
- K1744 static-valued under smoke loss — dynamical guarantee still pinned to
  F#667 Theorem 1 (exact-arithmetic ρ<1), not directly exercised here.
