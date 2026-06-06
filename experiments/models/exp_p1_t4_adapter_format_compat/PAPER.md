# PAPER.md — T4.5: Pierre Adapter Format Compatibility (AUDIT-RERUN)

**Verdict: KILLED** — K1089 and K1090 are structurally unreachable on the MLX /
Apple Silicon target (they require CUDA runtimes). Per PLAN.md §1 verdict
consistency, an experiment with un-satisfied KCs cannot be marked `supported`.
K1088 and K1091 pass cleanly on synthetic Grassmannian adapters.

## Context

Audit findings (`LOOPHOLE_{FINDING,CODE,METHODOLOGY}.md`) flagged the prior run
as invalid: hardcoded PEFT bypass, vLLM/Unsloth claims via subset fallacy, and a
falsified Grassmannian property tag (measured deviation 0.579 written alongside
`"property": "orthonormal_rows"`). This rerun fixes the code-level bugs and
produces an honest verdict.

## Prediction vs Measurement

| Prediction (MATH.md, updated) | Measured | Verdict |
|---|---|---|
| `peft` is HARD-required (no bypass) | peft 0.18.1 loaded, no bypass path | ✓ |
| `peft.LoraConfig(**cfg)` constructs | Constructed (r=6, α=6.0, target=[q_proj]) | ✓ |
| `peft.PeftConfig.from_pretrained(dir)` round-trips | `r=6`, `target_modules=['q_proj']`, `peft_type=LORA` | ✓ |
| 42 `lora_A.weight` + 42 `lora_B.weight` keys | 42 + 42 | ✓ |
| 6/6 required PEFT fields | 6/6 | ✓ |
| Synthetic Grassmannian max-deviation < 1e-6 | 2.38e-07 | ✓ |
| Metadata round-trip exact | verified_max_deviation bit-exact | ✓ |
| `"property"` tag truthful vs deviation | `"orthonormal_rows"` (deviation < tol) | ✓ |
| K1088 (PEFT LoraConfig) | PASS | ✓ PASS |
| K1089 (vLLM runtime) | SKIP — CUDA unavailable | ✗ UNREACHABLE |
| K1090 (Unsloth runtime) | SKIP — CUDA unavailable | ✗ UNREACHABLE |
| K1091 (Grassmannian metadata) | PASS | ✓ PASS |

## Kill Criteria Results

| K# | Description | Status | Evidence |
|----|---|---|---|
| K1088 | Adapter loads in HF PEFT via LoraConfig | **PASS** | `LoraConfig(**cfg)` + `PeftConfig.from_pretrained` both succeed; 42 A + 42 B keys |
| K1089 | Adapter loads in vLLM runtime LoRA | **SKIP** | CUDA/vLLM unavailable on Apple Silicon; Theorem 3 subset fallacy retracted |
| K1090 | Adapter trains in Unsloth QLoRA pipeline | **SKIP** | CUDA/bitsandbytes unavailable on Apple Silicon |
| K1091 | Grassmannian A in adapter_config metadata | **PASS** | Synthetic QR: deviation 2.38e-07 < 1e-6 tol; `property` tag conditional on deviation |

## Why the verdict is KILLED (not `supported`)

PLAN.md §1 verdict consistency requires `results.json["all_pass"] == True` before
`supported`. K1089 and K1090 cannot return PASS on this platform by construction
— they require CUDA. Reporting them as PASS via string-suffix tricks (what the
prior run did) is the exact antipattern this rerun was commissioned to eliminate.

The correct routing is:
- Mark this experiment `killed` (KCs unreached, not failed-but-finished).
- The follow-up `exp_followup_format_compat_peft_required` already exists in the
  backlog and should run on CUDA hardware. Its KC is exactly the runtime test
  we are skipping here.

## Retractions

- **Theorem 3 retracted.** The subset inclusion `F_vllm ⊆ F_peft` (vLLM imposes
  stricter constraints) does not imply `f ∈ F_peft ⟹ f ∈ F_vllm`. The original
  proof reversed the direction. The retraction is noted in MATH.md.
- **"Orthonormal_rows" metadata tag on drifted weights** (prior run): the tag
  is now conditional on measured deviation. If deviation > tolerance, the tag
  becomes `"drifted_from_orthonormal"`. This prevents downstream systems from
  trusting a false mathematical guarantee.

## Assumptions

1. Synthetic QR-initialized A-matrices are a valid substrate for a FORMAT test.
   Format compatibility concerns file schema + key bijection, not weight
   distribution. Training drift of Grassmannian weights is a separate
   scientific question (interference experiments), not format compat.
2. K1089/K1090 as written in the DB require runtime load on hardware we do not
   have. We report SKIP rather than silently downgrade to a structural check —
   prior run did the downgrade and produced an invalid claim.

## Forward work

- `exp_followup_format_compat_peft_required`: CUDA-hosted rerun, with real
  `PeftModel.from_pretrained(base, adapter)` forward pass + fused-QKV vLLM
  probe. This experiment's KCs K1089/K1090 transfer directly.
- Training-drift interference experiments already exist; they should NOT be
  folded into format-compat again.

## Elapsed
9.34 s on CPU.
