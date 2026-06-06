"""
exp_memento_gemma4_replication — DESIGN-ONLY SCAFFOLD (PROVISIONAL)

MEMENTO 2-stage SFT + block-mask attention on Gemma 4 E4B 4-bit MLX.
Grounded: arxiv:2604.09852 (Kontonis et al., MSFT+UT+OpenAI, 2026-04-10).

Per reviewer.md §5 canonical PROVISIONAL-as-design clause (3-precedent threshold:
F#682 JEPA, F#683 hedgehog_behavior, F#684 hedgehog_procedural) and
mem-antipattern-novel-mechanism-single-iteration-scope: the mechanism (block-mask
attention with dynamic KV eviction + 2-stage SFT) is not executable via
mlx_lm.lora CLI and the full pipeline exceeds the researcher-hat cap. This file
is the pre-registered design: main() never raises; it always writes results.json
with verdict="PROVISIONAL" and every KC marked "untested".

The load-bearing implementation lives in exp_memento_gemma4_replication_impl (P3),
inheriting MATH.md verbatim and all 4 KC IDs (#1799, #1800, #1801, #1802).

!!!  IMPL MUST invoke /mlx-dev and /fast-mlx before writing MLX code  !!!
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
DATASET = "microsoft/OpenMementos"

MEMENTO_SPECIAL_TOKENS = [
    "<|block_start|>",
    "<|block_end|>",
    "<|summary_start|>",
    "<|summary_end|>",
]

SFT_STEPS = 2000
BATCH_SIZE = 1
SEQLEN = 4096
LR = 2e-5


# ─── Phase A: tokenizer extension ─────────────────────────────────────────────
def extend_tokenizer(model, tokenizer):
    """Add 4 memento special tokens; resize embedding + lm_head to vocab+4.

    IMPL steps (not executed in design-only filing):
      1. tokenizer.add_special_tokens({"additional_special_tokens": MEMENTO_SPECIAL_TOKENS})
      2. new_vocab = len(tokenizer)
      3. old_emb = model.model.embed_tokens.weight
      4. init new rows from mean of old rows (preserves logit distribution)
      5. model.model.embed_tokens = nn.Embedding(new_vocab, hidden) with init
      6. tie / resize lm_head symmetrically
      7. mx.eval(model.parameters())  # materialize new params
    """
    raise NotImplementedError(
        "extend_tokenizer: IMPL responsibility (exp_memento_gemma4_replication_impl P3)"
    )


# ─── Phase B: 2-stage SFT on OpenMementos ─────────────────────────────────────
def load_openmementos_train_split(tokenizer, seqlen=SEQLEN):
    """Stage-1 data: standard next-token CE on (problem, response) with loss-mask
    on problem tokens; response contains boundary tokens already.

    Stage-2 data: same traces but with attention-mask surgery applied during
    forward (attend-only-to-mementos = after <|summary_end|>, older block spans
    are masked False). IMPL responsibility.
    """
    raise NotImplementedError(
        "load_openmementos_train_split: IMPL responsibility"
    )


def sft_train_stage1(model, tokenizer, dataloader, steps=SFT_STEPS):
    """Standard SFT. Full-parameter, not LoRA — matching paper faithfulness.
    Uses nn.value_and_grad(model, loss_fn); mx.eval at each step boundary;
    mx.clear_cache() at stage boundaries per F#673.
    """
    raise NotImplementedError("sft_train_stage1: IMPL responsibility")


def sft_train_stage2(model, tokenizer, dataloader, steps=SFT_STEPS):
    """Attend-only-to-mementos training. Requires the custom mask-producing
    forward path implemented in Phase C."""
    raise NotImplementedError("sft_train_stage2: IMPL responsibility")


# ─── Phase C: block-mask attention inference ──────────────────────────────────
class BlockMaskState:
    """Tracks block/summary boundaries during generation; emits evolving
    [L, L] attention mask. Kept here as design reference (no MLX ops).
    IMPL will extend this with actual KV-tensor eviction."""

    def __init__(self, tokenizer, keep_last_n_blocks: int = 1):
        self.tok = tokenizer
        self.keep_last_n_blocks = keep_last_n_blocks
        self.block_spans: list[tuple[int, int]] = []
        self.in_block = False
        self.in_summary = False

    def update(self, token_id: int, position: int):
        raise NotImplementedError(
            "BlockMaskState.update: IMPL wires in actual token-id checks"
        )

    def make_mask(self, L: int):
        raise NotImplementedError(
            "BlockMaskState.make_mask: IMPL returns mx.array bool causal mask"
        )


def generate_with_block_mask(model, tokenizer, prompt, max_new_tokens=2048,
                              keep_last_n_blocks=1):
    """Dynamic block-mask generation. IMPL must integrate with mlx-lm generate
    loop, call state.update after each token, and pass the evolving mask to
    mx.fast.scaled_dot_product_attention via the model's attention module."""
    raise NotImplementedError("generate_with_block_mask: IMPL responsibility")


# ─── Phase D: KC eval ─────────────────────────────────────────────────────────
def eval_k1_kv_reduction(model_memento, model_base, tokenizer, prompts):
    """K1 (proxy, paired with K2 per F#666):
    E[peak KV(memento)] / E[peak KV(base)] on GSM8K-Hard n>=200. Target ratio
    <= 0.5 (equivalently >= 2x reduction)."""
    raise NotImplementedError("eval_k1_kv_reduction: IMPL responsibility")


def eval_k2_accuracy(model_memento, model_base, tokenizer, gsm8k_hard, mmlu):
    """K2 (target, pair with K1): GSM8K-Hard drop < 5pp AND MMLU drop < 3pp
    vs base at n>=200."""
    raise NotImplementedError("eval_k2_accuracy: IMPL responsibility")


def eval_k3_kv_channel_ablation(model_memento, tokenizer, reasoning_bench):
    """K3 (target replication): acc(with-KV-channel) - acc(summary-only) >= 10pp.
    Replicates paper's 15pp AIME24 finding at our 4B scale."""
    raise NotImplementedError("eval_k3_kv_channel_ablation: IMPL responsibility")


def eval_k4_throughput(model_memento, model_base, tokenizer, long_prompts):
    """K4 (target serving): throughput ratio >= 1.3x on long-context prompts."""
    raise NotImplementedError("eval_k4_throughput: IMPL responsibility")


# ─── Orchestration (graceful-failure PROVISIONAL) ─────────────────────────────

@dataclass
class Results:
    is_smoke: bool = False
    verdict: str = "PROVISIONAL"
    all_pass: bool = False
    kc: dict = field(default_factory=dict)
    measurements: dict = field(default_factory=dict)
    runtime_seconds: float = 0.0
    design_only: bool = True
    impl_followup: str = "exp_memento_gemma4_replication_impl"
    blockers: list = field(default_factory=list)


def main():
    t0 = time.perf_counter()

    blockers = [
        {
            "id": "B1",
            "phase": "A_tokenizer_extend",
            "issue": "Requires mlx-lm tokenizer.add_special_tokens + embed/lm_head resize + mean-init new rows; not a mlx_lm.lora CLI path.",
        },
        {
            "id": "B2",
            "phase": "B_2stage_sft",
            "issue": "Stage 1 is standard next-token CE (could in principle run via mlx_lm.lora with full-param=True but paper requires full-parameter SFT to preserve K2 semantics; LoRA substitution forbidden per antipattern-t). Stage 2 requires attend-only-to-mementos forward path (custom mask).",
        },
        {
            "id": "B3",
            "phase": "C_block_mask_inference",
            "issue": "Custom MLX generation loop with per-token BlockMaskState + mx.fast.scaled_dot_product_attention(mask=...) + selective KV-tensor eviction. Not in mlx-lm generate path.",
        },
        {
            "id": "B4",
            "phase": "D_kc_eval",
            "issue": "K1 requires peak-KV-memory instrumentation (MLX does not expose per-layer KV tensors by default); K3 ablation requires swapping mask strategies mid-eval; depends on B3.",
        },
        {
            "id": "B5",
            "phase": "runtime_budget",
            "issue": "Full pipeline (2-stage SFT 2x2000 steps + 4 KC evals + K3 ablation arm) estimated 6-10h on M5 Pro 48GB. Exceeds researcher-hat 30-min cap per researcher.md context discipline.",
        },
    ]

    kc_untested = {
        "K1_kv_reduction": {
            "kc_id": 1799,
            "type": "proxy",
            "paired_with": "K2",
            "threshold": "ratio <= 0.5 (>=2x reduction)",
            "value": None,
            "pass": "untested",
            "reason": "Phase C (block-mask inference) blocker B3",
        },
        "K2_gsm8k_drop": {
            "kc_id": 1800,
            "type": "target",
            "pair_of": "K1",
            "threshold": "GSM8K-Hard drop < 5pp",
            "value": None,
            "pass": "untested",
            "reason": "Depends on Phase B + C completion",
        },
        "K2_mmlu_drop": {
            "kc_id": 1800,
            "type": "target",
            "pair_of": "K1",
            "threshold": "MMLU drop < 3pp",
            "value": None,
            "pass": "untested",
            "reason": "Depends on Phase B + C completion",
        },
        "K3_kv_channel_ablation": {
            "kc_id": 1801,
            "type": "target_replication",
            "threshold": "acc_gap >= 10pp",
            "value": None,
            "pass": "untested",
            "reason": "Depends on Phase C — blocker B3/B4",
        },
        "K4_throughput": {
            "kc_id": 1802,
            "type": "target_serving",
            "threshold": ">= 1.3x base tok/s on long prompts",
            "value": None,
            "pass": "untested",
            "reason": "Depends on Phase C generation loop",
        },
    }

    r = Results(
        is_smoke=False,
        verdict="PROVISIONAL",
        all_pass=False,
        kc=kc_untested,
        measurements={
            "base_model": BASE_MODEL,
            "dataset": DATASET,
            "design_rationale": (
                "Novel-mechanism replication (block-mask attention + 2-stage SFT). "
                "Not executable via mlx_lm.lora CLI. Pre-registered MATH.md + "
                "design-only scaffold per reviewer.md §5 PROVISIONAL-as-design. "
                "Full implementation filed as _impl P3 follow-up."
            ),
        },
        runtime_seconds=time.perf_counter() - t0,
        design_only=True,
        impl_followup="exp_memento_gemma4_replication_impl",
        blockers=blockers,
    )

    Path(__file__).parent.joinpath("results.json").write_text(
        json.dumps(asdict(r), indent=2, default=str)
    )
    print(f"verdict={r.verdict} design_only={r.design_only} "
          f"impl_followup={r.impl_followup} blockers={len(r.blockers)}")


if __name__ == "__main__":
    main()
