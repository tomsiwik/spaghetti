#!/usr/bin/env python3
"""Train a reasoning LoRA expert via distillation from DeepSeek-R1 traces.

Hard distillation: QLoRA rank-16 on Qwen2.5-7B with <think> token reasoning
traces. Uses rasbt/math_distill dataset (DeepSeek-R1 671B generated traces
on MATH problems).

Pipeline:
  1. Download/prepare rasbt/math_distill dataset
  2. Format as chat messages with <think>...</think> reasoning traces
  3. QLoRA fine-tune Qwen2.5-7B (same config as pilot50 experts)
  4. Save reasoning adapter

Usage (on RunPod):
    cd /workspace/llm
    python experiments/models/reasoning_expert_distillation/train_reasoning_expert.py

Expected runtime: ~30-45 min on RTX 4090 (24GB)
Expected cost: ~$0.25 (RunPod 4090 at $0.34/hr)
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ── Monkey-patch set_submodule for PyTorch builds missing it ──────────────────
import torch
if not hasattr(torch.nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: "torch.nn.Module") -> None:
        atoms = target.split(".")
        mod = self
        for item in atoms[:-1]:
            mod = getattr(mod, item)
        setattr(mod, atoms[-1], module)
    torch.nn.Module.set_submodule = _set_submodule
    print(f"[patch] set_submodule monkey-patched onto torch {torch.__version__}")

# ── Configuration ─────────────────────────────────────────────────────────────

# Paths (RunPod layout)
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/Qwen2.5-7B")
HF_CACHE = os.environ.get("HF_CACHE", "/workspace/hf_cache")
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # /workspace/llm
OUTPUT_DIR = REPO_ROOT / "micro" / "models" / "reasoning_expert_distillation"
ADAPTER_DIR = OUTPUT_DIR / "reasoning_adapter"

# Dataset config
DATASET_NAME = "rasbt/math_distill"
DATASET_CONFIG = "math_train"  # Config with DeepSeek-R1 reasoning traces
DATASET_SPLIT = "train"
# Use the 12K math training problems (not the 500 test problems)
# Filter: only examples with non-empty thinking traces
MAX_TRAIN_EXAMPLES = 10000  # Cap to control training time
MAX_SEQ_LENGTH = 2048  # Reasoning traces can be long

# QLoRA config (matches pilot50 for composition compatibility)
LORA_RANK = 16
LORA_ALPHA = 16
LORA_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# Training config
TRAIN_STEPS = 500           # More steps than pilot50 (300) due to reasoning complexity
BATCH_SIZE = 1
GRAD_ACCUM = 4              # Effective batch = 4
LR = 1e-4                   # Lower LR for reasoning (longer traces, more complex signal)
WARMUP_STEPS = 20
SEED = 42


def log(msg: str):
    """Timestamped logging."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare_reasoning_dataset(tokenizer, max_examples: int = MAX_TRAIN_EXAMPLES):
    """Load and format rasbt/math_distill for reasoning distillation.

    The dataset has columns:
      - problem: math question text
      - gtruth_answer: ground truth answer
      - message_thinking: DeepSeek-R1's chain-of-thought reasoning trace
      - message_content: DeepSeek-R1's final answer

    We format as chat messages with <think>...</think> tags wrapping the
    reasoning trace, following the DeepSeek-R1 output format.

    Returns formatted HuggingFace dataset.
    """
    from datasets import load_dataset

    log(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, cache_dir=HF_CACHE)
    log(f"  Raw dataset size: {len(ds)}")

    # Filter: only examples with non-empty thinking traces
    ds = ds.filter(
        lambda x: (x.get("message_thinking") or "").strip() != "",
        desc="Filtering empty thinking traces"
    )
    log(f"  After filtering empty traces: {len(ds)}")

    # Filter: exclude very long traces (>10K chars) to avoid OOM
    ds = ds.filter(
        lambda x: len(x.get("message_thinking", "")) < 10000,
        desc="Filtering very long traces"
    )
    log(f"  After length filter: {len(ds)}")

    # Shuffle and cap
    ds = ds.shuffle(seed=SEED)
    if len(ds) > max_examples:
        ds = ds.select(range(max_examples))
    log(f"  Training examples: {len(ds)}")

    # Format as chat messages with <think> tags
    def format_reasoning(example):
        """Format a single example as a chat message with reasoning trace.

        Output format:
            User: [math problem]
            Assistant: <think>[reasoning trace]</think>

            [final answer with \\boxed{...}]
        """
        problem = example["problem"]
        thinking = example.get("message_thinking", "")
        content = example.get("message_content", "")

        # Construct the full assistant response with thinking tags
        if thinking:
            assistant_response = f"<think>\n{thinking}\n</think>\n\n{content}"
        else:
            assistant_response = content

        # Format as chat messages
        messages = [
            {"role": "system", "content": (
                "You are a helpful math assistant that shows your reasoning "
                "step by step inside <think>...</think> tags before giving "
                "your final answer."
            )},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_response},
        ]

        # Apply tokenizer chat template
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    ds = ds.map(format_reasoning, remove_columns=ds.column_names,
                desc="Formatting reasoning traces")

    # Log token length statistics
    sample_lengths = []
    for i in range(min(100, len(ds))):
        tokens = tokenizer(ds[i]["text"], truncation=False)["input_ids"]
        sample_lengths.append(len(tokens))

    import numpy as np
    log(f"  Token length stats (sample of {len(sample_lengths)}):")
    log(f"    Mean: {np.mean(sample_lengths):.0f}")
    log(f"    Median: {np.median(sample_lengths):.0f}")
    log(f"    Max: {np.max(sample_lengths)}")
    log(f"    Min: {np.min(sample_lengths)}")
    log(f"    >2048: {sum(1 for l in sample_lengths if l > 2048)}")

    return ds


def train_reasoning_adapter(args):
    """Train the reasoning QLoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    base_model = args.base_model
    adapter_out = Path(args.output_dir)
    steps = args.steps
    lr = args.lr

    # Check if already trained
    if (adapter_out / "adapter_config.json").exists() and not args.force:
        log(f"Adapter already exists at {adapter_out}, use --force to retrain")
        return

    adapter_out.mkdir(parents=True, exist_ok=True)

    # Nanochat pattern: disable GC during training to avoid ~500ms/step overhead
    # from Python cycle detection on long-lived PyTorch tensors
    gc.disable()
    gc.collect()

    log("=" * 72)
    log("REASONING EXPERT DISTILLATION")
    log(f"  Base model: {base_model}")
    log(f"  Output: {adapter_out}")
    log(f"  Rank: {LORA_RANK}, Steps: {steps}, LR: {lr}")
    log(f"  Max seq length: {MAX_SEQ_LENGTH}")
    log("=" * 72)

    # Load tokenizer first (needed for dataset formatting)
    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    dataset = prepare_reasoning_dataset(tokenizer, max_examples=args.max_examples)

    # Load base model with 4-bit quantization
    log("Loading base model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters (ALL MODULES -- matches pilot50 for composition)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_MODULES,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Train
    t0 = time.time()
    ckpt_dir = adapter_out / "checkpoints"
    # Nanochat pattern: explicit precision detection (no autocast)
    # SM 80+ (A100/H100/A5000/4090) → bf16, older → fp16
    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(ckpt_dir),
            max_steps=steps,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=lr,
            warmup_steps=WARMUP_STEPS,
            lr_scheduler_type="cosine",
            logging_steps=25,
            save_steps=100,  # Intermediate checkpoints every 100 steps
            save_total_limit=2,  # Keep only last 2 checkpoints to save disk
            bf16=use_bf16,
            fp16=not use_bf16 if torch.cuda.is_available() else False,
            optim="adamw_8bit",
            seed=SEED,
            dataset_text_field="text",
            max_length=MAX_SEQ_LENGTH,
            packing=True,
            report_to="none",
            # Nanochat pattern: warmdown ratio for cosine schedule (decay to near-zero)
            warmup_ratio=0.0,  # Use warmup_steps instead
            # Nanochat: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set in gpu_queue worker)
            dataloader_pin_memory=True,  # Faster H-to-D transfers
            dataloader_num_workers=2,  # Async data loading
        ),
    )

    log("Starting training...")
    train_result = trainer.train()
    train_loss = train_result.training_loss
    elapsed = time.time() - t0

    log(f"Training complete: loss={train_loss:.4f}, time={elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save adapter
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    # Cleanup checkpoints
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Save training metadata
    meta = {
        "experiment": "reasoning_expert_distillation",
        "type": "reasoning_capability_adapter",
        "base_model": base_model,
        "dataset": DATASET_NAME,
        "n_train_examples": len(dataset),
        "max_seq_length": MAX_SEQ_LENGTH,
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "target_modules": LORA_MODULES,
        "steps": steps,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "lr": lr,
        "warmup_steps": WARMUP_STEPS,
        "train_loss": float(train_loss),
        "train_time_s": float(elapsed),
        "trainable_params": trainable,
        "total_params": total,
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(adapter_out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log(f"Adapter saved to {adapter_out}")
    log(f"Training metadata saved to {adapter_out / 'train_meta.json'}")

    # Re-enable GC and cleanup
    gc.enable()
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Train reasoning LoRA expert via distillation"
    )
    parser.add_argument("--base-model", default=BASE_MODEL,
                        help="Path to base model")
    parser.add_argument("--output-dir", default=str(ADAPTER_DIR),
                        help="Output directory for adapter")
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS,
                        help="Training steps")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Learning rate")
    parser.add_argument("--max-examples", type=int, default=MAX_TRAIN_EXAMPLES,
                        help="Max training examples")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if adapter exists")
    args = parser.parse_args()

    meta = train_reasoning_adapter(args)
    if meta:
        log(f"\nDone. Train loss: {meta['train_loss']:.4f}")
        est_cost = meta["train_time_s"] / 3600 * 0.34
        log(f"Estimated cost: ${est_cost:.2f}")


if __name__ == "__main__":
    main()
