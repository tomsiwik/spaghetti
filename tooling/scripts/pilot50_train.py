#!/usr/bin/env python3
"""Train QLoRA experts for all 50 domains on RunPod.

Designed to run ON the RunPod instance. Trains one expert at a time,
resuming from any existing adapters (idempotent).

Usage (on RunPod):
    cd /workspace/llm
    python scripts/pilot50_train.py [--rank 16] [--steps 300]
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "distillation"
ADAPTER_DIR = REPO_ROOT / "adapters"
HF_CACHE = "/workspace/hf_cache"


def train_one_expert(base_model: str, data_path: Path, output_dir: Path,
                     rank: int = 16, steps: int = 300, lr: float = 2e-4):
    """Train a single QLoRA expert. Returns training time in seconds."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    domain = data_path.parent.name
    adapter_out = output_dir / domain
    adapter_out.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    if (adapter_out / "adapter_config.json").exists():
        print(f"  {domain}: adapter already exists, skipping")
        return 0.0

    print(f"\n{'='*60}")
    print(f"  Training expert: {domain}")
    print(f"    Data: {data_path}")
    print(f"    Output: {adapter_out}")
    print(f"    Rank: {rank}, Steps: {steps}, LR: {lr}")
    start = time.time()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 4-bit quantization
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

    # Add LoRA adapters — ALL MODULES (q/k/v/o/gate/up/down)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # Print trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")

    # Apply chat template
    def format_messages(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_messages)

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(adapter_out / "checkpoints"),
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=lr,
            warmup_steps=min(10, steps // 10),
            logging_steps=50,
            save_steps=steps,  # save only at end
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=1024,
            packing=True,
            report_to="none",
        ),
    )

    train_result = trainer.train()
    train_loss = train_result.training_loss

    # Save adapter
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    # Cleanup checkpoints to save space
    ckpt_dir = adapter_out / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    elapsed = time.time() - start
    print(f"  {domain}: done in {elapsed:.0f}s, loss={train_loss:.4f}")

    # Save training metadata
    meta = {
        "domain": domain,
        "base_model": base_model,
        "rank": rank,
        "steps": steps,
        "lr": lr,
        "train_loss": train_loss,
        "train_time_s": elapsed,
        "trainable_params": trainable,
        "total_params": total,
    }
    with open(adapter_out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Free GPU memory
    del trainer, model, tokenizer, dataset
    gc.collect()
    import torch as _torch
    _torch.cuda.empty_cache()
    _torch.cuda.reset_peak_memory_stats()

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Train 50 QLoRA experts on RunPod")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--domains", nargs="*", help="Specific domains to train")
    parser.add_argument("--single", action="store_true", help="Train in-process (used by subprocess spawner)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Find all domain training files
    train_files = sorted(DATA_DIR.glob("*/train.jsonl"))
    if args.domains:
        train_files = [f for f in train_files if f.parent.name in args.domains]

    if not train_files:
        print(f"No train.jsonl files found in {DATA_DIR}/*/")
        sys.exit(1)

    # Check what needs training
    needs_training = []
    already_done = []
    for tf in train_files:
        domain = tf.parent.name
        adapter_dir = ADAPTER_DIR / domain
        if (adapter_dir / "adapter_config.json").exists():
            already_done.append(domain)
        else:
            needs_training.append((domain, tf))

    print(f"50-Domain QLoRA Training")
    print(f"  Base: {args.base}")
    print(f"  Rank: {args.rank}, Steps: {args.steps}")
    print(f"  Total data dirs: {len(train_files)}")
    print(f"  Already trained: {len(already_done)}")
    print(f"  Need training: {len(needs_training)}")
    est_time = len(needs_training) * 15  # ~15 min each
    est_cost = len(needs_training) * 15 / 60 * 0.34  # $0.34/hr for 4090
    print(f"  Estimated time: {est_time} min ({est_time/60:.1f} hr)")
    print(f"  Estimated cost: ${est_cost:.2f}")
    print()

    if args.dry_run:
        print("Would train:")
        for domain, tf in needs_training:
            with open(tf) as f:
                n = sum(1 for _ in f)
            print(f"  {domain}: {n} examples")
        return

    # Single-domain in-process mode (called by subprocess spawner)
    if args.single:
        for domain, tf in needs_training:
            train_one_expert(
                base_model=args.base,
                data_path=tf,
                output_dir=ADAPTER_DIR,
                rank=args.rank,
                steps=args.steps,
                lr=args.lr,
            )
        return

    total_time = 0.0
    trained = 0
    failed = []

    for i, (domain, tf) in enumerate(needs_training):
        print(f"\n[{i+1}/{len(needs_training)}] Training {domain}...", flush=True)
        # Spawn as subprocess to fully release GPU memory between experts
        result = subprocess.run(
            [sys.executable, __file__,
             "--base", args.base,
             "--rank", str(args.rank),
             "--steps", str(args.steps),
             "--lr", str(args.lr),
             "--single",
             "--domains", domain],
            cwd=str(REPO_ROOT),
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        if result.returncode == 0:
            # Read elapsed time from train_meta.json
            meta_file = ADAPTER_DIR / domain / "train_meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text())
                elapsed = meta.get("train_time_s", 0)
                total_time += elapsed
                trained += 1
                print(f"  {domain}: done in {elapsed:.0f}s", flush=True)
            else:
                trained += 1
        else:
            print(f"  FAILED: {domain}: subprocess exited {result.returncode}", flush=True)
            failed.append((domain, f"exit code {result.returncode}"))

    # Final summary
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Trained: {trained}/{len(needs_training)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Avg time/expert: {total_time/max(trained,1):.0f}s")
    actual_cost = total_time / 3600 * 0.34
    print(f"  Actual cost: ${actual_cost:.2f}")

    if failed:
        print(f"\nFailed domains:")
        for domain, err in failed:
            print(f"  {domain}: {err}")

    # Verify all adapters
    print(f"\nAdapter verification:")
    all_ok = 0
    for tf in sorted(DATA_DIR.glob("*/train.jsonl")):
        domain = tf.parent.name
        adapter_dir = ADAPTER_DIR / domain
        has_config = (adapter_dir / "adapter_config.json").exists()
        has_weights = (adapter_dir / "adapter_model.safetensors").exists()
        status = "OK" if (has_config and has_weights) else "MISSING"
        if has_config and has_weights:
            all_ok += 1
        print(f"  {domain}: {status}")
    print(f"\n{all_ok}/{len(list(DATA_DIR.glob('*/train.jsonl')))} adapters ready")


if __name__ == "__main__":
    main()
