#!/usr/bin/env python3
"""Train QLoRA experts from 8B teacher data on RunPod.

Same hyperparameters as pilot50_train.py but uses data from data/distillation_8b/.
Saves adapters to adapters_8b/ to compare with existing 70B adapters in adapters/.

Usage (on RunPod, submitted via gpu_queue.py):
    python experiments/models/distillation_quality_vs_teacher/train_8b_adapters.py
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "distillation_8b"
ADAPTER_DIR = REPO_ROOT / "adapters_8b"
HF_CACHE = "/workspace/hf_cache"

SELECTED_DOMAINS = [
    "python", "sql", "bash", "physics", "accounting",
    "ethics", "creative-fiction", "causal-reasoning", "legal", "game-theory",
]


def train_one_expert(base_model, data_path, output_dir, rank=16, steps=300, lr=2e-4):
    """Train a single QLoRA expert. Same config as pilot50_train.py."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    domain = data_path.parent.name
    adapter_out = output_dir / domain
    adapter_out.mkdir(parents=True, exist_ok=True)

    if (adapter_out / "adapter_config.json").exists():
        print(f"  {domain}: adapter already exists, skipping")
        return 0.0

    print(f"\n{'='*60}")
    print(f"  Training 8B-teacher expert: {domain}")
    print(f"    Data: {data_path}")
    print(f"    Output: {adapter_out}")
    print(f"    Rank: {rank}, Steps: {steps}, LR: {lr}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    def format_messages(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_messages)

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
            save_steps=steps,
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

    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))

    ckpt_dir = adapter_out / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    elapsed = time.time() - start
    print(f"  {domain}: done in {elapsed:.0f}s, loss={train_loss:.4f}")

    meta = {
        "domain": domain,
        "teacher_model": "llama-3.1-8b-instant",
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

    del trainer, model, tokenizer, dataset
    gc.collect()
    import torch as _torch
    _torch.cuda.empty_cache()
    _torch.cuda.reset_peak_memory_stats()

    return elapsed


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--domains", nargs="*", help="Specific domains")
    args = parser.parse_args()

    domains = args.domains or SELECTED_DOMAINS
    print(f"Teacher Size Experiment: Train 8B-teacher Adapters")
    print(f"  Base: {args.base}")
    print(f"  Rank: {args.rank}, Steps: {args.steps}")
    print(f"  Domains: {domains}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Output dir: {ADAPTER_DIR}")
    print()

    total_time = 0.0
    trained = 0
    failed = []

    for domain in domains:
        data_path = DATA_DIR / domain / "train.jsonl"
        if not data_path.exists():
            print(f"  {domain}: NO DATA at {data_path}, skipping")
            failed.append((domain, "no data"))
            continue

        with open(data_path) as f:
            n_examples = sum(1 for _ in f)
        print(f"  {domain}: {n_examples} training examples")

        try:
            elapsed = train_one_expert(
                base_model=args.base,
                data_path=data_path,
                output_dir=ADAPTER_DIR,
                rank=args.rank,
                steps=args.steps,
                lr=args.lr,
            )
            total_time += elapsed
            if elapsed > 0:
                trained += 1
        except Exception as e:
            print(f"  FAILED: {domain}: {e}")
            failed.append((domain, str(e)))

    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"  Trained: {trained}")
    print(f"  Failed: {len(failed)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    if trained > 0:
        print(f"  Avg time/expert: {total_time/trained:.0f}s")
    actual_cost = total_time / 3600 * 0.16  # A5000 rate
    print(f"  Estimated cost: ${actual_cost:.2f}")

    if failed:
        print(f"\nFailed:")
        for domain, err in failed:
            print(f"  {domain}: {err}")

    # Verify
    print(f"\nAdapter verification:")
    ok = 0
    for domain in domains:
        adapter_dir = ADAPTER_DIR / domain
        has_config = (adapter_dir / "adapter_config.json").exists()
        has_weights = (adapter_dir / "adapter_model.safetensors").exists()
        status = "OK" if (has_config and has_weights) else "MISSING"
        if has_config and has_weights:
            ok += 1
        print(f"  {domain}: {status}")
    print(f"\n{ok}/{len(domains)} adapters ready")


if __name__ == "__main__":
    main()
