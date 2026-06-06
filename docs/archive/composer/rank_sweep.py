#!/usr/bin/env python3
"""Rank sweep experiment: find optimal rank per domain for FFN-only LoRA.

Tests the hypothesis that:
1. FFN-only adapters (frozen attention) compose better
2. Simpler domains saturate at lower ranks
3. There's a sweet spot between rank, quality, and training cost

Usage:
    python -m composer.rank_sweep \
        --base /workspace/models/Qwen2.5-7B \
        --data data/distillation/ \
        --output results/rank_sweep.json \
        --ranks 8 16 32 64 128 256 \
        --domains bash python math \
        --steps 100
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path


def train_and_eval(base_model: str, data_path: Path, output_dir: Path,
                   rank: int, steps: int, target: str, eval_path: Path):
    """Train a single adapter and evaluate it. Returns metrics dict."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    domain = data_path.parent.name
    adapter_out = output_dir / f"{domain}_r{rank}_{target}"
    adapter_out.mkdir(parents=True, exist_ok=True)

    # Skip if already done
    results_file = adapter_out / "metrics.json"
    if results_file.exists():
        print(f"    {domain} r={rank} {target}: cached")
        return json.loads(results_file.read_text())

    t_start = time.time()

    # Target modules
    if target == "ffn":
        target_modules = ["gate_proj", "up_proj", "down_proj"]
    elif target == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    else:  # "all"
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
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
        dtype=torch.bfloat16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=target_modules,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Load dataset
    dataset = load_dataset("json", data_files=str(data_path), split="train")

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
            learning_rate=2e-4,
            warmup_steps=min(10, steps // 10),
            logging_steps=steps,  # only log at end
            save_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=1024,
            packing=True,
        ),
    )

    train_result = trainer.train()
    train_loss = train_result.training_loss
    t_train = time.time() - t_start

    # Eval perplexity using trainer (handles device placement correctly)
    eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")
    eval_dataset = eval_dataset.map(format_messages)

    eval_trainer = SFTTrainer(
        model=model,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(adapter_out / "eval_tmp"),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            dataset_text_field="text",
            max_length=1024,
            packing=True,
            report_to="none",
        ),
    )
    eval_metrics = eval_trainer.evaluate()
    ppl = math.exp(eval_metrics["eval_loss"])
    del eval_trainer

    # Save adapter
    model.save_pretrained(str(adapter_out))

    # Adapter size on disk
    adapter_file = adapter_out / "adapter_model.safetensors"
    adapter_size_mb = adapter_file.stat().st_size / 1024 / 1024 if adapter_file.exists() else 0

    metrics = {
        "domain": domain,
        "rank": rank,
        "target": target,
        "target_modules": target_modules,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(trainable / total * 100, 4),
        "train_loss": round(train_loss, 4),
        "eval_ppl": round(ppl, 4),
        "adapter_size_mb": round(adapter_size_mb, 2),
        "train_time_s": round(t_train, 1),
        "steps": steps,
    }

    results_file.write_text(json.dumps(metrics, indent=2))

    # Cleanup
    import shutil
    for d in ["checkpoints", "eval_tmp"]:
        p = adapter_out / d
        if p.exists():
            shutil.rmtree(p)

    del trainer, model, tokenizer, dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Rank sweep for FFN-only LoRA")
    parser.add_argument("--base", required=True, help="Base model path")
    parser.add_argument("--data", required=True, help="Data directory")
    parser.add_argument("--output", default="results/rank_sweep.json")
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--domains", nargs="+", default=["bash", "python", "math"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--targets", nargs="+", default=["ffn", "all"],
                        help="Module targets to compare: ffn, attn, all")

    args = parser.parse_args()
    import torch
    data_dir = Path(args.data)
    output_dir = Path("adapters/sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    total_runs = len(args.domains) * len(args.ranks) * len(args.targets)
    run = 0

    for domain in args.domains:
        train_file = data_dir / domain / "train.jsonl"
        eval_file = data_dir / domain / "eval.jsonl"
        if not train_file.exists():
            print(f"  {domain}: no training data, skipping")
            continue
        if not eval_file.exists():
            print(f"  {domain}: no eval data, skipping")
            continue

        for target in args.targets:
            for rank in args.ranks:
                run += 1
                print(f"\n[{run}/{total_runs}] {domain} rank={rank} target={target}")
                try:
                    metrics = train_and_eval(
                        base_model=args.base,
                        data_path=train_file,
                        output_dir=output_dir,
                        rank=rank,
                        steps=args.steps,
                        target=target,
                        eval_path=eval_file,
                    )
                    all_results.append(metrics)
                    print(f"    PPL={metrics['eval_ppl']:.2f}  "
                          f"size={metrics['adapter_size_mb']:.1f}MB  "
                          f"time={metrics['train_time_s']:.0f}s  "
                          f"params={metrics['trainable_pct']:.2f}%")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"    OOM — rank {rank} too large for GPU, skipping")
                    gc.collect()
                    torch.cuda.empty_cache()
                    all_results.append({
                        "domain": domain, "rank": rank, "target": target,
                        "eval_ppl": float("inf"), "adapter_size_mb": 0,
                        "train_time_s": 0, "trainable_pct": 0, "error": "OOM",
                    })

    # Save all results
    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"  RANK SWEEP RESULTS")
    print(f"{'='*80}")
    print(f"{'Domain':<10} {'Target':<6} {'Rank':<6} {'PPL':<10} {'Size(MB)':<10} {'Time(s)':<10} {'Params%':<10}")
    print("-" * 72)
    for r in sorted(all_results, key=lambda x: (x["domain"], x["target"], x["rank"])):
        print(f"{r['domain']:<10} {r['target']:<6} {r['rank']:<6} "
              f"{r['eval_ppl']:<10.2f} {r['adapter_size_mb']:<10.1f} "
              f"{r['train_time_s']:<10.0f} {r['trainable_pct']:<10.4f}")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
