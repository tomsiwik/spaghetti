#!/usr/bin/env python3
"""Train 5 FFN-only rank-16 adapters on Qwen2.5-7B.

This script is designed to run on RunPod with an A5000 GPU.
It trains FFN-only adapters (gate_proj, up_proj, down_proj) on the same
distillation data used for the existing all-modules adapters, with identical
hyperparameters except for target_modules.

Usage (on RunPod):
    python3 experiments/models/ffn_only_matched_rank/train_ffn_only.py \
        --base /workspace/models/Qwen2.5-7B \
        --data data/distillation/ \
        --output adapters_ffn_only/ \
        --eval-data data/distillation/

The script also trains all-modules adapters for a fair comparison (same seed,
same training run) if --also-train-all is passed.
"""

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path


DOMAINS = ["bash", "math", "medical", "python", "sql"]

FFN_MODULES = ["gate_proj", "up_proj", "down_proj"]
ALL_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]


def train_single_adapter(
    base_model: str,
    train_path: Path,
    eval_path: Path,
    output_dir: Path,
    target_modules: list[str],
    rank: int = 16,
    steps: int = 300,
    lr: float = 2e-4,
    seed: int = 42,
) -> dict:
    """Train a single QLoRA adapter and evaluate PPL. Returns metrics dict."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    domain = train_path.parent.name
    target_label = "ffn" if "q_proj" not in target_modules else "all"

    # Check if already trained
    results_file = output_dir / "metrics.json"
    if results_file.exists():
        print(f"  {domain} ({target_label}): already trained, loading cached metrics")
        return json.loads(results_file.read_text())

    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"\n  Training: {domain} ({target_label})")
    print(f"    Target modules: {target_modules}")
    print(f"    Rank: {rank}, Steps: {steps}, LR: {lr}, Seed: {seed}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
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
    )
    model = prepare_model_for_kbit_training(model)

    # Add LoRA
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
    print(f"    Trainable: {trainable:,} / {total:,} ({trainable/total*100:.4f}%)")

    # Load and format dataset
    dataset = load_dataset("json", data_files=str(train_path), split="train")

    def format_messages(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_messages)

    # Train
    from transformers import TrainerCallback

    class _LossCallback(TrainerCallback):
        def __init__(self):
            self.loss_curve = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                self.loss_curve.append({
                    "step": state.global_step,
                    "loss": round(logs["loss"], 6),
                })

    loss_cb = _LossCallback()

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        callbacks=[loss_cb],
        args=SFTConfig(
            output_dir=str(output_dir / "checkpoints"),
            max_steps=steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=lr,
            warmup_steps=min(10, steps // 10),
            logging_steps=10,
            save_strategy="no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=seed,
            dataset_text_field="text",
            max_length=1024,
            packing=True,
        ),
    )

    train_result = trainer.train()
    train_loss = train_result.training_loss
    t_train = time.time() - t_start

    # Save loss curve
    loss_curve_path = output_dir / "loss_curve.json"
    loss_curve_path.write_text(json.dumps(loss_cb.loss_curve, indent=2))

    # Evaluate PPL
    eval_dataset = load_dataset("json", data_files=str(eval_path), split="train")
    eval_dataset = eval_dataset.map(format_messages)

    eval_trainer = SFTTrainer(
        model=model,
        train_dataset=eval_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            output_dir=str(output_dir / "eval_tmp"),
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
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Adapter size
    adapter_file = output_dir / "adapter_model.safetensors"
    adapter_size_mb = adapter_file.stat().st_size / 1024 / 1024 if adapter_file.exists() else 0

    # Check convergence: last 50 steps should show <5% improvement
    converged = True
    if len(loss_cb.loss_curve) >= 4:
        recent = [p["loss"] for p in loss_cb.loss_curve[-3:]]
        earlier = [p["loss"] for p in loss_cb.loss_curve[-4:-1]]
        if statistics.mean(earlier) > 0:
            improvement = (statistics.mean(earlier) - statistics.mean(recent)) / statistics.mean(earlier) * 100
            converged = improvement < 5.0
            print(f"    Convergence: last-3 improvement={improvement:.2f}% {'(converged)' if converged else '(NOT converged — may need more steps)'}")

    metrics = {
        "domain": domain,
        "target": target_label,
        "target_modules": target_modules,
        "rank": rank,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "trainable_params": trainable,
        "total_params": total,
        "train_loss": round(train_loss, 4),
        "eval_ppl": round(ppl, 4),
        "adapter_size_mb": round(adapter_size_mb, 2),
        "train_time_s": round(t_train, 1),
        "converged": converged,
    }

    results_file.write_text(json.dumps(metrics, indent=2))
    print(f"    PPL: {ppl:.4f}, Loss: {train_loss:.4f}, "
          f"Size: {adapter_size_mb:.1f}MB, Time: {t_train:.0f}s")

    # Cleanup
    import shutil
    for d in ["checkpoints", "eval_tmp"]:
        p = output_dir / d
        if p.exists():
            shutil.rmtree(p)

    del trainer, model, tokenizer, dataset, eval_dataset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train FFN-only rank-16 adapters for matched-rank comparison")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B",
                        help="Base model (default: Qwen/Qwen2.5-7B)")
    parser.add_argument("--data", default="data/distillation/",
                        help="Training data directory")
    parser.add_argument("--eval-data", default="data/distillation/",
                        help="Eval data directory (uses eval.jsonl from each domain)")
    parser.add_argument("--output", default="adapters_ffn_only/",
                        help="Output directory for FFN-only adapters")
    parser.add_argument("--output-all", default="adapters_all_retrained/",
                        help="Output directory for retrained all-modules adapters")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--domains", nargs="+", default=DOMAINS)
    parser.add_argument("--also-train-all", action="store_true",
                        help="Also train all-modules adapters for a fair comparison")

    args = parser.parse_args()
    data_dir = Path(args.data)
    eval_dir = Path(args.eval_data)

    all_results = []
    t_total = time.time()

    # ---- Train FFN-only adapters ----
    print("=" * 70)
    print("  PHASE 1: Training FFN-only adapters")
    print("=" * 70)

    for domain in args.domains:
        train_file = data_dir / domain / "train.jsonl"
        eval_file = eval_dir / domain / "eval.jsonl"
        if not train_file.exists():
            print(f"  {domain}: no training data at {train_file}")
            continue
        if not eval_file.exists():
            print(f"  {domain}: no eval data at {eval_file}")
            continue

        output_dir = Path(args.output) / domain
        metrics = train_single_adapter(
            base_model=args.base,
            train_path=train_file,
            eval_path=eval_file,
            output_dir=output_dir,
            target_modules=FFN_MODULES,
            rank=args.rank,
            steps=args.steps,
            lr=args.lr,
            seed=args.seed,
        )
        all_results.append(metrics)

    # ---- Optionally train all-modules adapters ----
    if args.also_train_all:
        print("\n" + "=" * 70)
        print("  PHASE 2: Training all-modules adapters (fair comparison)")
        print("=" * 70)

        for domain in args.domains:
            train_file = data_dir / domain / "train.jsonl"
            eval_file = eval_dir / domain / "eval.jsonl"
            if not train_file.exists() or not eval_file.exists():
                continue

            output_dir = Path(args.output_all) / domain
            metrics = train_single_adapter(
                base_model=args.base,
                train_path=train_file,
                eval_path=eval_file,
                output_dir=output_dir,
                target_modules=ALL_MODULES,
                rank=args.rank,
                steps=args.steps,
                lr=args.lr,
                seed=args.seed,
            )
            all_results.append(metrics)

    # ---- Summary ----
    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"\n  {'Domain':<12} {'Target':<6} {'PPL':<10} {'Loss':<10} {'Size(MB)':<10} {'Time(s)':<10}")
    print("  " + "-" * 60)
    for r in sorted(all_results, key=lambda x: (x["domain"], x["target"])):
        print(f"  {r['domain']:<12} {r['target']:<6} {r['eval_ppl']:<10.4f} "
              f"{r['train_loss']:<10.4f} {r['adapter_size_mb']:<10.1f} "
              f"{r['train_time_s']:<10.0f}")

    # Save summary
    summary_path = Path(args.output) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
