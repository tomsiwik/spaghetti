#!/usr/bin/env python3
"""Train 450 new QLoRA adapters for scale-500 experiment.

Trains one adapter at a time via subprocess isolation (same pattern as
pilot50_train.py). Uses data prepared by prepare_scale_data.py.

Resumes from existing adapters (idempotent).

Supports SMOKE_TEST=1 for quick validation (trains 2 adapters, 10 steps each).

Usage (on RunPod via gpu_queue):
    uv run python3 tools/gpu_queue.py submit macro/scale_500_experts/train_scale_adapters.py
"""

import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

REPO_ROOT = Path("/workspace/llm")
DATA_DIR = REPO_ROOT / "data" / "scale_500"
ADAPTER_DIR = REPO_ROOT / "data" / "adapters"  # Same dir as pilot50 adapters
HF_CACHE = "/workspace/hf_cache"

BASE_MODEL = "Qwen/Qwen2.5-7B"
RANK = 16
STEPS = 10 if IS_SMOKE else 100  # Fewer steps than pilot50 (100 vs 300) for cost
LR = 2e-4
N_NEW_DOMAINS = 2 if IS_SMOKE else 450


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def adapter_is_complete(domain: str) -> bool:
    d = ADAPTER_DIR / domain
    return (d / "adapter_config.json").exists() and (d / "adapter_model.safetensors").exists()


def train_one_adapter(domain: str, data_path: Path):
    """Train a single QLoRA adapter. Returns elapsed time."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    adapter_out = ADAPTER_DIR / domain
    adapter_out.mkdir(parents=True, exist_ok=True)

    if adapter_is_complete(domain):
        log(f"  {domain}: already trained, skipping")
        return 0.0

    log(f"  Training {domain} from {data_path}")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, cache_dir=HF_CACHE, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=RANK,
        lora_alpha=RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

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
            max_steps=STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=LR,
            warmup_steps=min(10, STEPS // 10),
            logging_steps=max(1, STEPS // 5),
            save_steps=STEPS,
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
    log(f"  {domain}: done in {elapsed:.0f}s, loss={train_loss:.4f}")

    meta = {
        "domain": domain,
        "base_model": BASE_MODEL,
        "rank": RANK,
        "steps": STEPS,
        "lr": LR,
        "train_loss": train_loss,
        "train_time_s": elapsed,
        "scale_experiment": True,
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
    t0 = time.time()
    log("=" * 60)
    log("Scale 500 Adapter Training")
    log(f"  New domains: {N_NEW_DOMAINS}")
    log(f"  Steps: {STEPS}, Rank: {RANK}, LR: {LR}")
    log(f"  Smoke test: {IS_SMOKE}")
    log("=" * 60)

    # Find domains that need training
    needs_training = []
    already_done = []
    for i in range(N_NEW_DOMAINS):
        domain = f"domain_{i + 50:03d}"
        data_path = DATA_DIR / domain / "train.jsonl"
        if not data_path.exists():
            log(f"SKIP {domain}: no training data at {data_path}")
            continue
        if adapter_is_complete(domain):
            already_done.append(domain)
        else:
            needs_training.append((domain, data_path))

    log(f"\n  Already trained: {len(already_done)}")
    log(f"  Need training: {len(needs_training)}")
    est_min = len(needs_training) * 5  # ~5 min each at 100 steps
    est_cost = est_min / 60 * 0.34
    log(f"  Estimated time: {est_min} min ({est_min / 60:.1f} hr)")
    log(f"  Estimated cost: ${est_cost:.2f}")

    if not needs_training:
        log("All adapters trained. Nothing to do.")
        return

    trained = 0
    failed = []
    total_train_time = 0.0

    for i, (domain, data_path) in enumerate(needs_training):
        log(f"\n[{i + 1}/{len(needs_training)}] {domain}")
        try:
            elapsed = train_one_adapter(domain, data_path)
            total_train_time += elapsed
            if adapter_is_complete(domain):
                trained += 1
            else:
                failed.append((domain, "adapter not complete after training"))
        except Exception as e:
            log(f"  FAILED {domain}: {e}")
            failed.append((domain, str(e)))
            # Try to recover GPU memory
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # Summary
    wall_time = time.time() - t0
    log(f"\n{'=' * 60}")
    log(f"TRAINING COMPLETE")
    log(f"  Trained: {trained}/{len(needs_training)}")
    log(f"  Failed: {len(failed)}")
    log(f"  Train time: {total_train_time:.0f}s ({total_train_time / 60:.1f} min)")
    log(f"  Wall time: {wall_time:.0f}s ({wall_time / 60:.1f} min)")
    log(f"  Cost: ${wall_time / 3600 * 0.34:.2f}")

    # Count total adapters (pilot50 + scale)
    total_adapters = sum(1 for d in ADAPTER_DIR.iterdir()
                        if d.is_dir() and (d / "adapter_config.json").exists())
    log(f"\n  Total adapters available: {total_adapters}")

    if failed:
        log(f"\nFailed domains:")
        for domain, err in failed[:20]:
            log(f"  {domain}: {err}")
        sys.exit(1)

    # Save results
    results = {
        "trained": trained,
        "failed": len(failed),
        "total_train_time_s": total_train_time,
        "wall_time_s": wall_time,
        "cost_usd": wall_time / 3600 * 0.34,
        "total_adapters": total_adapters,
        "steps": STEPS,
        "rank": RANK,
    }
    results_path = REPO_ROOT / "results" / "scale_500_training.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
