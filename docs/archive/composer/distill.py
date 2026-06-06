#!/usr/bin/env python3
"""Distillation pipeline — generate training data and train LoRA experts.

Two subcommands:
  generate  — Call an OpenAI-compatible API (Groq, Cerebras, local) to produce
               domain-specific instruction-response training data.
  train     — Fine-tune QLoRA rank-16 adapters on Qwen2.5-7B using Unsloth + TRL.

Usage:
    python -m composer.distill generate --domains domains.yml --output data/distillation/
    python -m composer.distill train --data data/distillation/ --base Qwen/Qwen2.5-7B --output adapters/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ── Data Generation ────────────────────────────────────────────────────

DEFAULT_DOMAINS = {
    "python": "Python programming: write functions, classes, algorithms, data structures, and debugging.",
    "sql": "SQL database queries: SELECT, JOIN, GROUP BY, subqueries, window functions, optimization.",
    "bash": "Bash shell scripting: file manipulation, text processing, pipelines, system administration.",
    "medical": "Clinical medicine: symptoms, diagnoses, treatment plans, drug interactions, patient management.",
    "math": "Mathematical reasoning: algebra, calculus, probability, proofs, word problems.",
}

SYSTEM_PROMPT = """You are an expert training data generator. Given a domain description,
generate a high-quality instruction-response pair for fine-tuning a language model.

The instruction should be a realistic user question or task in this domain.
The response should be thorough, correct, and demonstrate domain expertise.
Vary the difficulty: mix simple factual questions with complex multi-step problems.

Output ONLY valid JSON with exactly two fields:
{"instruction": "...", "response": "..."}"""


def generate_one(client, model: str, domain: str, description: str, temperature: float = 0.9) -> dict | None:
    """Generate a single instruction-response pair."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Domain: {domain}\nDescription: {description}\n\nGenerate one training example."},
            ],
            temperature=temperature,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content.strip()
        # Handle markdown code blocks
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        # LLMs often emit literal newlines/tabs inside JSON strings (code examples)
        text = text.replace("\t", "\\t")
        text = text.replace("\r", "\\r")
        # Replace unescaped newlines inside JSON string values
        import re
        text = re.sub(r'(?<!\\)\n', '\\\\n', text)
        pair = json.loads(text)
        if "instruction" in pair and "response" in pair:
            return pair
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"  Parse error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  API error: {e}", file=sys.stderr)
    return None


def generate_domain_data(client, model: str, domain: str, description: str,
                         n_examples: int, output_dir: Path):
    """Generate training data for one domain."""
    out_file = output_dir / domain / "train.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing file
    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)
        if existing >= n_examples:
            print(f"  {domain}: {existing}/{n_examples} already done, skipping")
            return existing

    print(f"  {domain}: generating {n_examples - existing} examples (have {existing})...")
    generated = 0
    retries = 0
    max_retries = n_examples * 2  # Allow 2x retries

    with open(out_file, "a") as f:
        while existing + generated < n_examples and retries < max_retries:
            pair = generate_one(client, model, domain, description)
            if pair:
                # Format as chat messages for TRL SFTTrainer
                record = {
                    "messages": [
                        {"role": "user", "content": pair["instruction"]},
                        {"role": "assistant", "content": pair["response"]},
                    ]
                }
                f.write(json.dumps(record) + "\n")
                generated += 1
                if generated % 50 == 0:
                    print(f"    {domain}: {existing + generated}/{n_examples}")
            else:
                retries += 1
                time.sleep(0.5)

    total = existing + generated
    print(f"  {domain}: {total}/{n_examples} complete")
    return total


HF_DATASETS = {
    "python": {
        "path": "codeparrot/codeparrot-clean",
        "split": "train",
        "mode": "completion",
        "text_field": "content",
    },
    "sql": {
        "path": "b-mc2/sql-create-context",
        "split": "train",
        "mode": "instruction",
        "instruction_field": "question",
        "response_field": "answer",
        "context_field": "context",  # table schema
    },
    "medical": {
        "path": "openlifescienceai/medmcqa",
        "split": "train",
        "mode": "mcqa",
        "question_field": "question",
        "options_fields": ["opa", "opb", "opc", "opd"],
        "answer_field": "cop",  # 0-3 index
        "explanation_field": "exp",
    },
    "math": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "mode": "instruction",
        "instruction_field": "question",
        "response_field": "answer",
    },
    # bash: no good public dataset — use synthetic generation via cmd_generate
}


def download_hf_domain(domain: str, output_dir: Path, n_examples: int) -> int:
    """Download training data from a HuggingFace dataset."""
    from datasets import load_dataset

    cfg = HF_DATASETS.get(domain)
    if not cfg:
        return 0

    out_file = output_dir / domain / "train.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Check existing
    existing = 0
    if out_file.exists():
        with open(out_file) as f:
            existing = sum(1 for _ in f)
        if existing >= n_examples:
            print(f"  {domain}: {existing}/{n_examples} already done, skipping")
            return existing

    print(f"  {domain}: downloading from {cfg['path']}...")
    load_kwargs = {"path": cfg["path"], "split": cfg["split"], "streaming": True}
    if "name" in cfg:
        load_kwargs["name"] = cfg["name"]
    ds = load_dataset(**load_kwargs)

    mode = cfg.get("mode", "instruction")
    written = 0
    with open(out_file, "a") as f:
        for example in ds:
            if existing + written >= n_examples:
                break

            record = None

            if mode == "completion":
                # Raw code — wrap as "complete this code" task
                text = example.get(cfg["text_field"], "")
                if not text or len(text) < 100:
                    continue
                # Split: first ~30% is prompt, rest is completion
                split_point = len(text) // 3
                prompt = text[:split_point].rstrip()
                completion = text[split_point:].strip()
                if not prompt or not completion:
                    continue
                record = {"messages": [
                    {"role": "user", "content": f"Complete the following code:\n\n{prompt}"},
                    {"role": "assistant", "content": completion},
                ]}

            elif mode == "mcqa":
                # Multiple-choice QA (medmcqa)
                question = example.get(cfg["question_field"], "")
                if not question:
                    continue
                options = [example.get(f, "") for f in cfg["options_fields"]]
                answer_idx = example.get(cfg["answer_field"])
                explanation = example.get(cfg.get("explanation_field", ""), "") or ""
                if answer_idx is None or not all(options):
                    continue
                labels = ["A", "B", "C", "D"]
                options_text = "\n".join(f"{labels[i]}. {opt}" for i, opt in enumerate(options))
                answer_text = f"{labels[answer_idx]}. {options[answer_idx]}"
                if explanation:
                    answer_text = f"{answer_text}\n\nExplanation: {explanation}"
                record = {"messages": [
                    {"role": "user", "content": f"{question}\n\n{options_text}"},
                    {"role": "assistant", "content": answer_text},
                ]}

            else:  # instruction mode
                instruction = example.get(cfg["instruction_field"], "")
                response = example.get(cfg["response_field"], "")
                if not instruction or not response:
                    continue
                context_field = cfg.get("context_field")
                if context_field:
                    context = example.get(context_field, "")
                    if context:
                        instruction = f"{instruction}\n\nContext:\n{context}"
                record = {"messages": [
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response},
                ]}

            if record:
                f.write(json.dumps(record) + "\n")
                written += 1

    total = existing + written
    print(f"  {domain}: {total}/{n_examples} from HuggingFace")
    return total


def cmd_download(args):
    """Download training data from HuggingFace datasets."""
    output_dir = Path(args.output)
    domains = args.domains_list or list(HF_DATASETS.keys())

    print(f"Downloading from HuggingFace for {len(domains)} domains")
    print(f"  Output: {output_dir}")

    for domain in domains:
        if domain not in HF_DATASETS:
            print(f"  {domain}: no HuggingFace dataset configured, skipping")
            continue
        download_hf_domain(domain, output_dir, args.n_examples)

    # Summary
    total = 0
    for domain in domains:
        fpath = output_dir / domain / "train.jsonl"
        if fpath.exists():
            with open(fpath) as fh:
                n = sum(1 for _ in fh)
            total += n
            print(f"  {domain}: {n} examples")
    print(f"\nTotal: {total} examples across {len(domains)} domains")


def cmd_generate(args):
    """Generate distillation training data via API."""
    from dotenv import load_dotenv
    load_dotenv()
    from openai import OpenAI

    # Load domains
    if args.domains:
        import yaml
        with open(args.domains) as f:
            domains = yaml.safe_load(f)
    else:
        domains = DEFAULT_DOMAINS

    # Init API client
    client = OpenAI(
        api_key=os.environ.get(args.api_key_env, os.environ.get("OPENAI_API_KEY", "")),
        base_url=args.base_url,
    )

    output_dir = Path(args.output)
    print(f"Generating {args.n_examples} examples each for {len(domains)} domains")
    print(f"  Model: {args.model}")
    print(f"  API: {args.base_url}")
    print(f"  Output: {output_dir}")

    for domain, description in domains.items():
        generate_domain_data(client, args.model, domain, description,
                             args.n_examples, output_dir)

    # Summary
    total = 0
    for domain in domains:
        f = output_dir / domain / "train.jsonl"
        if f.exists():
            with open(f) as fh:
                n = sum(1 for _ in fh)
            total += n
            print(f"  {domain}: {n} examples")
    print(f"\nTotal: {total} examples across {len(domains)} domains")


# ── Training ───────────────────────────────────────────────────────────

def train_one_expert(base_model: str, data_path: Path, output_dir: Path,
                     rank: int = 16, steps: int = 300, lr: float = 2e-4,
                     batch_size: int = 4):
    """Train a single QLoRA expert using BitsAndBytes + PEFT + TRL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    import torch

    domain = data_path.parent.name
    adapter_out = output_dir / domain
    adapter_out.mkdir(parents=True, exist_ok=True)

    # Check if already trained
    if (adapter_out / "adapter_config.json").exists():
        print(f"  {domain}: adapter already exists, skipping")
        return

    print(f"\n  Training expert: {domain}")
    print(f"    Data: {data_path}")
    print(f"    Output: {adapter_out}")
    print(f"    Rank: {rank}, Steps: {steps}, LR: {lr}")

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

    # Add LoRA adapters
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
            logging_steps=10,
            save_steps=steps,  # save only at end
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=1024,
            packing=True,
        ),
    )

    trainer.train()

    # Save adapter in PEFT format
    model.save_pretrained(str(adapter_out))
    tokenizer.save_pretrained(str(adapter_out))
    print(f"  {domain}: saved to {adapter_out}")

    # Cleanup checkpoints to save space
    import shutil
    ckpt_dir = adapter_out / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    # Free GPU memory before next expert
    import gc
    del trainer, model, tokenizer, dataset
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cmd_train(args):
    """Train QLoRA experts from distillation data."""
    data_dir = Path(args.data)
    output_dir = Path(args.output)

    # Find all domain training files
    train_files = sorted(data_dir.glob("*/train.jsonl"))
    if not train_files:
        print(f"No train.jsonl files found in {data_dir}/*/")
        sys.exit(1)

    print(f"Training {len(train_files)} experts")
    print(f"  Base: {args.base}")
    print(f"  Output: {output_dir}")
    print(f"  Rank: {args.rank}, Steps: {args.steps}, LR: {args.lr}")

    for train_file in train_files:
        train_one_expert(
            base_model=args.base,
            data_path=train_file,
            output_dir=output_dir,
            rank=args.rank,
            steps=args.steps,
            lr=args.lr,
        )

    # Verify outputs
    print("\nVerifying outputs:")
    for train_file in train_files:
        domain = train_file.parent.name
        adapter_dir = output_dir / domain
        has_config = (adapter_dir / "adapter_config.json").exists()
        has_weights = (adapter_dir / "adapter_model.safetensors").exists()
        status = "OK" if (has_config and has_weights) else "MISSING"
        print(f"  {domain}: {status}")


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Distillation pipeline: generate data + train experts")
    sub = parser.add_subparsers(dest="command")

    # Generate subcommand
    p_gen = sub.add_parser("generate", help="Generate training data via API")
    p_gen.add_argument("--domains", help="YAML file with domain->description mapping")
    p_gen.add_argument("--output", default="data/distillation/",
                       help="Output directory (default: data/distillation/)")
    p_gen.add_argument("--model", default="llama-3.3-70b-versatile",
                       help="Teacher model (default: llama-3.3-70b-versatile)")
    p_gen.add_argument("--base-url", default="https://api.groq.com/openai/v1",
                       help="API base URL (default: Groq)")
    p_gen.add_argument("--api-key-env", default="GROQ_API_KEY",
                       help="Env var for API key (default: GROQ_API_KEY)")
    p_gen.add_argument("--n-examples", type=int, default=1000,
                       help="Examples per domain (default: 1000)")

    # Download subcommand
    p_dl = sub.add_parser("download", help="Download training data from HuggingFace")
    p_dl.add_argument("--output", default="data/distillation/",
                      help="Output directory (default: data/distillation/)")
    p_dl.add_argument("--n-examples", type=int, default=1000,
                      help="Examples per domain (default: 1000)")
    p_dl.add_argument("--domains-list", nargs="+",
                      help=f"Domains to download (default: {list(HF_DATASETS.keys())})")

    # Train subcommand
    p_train = sub.add_parser("train", help="Train QLoRA experts from data")
    p_train.add_argument("--data", default="data/distillation/",
                         help="Data directory with domain/train.jsonl files")
    p_train.add_argument("--base", default="Qwen/Qwen2.5-7B",
                         help="Base model (default: Qwen/Qwen2.5-7B)")
    p_train.add_argument("--output", default="adapters/",
                         help="Output directory for adapters")
    p_train.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    p_train.add_argument("--steps", type=int, default=300, help="Training steps (default: 300)")
    p_train.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4)")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"generate": cmd_generate, "train": cmd_train, "download": cmd_download}[args.command](args)


if __name__ == "__main__":
    main()
