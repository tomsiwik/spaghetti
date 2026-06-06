#!/usr/bin/env python3
"""Clone-and-compete expert evolution.

Implements the living model's self-improvement mechanism:
1. Clone an expert
2. Fine-tune clone with corrections (50-100 steps)
3. Register both on hash ring
4. Shadow score: compare perplexity on real queries
5. Prune the loser

Usage:
    python -m composer.evolve tournament \
        --expert adapters/python/ \
        --corrections data/corrections/python_fixes.jsonl \
        --eval-queries 1000 \
        --output results/tournament.json
"""

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path


def clone_expert(expert_dir: Path, clone_dir: Path):
    """Clone an expert adapter directory."""
    if clone_dir.exists():
        shutil.rmtree(clone_dir)
    shutil.copytree(expert_dir, clone_dir)
    print(f"  Cloned {expert_dir.name} → {clone_dir.name}")


def finetune_clone(base_model_name: str, clone_dir: Path, corrections_file: Path,
                   steps: int = 50, lr: float = 1e-4):
    """Fine-tune a cloned adapter with correction data."""
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    print(f"  Fine-tuning clone with {corrections_file} ({steps} steps)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_name,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    # Load existing adapter weights into the model
    model = PeftModel.from_pretrained(model, str(clone_dir), is_trainable=True)

    # Load correction data
    dataset = load_dataset("json", data_files=str(corrections_file), split="train")

    def format_messages(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(format_messages)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(clone_dir / "ft_checkpoints"),
            max_steps=steps,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=lr,
            warmup_steps=5,
            logging_steps=10,
            save_steps=steps,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataset_text_field="text",
            max_length=2048,
        ),
    )

    trainer.train()
    model.save_pretrained(str(clone_dir))
    tokenizer.save_pretrained(str(clone_dir))

    # Cleanup
    ft_ckpt = clone_dir / "ft_checkpoints"
    if ft_ckpt.exists():
        shutil.rmtree(ft_ckpt)

    print(f"  Clone fine-tuned and saved to {clone_dir}")


def shadow_score(base_model, tokenizer, adapter_a_dir: Path, adapter_b_dir: Path,
                 eval_texts: list[str]) -> dict:
    """Compare two adapters via perplexity on eval queries.

    Returns dict with per-adapter scores and winner.
    """
    import torch
    from peft import PeftModel

    results = {"a_name": adapter_a_dir.name, "b_name": adapter_b_dir.name,
               "a_scores": [], "b_scores": []}

    # Score adapter A
    model_a = PeftModel.from_pretrained(base_model, str(adapter_a_dir))
    model_a.eval()
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(base_model.device)
            outputs = model_a(**inputs, labels=inputs["input_ids"])
            results["a_scores"].append(outputs.loss.item())
    del model_a
    torch.cuda.empty_cache()

    # Score adapter B
    model_b = PeftModel.from_pretrained(base_model, str(adapter_b_dir))
    model_b.eval()
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=512).to(base_model.device)
            outputs = model_b(**inputs, labels=inputs["input_ids"])
            results["b_scores"].append(outputs.loss.item())
    del model_b
    torch.cuda.empty_cache()

    # Aggregate
    avg_a = sum(results["a_scores"]) / len(results["a_scores"])
    avg_b = sum(results["b_scores"]) / len(results["b_scores"])
    results["avg_loss_a"] = avg_a
    results["avg_loss_b"] = avg_b
    results["ppl_a"] = math.exp(avg_a)
    results["ppl_b"] = math.exp(avg_b)
    results["winner"] = "a" if avg_a < avg_b else "b"
    results["winner_name"] = results["a_name"] if avg_a < avg_b else results["b_name"]
    results["margin"] = abs(avg_a - avg_b)

    return results


def cmd_tournament(args):
    """Run a clone-and-compete tournament."""
    expert_dir = Path(args.expert)
    corrections_file = Path(args.corrections)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    domain = expert_dir.name
    clone_dir = expert_dir.parent / f"{domain}_clone"

    print(f"Tournament: {domain}")
    print(f"  Expert: {expert_dir}")
    print(f"  Corrections: {corrections_file}")

    # Step 1: Clone
    clone_expert(expert_dir, clone_dir)

    # Step 2: Fine-tune clone with corrections
    finetune_clone(args.base, clone_dir, corrections_file,
                   steps=args.ft_steps, lr=args.ft_lr)

    # Step 3: Load eval queries
    eval_texts = []
    if args.eval_data:
        eval_file = Path(args.eval_data) / domain / "eval.jsonl"
        if eval_file.exists():
            with open(eval_file) as f:
                for line in f:
                    record = json.loads(line)
                    if "text" in record:
                        eval_texts.append(record["text"])
                    elif "messages" in record:
                        parts = [f"{m['role']}: {m['content']}" for m in record["messages"]]
                        eval_texts.append("\n".join(parts))
                    if len(eval_texts) >= args.eval_queries:
                        break

    # Also load corrections as eval (clone should win on these)
    correction_texts = []
    with open(corrections_file) as f:
        for line in f:
            record = json.loads(line)
            if "messages" in record:
                parts = [f"{m['role']}: {m['content']}" for m in record["messages"]]
                correction_texts.append("\n".join(parts))

    if not eval_texts and not correction_texts:
        print("No eval data found!")
        sys.exit(1)

    # Step 4: Shadow score
    print(f"\n  Shadow scoring on {len(eval_texts)} general + {len(correction_texts)} correction queries...")
    print("  Loading base model...")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto")

    tournament = {"domain": domain, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Score on correction queries (clone should win here)
    if correction_texts:
        print("\n  Scoring on correction queries...")
        corr_results = shadow_score(base_model, tokenizer, expert_dir, clone_dir, correction_texts)
        tournament["correction_queries"] = {
            "n_queries": len(correction_texts),
            "original_ppl": corr_results["ppl_a"],
            "clone_ppl": corr_results["ppl_b"],
            "winner": "clone" if corr_results["winner"] == "b" else "original",
            "margin": corr_results["margin"],
        }
        print(f"    Original PPL: {corr_results['ppl_a']:.2f}")
        print(f"    Clone PPL:    {corr_results['ppl_b']:.2f}")
        print(f"    Winner: {'CLONE' if corr_results['winner'] == 'b' else 'ORIGINAL'}")

    # Score on general queries (clone should not regress)
    if eval_texts:
        print("\n  Scoring on general queries...")
        gen_results = shadow_score(base_model, tokenizer, expert_dir, clone_dir, eval_texts)
        regression = (gen_results["ppl_b"] - gen_results["ppl_a"]) / gen_results["ppl_a"] * 100
        tournament["general_queries"] = {
            "n_queries": len(eval_texts),
            "original_ppl": gen_results["ppl_a"],
            "clone_ppl": gen_results["ppl_b"],
            "regression_pct": regression,
            "regressed": regression > 2.0,
        }
        print(f"    Original PPL: {gen_results['ppl_a']:.2f}")
        print(f"    Clone PPL:    {gen_results['ppl_b']:.2f}")
        print(f"    Regression:   {regression:+.1f}%")

    # Kill criteria assessment
    clone_wins_corrections = tournament.get("correction_queries", {}).get("winner") == "clone"
    no_regression = not tournament.get("general_queries", {}).get("regressed", True)

    tournament["verdict"] = {
        "clone_wins_corrections": clone_wins_corrections,
        "no_regression": no_regression,
        "overall": "PASS" if (clone_wins_corrections and no_regression) else "FAIL",
    }

    print(f"\n{'='*50}")
    print(f"Clone wins on corrections: {'YES' if clone_wins_corrections else 'NO'}")
    print(f"No regression on general:  {'YES' if no_regression else 'NO'}")
    print(f"Overall: {tournament['verdict']['overall']}")

    # Step 5: Prune loser (if tournament passes)
    if tournament["verdict"]["overall"] == "PASS" and not args.dry_run:
        print(f"\n  Promoting clone: {clone_dir.name} → {expert_dir.name}")
        shutil.rmtree(expert_dir)
        clone_dir.rename(expert_dir)
        tournament["action"] = "clone_promoted"
    else:
        if clone_dir.exists() and not args.keep_clone:
            shutil.rmtree(clone_dir)
            tournament["action"] = "clone_discarded"
        else:
            tournament["action"] = "dry_run" if args.dry_run else "clone_kept"

    with open(output_path, "w") as f:
        json.dump(tournament, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clone-and-compete expert evolution")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("tournament", help="Run clone-and-compete tournament")
    p.add_argument("--expert", required=True, help="Path to expert adapter directory")
    p.add_argument("--corrections", required=True, help="JSONL file with correction examples")
    p.add_argument("--base", default="Qwen/Qwen2.5-7B", help="Base model")
    p.add_argument("--eval-data", default="eval/", help="Eval data directory")
    p.add_argument("--eval-queries", type=int, default=1000, help="Max eval queries")
    p.add_argument("--ft-steps", type=int, default=50, help="Fine-tuning steps for clone")
    p.add_argument("--ft-lr", type=float, default=1e-4, help="Fine-tuning LR for clone")
    p.add_argument("--output", default="results/tournament.json", help="Output JSON")
    p.add_argument("--dry-run", action="store_true", help="Don't prune loser")
    p.add_argument("--keep-clone", action="store_true", help="Keep clone even if it loses")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    {"tournament": cmd_tournament}[args.command](args)


if __name__ == "__main__":
    main()
