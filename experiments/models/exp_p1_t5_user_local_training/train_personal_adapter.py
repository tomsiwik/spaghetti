#!/usr/bin/env python3
"""
Train a personal adapter from your conversation history.

Usage:
    uv run python train_personal_adapter.py \\
        --data my_conversations.jsonl \\
        --output my_adapter/

Data format (one JSON per line):
    {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def prepare_data(input_file: Path, output_dir: Path, val_split: float = 0.1) -> tuple:
    """Convert conversations.jsonl → mlx_lm train.jsonl + valid.jsonl."""
    examples = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    n_val = max(1, int(len(examples) * val_split))
    train_examples = examples[:-n_val]
    valid_examples = examples[-n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(output_dir / "valid.jsonl", "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Data: {len(train_examples)} train / {len(valid_examples)} valid")
    return train_examples, valid_examples


def write_config(config_path: Path, data_dir: Path, adapter_path: Path,
                 iters: int, rank: int) -> None:
    """Write mlx_lm.lora YAML config."""
    config = {
        "model": "mlx-community/gemma-4-e4b-it-4bit",
        "data": str(data_dir),
        "adapter_path": str(adapter_path),
        "train": True,
        "fine_tune_type": "lora",
        "iters": iters,
        "batch_size": 2,
        "num_layers": 16,
        "learning_rate": 1e-4,
        "lora_parameters": {
            "rank": rank,
            "scale": float(rank),
            "dropout": 0.0,
            "keys": ["self_attn.q_proj"],
        },
        "max_seq_length": 256,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "save_every": iters,
        "steps_per_report": 50,
        "seed": 42,
    }
    import yaml  # type: ignore
    with open(config_path, "w") as f:
        yaml.dump(config, f)


def train(config_path: Path) -> int:
    """Run mlx_lm.lora training. Returns exit code."""
    cmd = ["uv", "run", "python", "-m", "mlx_lm.lora", "--config", str(config_path)]
    print("Training personal adapter (this takes a few minutes)...")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Train a personal adapter")
    parser.add_argument("--data", required=True, help="Path to conversations.jsonl")
    parser.add_argument("--output", default="my_adapter", help="Output adapter directory")
    parser.add_argument("--iters", type=int, default=300, help="Training iterations (default 300)")
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (default 4)")
    args = parser.parse_args()

    input_file = Path(args.data)
    if not input_file.exists():
        print(f"Error: {input_file} not found", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output)

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "data"
        config_path = Path(tmpdir) / "config.yaml"

        prepare_data(input_file, data_dir)
        write_config(config_path, data_dir, output_dir, args.iters, args.rank)
        rc = train(config_path)

    if rc != 0:
        print("Training failed.", file=sys.stderr)
        sys.exit(rc)

    # Report adapter size
    safetensors = list(output_dir.glob("*.safetensors"))
    if safetensors:
        size_bytes = sum(f.stat().st_size for f in safetensors)
        size_mb = size_bytes / (1024 ** 2)
        print(f"Adapter saved: {size_mb:.2f}MB → {output_dir}/")
    else:
        print(f"Adapter saved → {output_dir}/")

    print("Done. Load with: mlx_lm.generate --model mlx-community/gemma-4-e4b-it-4bit "
          f"--adapter-path {output_dir}")


if __name__ == "__main__":
    main()
