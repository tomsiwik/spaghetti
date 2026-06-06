#!/usr/bin/env python3
"""Convert MLX-trained LoRA adapters (npz) to llama.cpp GGUF LoRA format.

The GGUF LoRA format stores lora_a and lora_b tensors with GGUF-convention
tensor names: blk.{layer}.{component}.weight.loraA / .loraB

Reference: llama.cpp convert_lora_to_gguf.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import gguf


# HuggingFace name -> GGUF name mapping for BitNet
HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}


def convert_adapter(
    adapter_path: Path,
    base_config_path: Path,
    output_path: Path,
    lora_alpha: float = 16.0,
):
    """Convert a single npz adapter to GGUF LoRA format."""
    # Load adapter
    data = np.load(adapter_path)

    # Load base model config for architecture info
    with open(base_config_path) as f:
        config = json.load(f)

    # Create GGUF writer
    writer = gguf.GGUFWriter(str(output_path), "llama")  # arch name

    # Write required metadata
    # llama.cpp expects these for LoRA adapters
    writer.add_string("general.type", "adapter")
    writer.add_string("general.architecture", "bitnet")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", lora_alpha)

    # Process each tensor
    tensor_count = 0
    for key in sorted(data.keys()):
        arr = data[key]

        # Parse the key: model.layers.{bid}.{component}.lora_{a|b}
        parts = key.split(".")
        if "lora_a" not in key and "lora_b" not in key:
            print(f"  Skipping non-LoRA tensor: {key}")
            continue

        # Extract layer index
        layer_idx = int(parts[2])  # model.layers.{idx}

        # Extract component (e.g., self_attn.q_proj)
        # key format: model.layers.{bid}.{module}.{submodule}.lora_{a|b}
        component = f"{parts[3]}.{parts[4]}"
        lora_type = parts[5]  # lora_a or lora_b

        # Map to GGUF name
        if component not in HF_TO_GGUF:
            print(f"  Unknown component: {component}, skipping")
            continue

        gguf_component = HF_TO_GGUF[component]

        # GGUF LoRA tensor naming convention (from llama-adapter.cpp):
        # blk.{layer}.{component}.weight.lora_a  (shape: rank x in_features)
        # blk.{layer}.{component}.weight.lora_b  (shape: out_features x rank)
        if lora_type == "lora_a":
            # Our npz stores lora_a as (in_features, rank)
            # llama.cpp LoRA: A is (rank, in_features)
            tensor_name = f"blk.{layer_idx}.{gguf_component}.weight.lora_a"
            # Transpose: our (in, rank) -> GGUF (rank, in)
            arr = arr.T.copy()
        elif lora_type == "lora_b":
            tensor_name = f"blk.{layer_idx}.{gguf_component}.weight.lora_b"
            # Our npz stores lora_b as (rank, out_features)
            # GGUF expects loraB as (out_features, rank)
            arr = arr.T.copy()
        else:
            continue

        # Convert to float32
        arr = arr.astype(np.float32)

        writer.add_tensor(tensor_name, arr)
        tensor_count += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"  Written {tensor_count} tensors to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024 * 1024):.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert MLX LoRA npz to GGUF")
    parser.add_argument("adapter_dir", type=Path, help="Directory containing adapter.npz files")
    parser.add_argument("--base-config", type=Path, required=True, help="Path to base model config.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for GGUF files")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (default: 16)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all adapter.npz files
    adapter_files = list(args.adapter_dir.glob("*/adapter.npz"))
    if not adapter_files:
        # Try the directory itself
        if (args.adapter_dir / "adapter.npz").exists():
            adapter_files = [args.adapter_dir / "adapter.npz"]

    print(f"Found {len(adapter_files)} adapters")

    for adapter_path in sorted(adapter_files):
        domain = adapter_path.parent.name
        output_path = args.output_dir / f"{domain}.gguf"
        print(f"\nConverting {domain}...")
        convert_adapter(adapter_path, args.base_config, output_path, args.alpha)


if __name__ == "__main__":
    main()
