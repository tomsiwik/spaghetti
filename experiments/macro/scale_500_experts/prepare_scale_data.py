#!/usr/bin/env python3
"""Prepare training data for 450 new domain experts (scaling from 50 to 500).

Downloads Open-Orca/SlimOrca from HuggingFace, partitions into 450 domain
buckets, and saves as train.jsonl files compatible with pilot50_train.py.

Existing pilot50 domains (in data/distillation/) are preserved. New domains
are named domain_050 through domain_499.

Supports SMOKE_TEST=1 for quick validation (3 domains, 10 examples each).

Usage (on RunPod):
    python macro/scale_500_experts/prepare_scale_data.py
"""

import json
import os
import sys
import time
from pathlib import Path

IS_SMOKE = os.environ.get("SMOKE_TEST") == "1"

REPO_ROOT = Path("/workspace/llm")
DATA_DIR = REPO_ROOT / "data" / "scale_500"
EXAMPLES_PER_DOMAIN = 10 if IS_SMOKE else 300
N_NEW_DOMAINS = 3 if IS_SMOKE else 450
DATASET_NAME = "Open-Orca/SlimOrca"


def log(msg):
    ts = time.strftime("%H:%M:%S", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def main():
    from datasets import load_dataset

    t0 = time.time()
    log("=" * 60)
    log(f"Scale 500 Data Preparation")
    log(f"  New domains: {N_NEW_DOMAINS}")
    log(f"  Examples/domain: {EXAMPLES_PER_DOMAIN}")
    log(f"  Total examples needed: {N_NEW_DOMAINS * EXAMPLES_PER_DOMAIN}")
    log(f"  Dataset: {DATASET_NAME}")
    log(f"  Smoke test: {IS_SMOKE}")
    log("=" * 60)

    # Check what already exists
    existing = 0
    for i in range(N_NEW_DOMAINS):
        domain = f"domain_{i + 50:03d}"
        train_file = DATA_DIR / domain / "train.jsonl"
        if train_file.exists():
            with open(train_file) as f:
                n = sum(1 for _ in f)
            if n >= EXAMPLES_PER_DOMAIN:
                existing += 1
    if existing == N_NEW_DOMAINS:
        log(f"All {N_NEW_DOMAINS} domains already prepared. Nothing to do.")
        return
    log(f"Already prepared: {existing}/{N_NEW_DOMAINS}")

    # Download dataset
    log(f"\nDownloading {DATASET_NAME}...")
    ds = load_dataset(DATASET_NAME, split="train",
                      cache_dir="/workspace/hf_cache")
    log(f"Dataset size: {len(ds)}")

    # Shuffle deterministically
    ds = ds.shuffle(seed=42)

    # Take what we need
    total_needed = N_NEW_DOMAINS * EXAMPLES_PER_DOMAIN
    if len(ds) < total_needed:
        log(f"WARNING: dataset has {len(ds)} < {total_needed} needed. "
            f"Reducing examples/domain.")
        examples_per = len(ds) // N_NEW_DOMAINS
    else:
        examples_per = EXAMPLES_PER_DOMAIN

    # Partition into domain buckets and save
    log(f"\nPartitioning into {N_NEW_DOMAINS} domains...")
    for i in range(N_NEW_DOMAINS):
        domain = f"domain_{i + 50:03d}"
        out_dir = DATA_DIR / domain
        train_file = out_dir / "train.jsonl"

        # Skip if already done
        if train_file.exists():
            with open(train_file) as f:
                n = sum(1 for _ in f)
            if n >= examples_per:
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        start_idx = i * examples_per
        end_idx = start_idx + examples_per

        with open(train_file, "w") as f:
            for j in range(start_idx, min(end_idx, len(ds))):
                row = ds[j]
                # SlimOrca uses 'conversations' list with {from, value} dicts
                convos = row["conversations"]
                role_map = {"system": "system", "human": "user", "gpt": "assistant"}
                messages = []
                for turn in convos:
                    role = role_map.get(turn["from"], turn["from"])
                    content = turn["value"].strip()
                    if content:
                        messages.append({"role": role, "content": content})
                if messages:
                    f.write(json.dumps({"messages": messages}) + "\n")

        if (i + 1) % 50 == 0 or i == N_NEW_DOMAINS - 1:
            log(f"  Prepared {i + 1}/{N_NEW_DOMAINS} domains")

    # Verify
    log("\nVerification:")
    complete = 0
    for i in range(N_NEW_DOMAINS):
        domain = f"domain_{i + 50:03d}"
        train_file = DATA_DIR / domain / "train.jsonl"
        if train_file.exists():
            with open(train_file) as f:
                n = sum(1 for _ in f)
            if n >= examples_per:
                complete += 1
    log(f"  Complete: {complete}/{N_NEW_DOMAINS}")

    elapsed = time.time() - t0
    log(f"\nData preparation done in {elapsed:.0f}s")

    # Save metadata
    meta = {
        "n_domains": N_NEW_DOMAINS,
        "examples_per_domain": examples_per,
        "dataset": DATASET_NAME,
        "complete": complete,
        "elapsed_s": round(elapsed, 1),
    }
    meta_file = DATA_DIR / "prep_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"Metadata saved to {meta_file}")


if __name__ == "__main__":
    main()
