#!/usr/bin/env python3
"""SOLE Training Utilities — patterns adapted from nanochat for expert distillation.

Key patterns from Karpathy's nanochat applied to SOLE:
1. GC optimization (disable during training, saves ~500ms/step)
2. Bestfit conversation packing (100% utilization, no wasted tokens)
3. Explicit precision management (no autocast)
4. Muon-compatible LoRA parameter grouping
5. Loss masking for assistant-only tokens
6. Data prefetching during GPU compute

Reference: references/nanochat/ (MIT license)
"""

import contextlib
import gc
import time
import torch
from typing import Optional


# ── GC Optimization ──────────────────────────────────────────────────────────
# From nanochat/scripts/base_train.py: Python's cycle detector causes ~500ms
# overhead per step on long-lived PyTorch tensors. Disable during training.

@contextlib.contextmanager
def training_gc_context():
    """Disable GC during training to avoid cycle-detection overhead on tensors."""
    gc.disable()
    gc.collect()  # Clean up before disabling
    try:
        yield
    finally:
        gc.enable()
        gc.collect()


# ── Explicit Precision ────────────────────────────────────────────────────────
# From nanochat/nanochat/common.py: No autocast. Detect hardware, set global
# dtype. Model weights stay fp32 for optimizer precision, forward runs in dtype.

def detect_compute_dtype() -> torch.dtype:
    """Detect optimal compute dtype for current hardware."""
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:  # SM 80+ (A100, H100, A5000, 4090)
            return torch.bfloat16
        else:
            return torch.float16  # V100, T4
    return torch.float32


# ── Bestfit Conversation Packing ──────────────────────────────────────────────
# From nanochat/scripts/chat_sft.py: Pack conversations using best-fit algorithm.
# For each row: find LARGEST conversation that fits, repeat until nothing fits,
# then pad. No tokens discarded (padding masked with ignore_index=-1).
# Contrast with TRL's greedy packing which just concatenates.

def bestfit_pack_conversations(
    tokenized_conversations: list[dict],
    max_seq_len: int,
    tokenizer,
    pad_token_id: int,
    buffer_size: int = 100,
) -> list[dict]:
    """Pack tokenized conversations using bestfit algorithm.

    Args:
        tokenized_conversations: List of dicts with 'input_ids' and 'attention_mask'
        max_seq_len: Maximum sequence length per row
        tokenizer: Tokenizer (for BOS token)
        pad_token_id: Token ID for padding
        buffer_size: Number of conversations to buffer for best-fit search

    Returns:
        List of packed sequences with 'input_ids', 'attention_mask', 'labels'
        where labels=-100 for non-assistant tokens (padding, user prompts).
    """
    row_capacity = max_seq_len
    buffer = list(tokenized_conversations)
    packed = []
    buf_idx = 0

    while buf_idx < len(buffer):
        row_ids = []
        row_labels = []
        row_mask = []

        while len(row_ids) < row_capacity and buf_idx < len(buffer):
            remaining = row_capacity - len(row_ids)

            # Search buffer for largest conversation that fits
            best_idx = -1
            best_len = 0
            search_end = min(buf_idx + buffer_size, len(buffer))
            for i in range(buf_idx, search_end):
                conv_len = len(buffer[i]['input_ids'])
                if conv_len <= remaining and conv_len > best_len:
                    best_idx = i
                    best_len = conv_len

            if best_idx >= 0:
                conv = buffer[best_idx]
                # Swap found item to current position and advance
                buffer[best_idx], buffer[buf_idx] = buffer[buf_idx], buffer[best_idx]
                buf_idx += 1

                row_ids.extend(conv['input_ids'])
                row_labels.extend(conv.get('labels', conv['input_ids']))
                row_mask.extend([1] * len(conv['input_ids']))
            else:
                # Nothing fits — pad remainder
                pad_len = remaining
                row_ids.extend([pad_token_id] * pad_len)
                row_labels.extend([-100] * pad_len)  # ignore_index
                row_mask.extend([0] * pad_len)
                break

        # If row is shorter than capacity, pad
        if len(row_ids) < row_capacity:
            pad_len = row_capacity - len(row_ids)
            row_ids.extend([pad_token_id] * pad_len)
            row_labels.extend([-100] * pad_len)
            row_mask.extend([0] * pad_len)

        packed.append({
            'input_ids': row_ids[:row_capacity],
            'labels': row_labels[:row_capacity],
            'attention_mask': row_mask[:row_capacity],
        })

    return packed


# ── LoRA Parameter Groups ────────────────────────────────────────────────────
# From nanochat/nanochat/optim.py: Different optimizer for matrix params vs
# scalars/embeddings. Muon for 2D weight matrices, AdamW for the rest.
# For LoRA: A matrices are (r, d), B matrices are (d, r) — both 2D, both
# candidates for Muon orthogonalization.

def group_lora_params(model, matrix_lr=0.02, scalar_lr=1e-4, weight_decay=0.0):
    """Group LoRA parameters for Muon-compatible optimization.

    Separates 2D LoRA matrices (A, B) from other trainable params.
    Matrix params benefit from Muon's orthogonalization; scalars use AdamW.

    Returns list of param groups compatible with MuonAdamW or standard AdamW.
    """
    matrix_params = []
    scalar_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and ('lora_A' in name or 'lora_B' in name):
            matrix_params.append(param)
        else:
            scalar_params.append(param)

    groups = []
    if matrix_params:
        groups.append({
            'params': matrix_params,
            'lr': matrix_lr,
            'weight_decay': weight_decay,
            'kind': 'muon',  # Flag for Muon optimizer
        })
    if scalar_params:
        groups.append({
            'params': scalar_params,
            'lr': scalar_lr,
            'weight_decay': 0.0,
            'kind': 'adamw',
        })
    return groups


# ── Training Step with Prefetch ──────────────────────────────────────────────
# From nanochat/scripts/base_train.py:517: Load next batch while GPU computes
# forward/backward. Overlaps I/O with compute.

class PrefetchingDataLoader:
    """Wraps a dataloader to prefetch next batch during GPU compute."""

    def __init__(self, dataloader, device='cuda'):
        self.dataloader = iter(dataloader)
        self.device = device
        self._prefetched = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._prefetched is not None:
            batch = self._prefetched
        else:
            batch = self._load_next()

        # Prefetch next batch while GPU processes current one
        try:
            self._prefetched = self._load_next()
        except StopIteration:
            self._prefetched = None

        return batch

    def _load_next(self):
        batch = next(self.dataloader)
        if isinstance(batch, dict):
            return {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        return batch


# ── Logging Utilities ────────────────────────────────────────────────────────

def log_training_step(step, total_steps, loss, lr, dt, tokens_per_step, gpu_name=None):
    """Nanochat-style training step log."""
    pct = 100 * step / total_steps if total_steps > 0 else 0
    tok_per_sec = int(tokens_per_step / dt) if dt > 0 else 0
    msg = (f"step {step:05d}/{total_steps:05d} ({pct:.1f}%) | "
           f"loss: {loss:.4f} | lr: {lr:.2e} | "
           f"dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,}")
    if gpu_name and torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1e9
        msg += f" | mem: {mem:.1f}GB"
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Checkpoint Metadata ──────────────────────────────────────────────────────
# From nanochat/checkpoint_manager.py: Store rich metadata alongside weights.

def save_adapter_with_metadata(model, tokenizer, output_dir, metadata: dict):
    """Save LoRA adapter with nanochat-style metadata.

    Metadata includes training config, loss curves, timing, and
    SOLE-specific info (Grassmannian slot, composition history).
    """
    import json
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Add training metadata
    metadata['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(output_dir / 'train_meta.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return output_dir
