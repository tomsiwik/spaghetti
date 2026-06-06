"""LoRA Orthogonality at Real Scale — Qwen2.5-0.5B.

Validates N_max proportional to d^2/r^2 by measuring pairwise cosine similarity
of LoRA deltas at ranks 4, 8, 16, 32 with 10 adapters.

Output: macro/ortho_scaling/results.json
"""

import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
RANKS = [4, 8, 16, 32]
N_ADAPTERS = 10
FINETUNE_STEPS = 100
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
LR = 2e-4
OUT_DIR = Path(__file__).parent / "ortho_scaling"


def create_domain_data(tokenizer, domain_id, n_samples=200, seed=42):
    rng = random.Random(seed + domain_id * 7)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=HF_HOME)
        texts = [t for t in ds["text"] if len(t) > 100]
        # Different slice per domain
        start = (domain_id * n_samples) % max(1, len(texts) - n_samples)
        texts = texts[start:start + n_samples]
    except Exception:
        words = "the a is was in of to and for on with at by from this that".split()
        texts = [" ".join(rng.choices(words, k=rng.randint(50, 200)))
                 for _ in range(n_samples)]

    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True,
                              max_length=MAX_SEQ_LEN + 1)
        if len(ids) > 10:
            encodings.append(ids)
    return encodings


def get_batch(encodings, batch_size, rng, device=DEVICE):
    seqs = rng.choices(encodings, k=batch_size)
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        input_ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, :len(seq)] = torch.tensor(seq)
        labels[i, 0] = -100
        attention_mask[i, :len(seq)] = 1
    return input_ids, labels, attention_mask


def get_lora_delta_flat(model):
    deltas = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            deltas.append(param.data.detach().reshape(-1))
    return torch.cat(deltas) if deltas else torch.zeros(1, device=DEVICE)


def get_lora_delta_per_layer(model):
    """Get LoRA deltas grouped by layer."""
    layer_deltas = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            # Extract layer index from name
            parts = name.split(".")
            layer_idx = None
            for i, p in enumerate(parts):
                if p == "layers" and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
            if layer_idx is not None:
                if layer_idx not in layer_deltas:
                    layer_deltas[layer_idx] = []
                layer_deltas[layer_idx].append(param.data.detach().reshape(-1))

    return {k: torch.cat(v) for k, v in sorted(layer_deltas.items())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"LoRA Orthogonality Scaling — {MODEL_NAME}")
    print(f"Ranks: {RANKS}, N_adapters: {N_ADAPTERS}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create domain data for each adapter
    domain_data = {}
    for i in range(N_ADAPTERS):
        domain_data[i] = create_domain_data(tokenizer, i, n_samples=200)
        print(f"  Domain {i}: {len(domain_data[i])} samples")

    results = {
        'model': MODEL_NAME,
        'n_adapters': N_ADAPTERS,
        'finetune_steps': FINETUNE_STEPS,
        'rank_results': {},
    }

    for rank in RANKS:
        print(f"\n{'='*70}")
        print(f"RANK = {rank}")
        print(f"{'='*70}")

        # Fine-tune N_ADAPTERS LoRA adapters at this rank
        all_deltas_flat = []
        all_deltas_per_layer = []

        for adapter_idx in range(N_ADAPTERS):
            print(f"\n  --- Adapter {adapter_idx}/{N_ADAPTERS} (rank={rank}) ---")
            torch.manual_seed(42 + adapter_idx + rank * 100)

            base_model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(DEVICE)

            lora_config = LoraConfig(
                r=rank, lora_alpha=rank,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                              "up_proj", "gate_proj", "down_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(base_model, lora_config)

            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=LR
            )
            rng = random.Random(42 + adapter_idx)
            model.train()
            for step in range(1, FINETUNE_STEPS + 1):
                input_ids, labels, attention_mask = get_batch(
                    domain_data[adapter_idx], BATCH_SIZE, rng)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1].contiguous()
                targets = labels[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      targets.view(-1), ignore_index=-100)
                loss.backward()
                optimizer.step()

            flat_delta = get_lora_delta_flat(model)
            per_layer = get_lora_delta_per_layer(model)
            all_deltas_flat.append(flat_delta)
            all_deltas_per_layer.append(per_layer)

            print(f"    Delta norm: {flat_delta.norm().item():.4f}, "
                  f"dim: {flat_delta.numel()}")

            del model, base_model, optimizer
            torch.cuda.empty_cache()

        # Compute all pairwise cosines (global)
        print(f"\n  Computing pairwise cosines (global)...")
        global_cosines = []
        for i in range(N_ADAPTERS):
            for j in range(i + 1, N_ADAPTERS):
                cos = F.cosine_similarity(
                    all_deltas_flat[i].unsqueeze(0),
                    all_deltas_flat[j].unsqueeze(0)
                ).item()
                global_cosines.append(cos)

        D_global = all_deltas_flat[0].numel()
        expected_global = rank / math.sqrt(D_global)

        print(f"  Global cosines: mean={statistics.mean(global_cosines):.6f}, "
              f"std={statistics.stdev(global_cosines):.6f}")
        print(f"  Expected (r/sqrt(D)): {expected_global:.6f}, D={D_global}")

        # Per-layer cosines
        print(f"\n  Computing per-layer cosines...")
        layer_indices = sorted(all_deltas_per_layer[0].keys())
        per_layer_stats = {}

        for l_idx in layer_indices:
            layer_cosines = []
            for i in range(N_ADAPTERS):
                for j in range(i + 1, N_ADAPTERS):
                    if l_idx in all_deltas_per_layer[i] and l_idx in all_deltas_per_layer[j]:
                        cos = F.cosine_similarity(
                            all_deltas_per_layer[i][l_idx].unsqueeze(0),
                            all_deltas_per_layer[j][l_idx].unsqueeze(0)
                        ).item()
                        layer_cosines.append(cos)

            if layer_cosines:
                D_layer = all_deltas_per_layer[0][l_idx].numel()
                expected_layer = rank / math.sqrt(D_layer)
                per_layer_stats[l_idx] = {
                    'mean_cos': statistics.mean(layer_cosines),
                    'std_cos': statistics.stdev(layer_cosines) if len(layer_cosines) > 1 else 0,
                    'D': D_layer,
                    'expected': expected_layer,
                }

        # Print per-layer summary (first and last 3)
        for l_idx in layer_indices[:3] + layer_indices[-3:]:
            if l_idx in per_layer_stats:
                s = per_layer_stats[l_idx]
                print(f"    Layer {l_idx}: cos={s['mean_cos']:.6f}+{s['std_cos']:.4f}, "
                      f"D={s['D']}, expected={s['expected']:.6f}")

        results['rank_results'][rank] = {
            'global_mean_cos': statistics.mean(global_cosines),
            'global_std_cos': statistics.stdev(global_cosines),
            'global_max_cos': max(global_cosines),
            'global_min_cos': min(global_cosines),
            'D_global': D_global,
            'expected_cos': expected_global,
            'ratio_actual_expected': statistics.mean(global_cosines) / expected_global if expected_global > 0 else 0,
            'n_pairs': len(global_cosines),
            'per_layer': {str(k): v for k, v in per_layer_stats.items()},
        }

    # Scaling analysis: does cos scale as r/sqrt(D)?
    print(f"\n{'='*70}")
    print("SCALING ANALYSIS: cos vs r/sqrt(D)")
    print(f"{'='*70}")

    for rank in RANKS:
        rr = results['rank_results'][rank]
        ratio = rr['ratio_actual_expected']
        print(f"  r={rank:2d}: actual={rr['global_mean_cos']:.6f}, "
              f"predicted={rr['expected_cos']:.6f}, ratio={ratio:.2f}")

    # Check if ratio is roughly constant across ranks
    ratios = [results['rank_results'][r]['ratio_actual_expected'] for r in RANKS]
    ratio_cv = statistics.stdev(ratios) / statistics.mean(ratios) if statistics.mean(ratios) > 0 else float('inf')
    print(f"\n  Ratio CV (should be small): {ratio_cv:.4f}")

    # Check scaling: cos(r=32) / cos(r=4) should be ~8 (= 32/4)
    if 4 in results['rank_results'] and 32 in results['rank_results']:
        cos_ratio = (results['rank_results'][32]['global_mean_cos'] /
                    results['rank_results'][4]['global_mean_cos'])
        print(f"  cos(r=32)/cos(r=4) = {cos_ratio:.2f} (predicted: 8.0)")

    # Kill criteria
    print(f"\n{'='*70}")
    print("KILL CRITERIA")
    print(f"{'='*70}")

    r16 = results['rank_results'].get(16, {})
    if r16:
        mean_cos = r16['global_mean_cos']
        if mean_cos < 0.05:
            print(f"  [PASS] Near-orthogonal at r=16: cos={mean_cos:.6f} < 0.05")
        else:
            print(f"  [WARN] cos={mean_cos:.6f} at r=16 (expected < 0.05)")

    if ratio_cv < 0.5:
        print(f"  [PASS] Scaling is consistent: CV={ratio_cv:.4f} < 0.5")
    else:
        print(f"  [KILL] Scaling inconsistent: CV={ratio_cv:.4f} >= 0.5")

    elapsed = time.time() - t0
    results['elapsed_s'] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
