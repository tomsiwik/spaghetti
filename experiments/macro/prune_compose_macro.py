"""Pre-Composition Pruning Pipeline at Macro — Qwen2.5-0.5B.

Compares two pipelines:
  A: compose -> prune -> calibrate (baseline)
  B: prune each independently -> compose -> calibrate (protocol)

Must prove: Pipeline B within 2% of Pipeline A (micro was +0.01%).

Output: macro/prune_compose_macro/results.json
"""

import json
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
LORA_RANK = 16
LORA_ALPHA = 16
FINETUNE_STEPS = 150
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
LR = 2e-4
PRUNE_FRAC = 0.2  # prune 20% of LoRA parameters
N_SEEDS = 3
OUT_DIR = Path(__file__).parent / "prune_compose_macro"


def create_domain_data(tokenizer, domain_id, n_samples=300, seed=42):
    rng = random.Random(seed + domain_id * 7)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=HF_HOME)
        texts = [t for t in ds["text"] if len(t) > 100]
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


@torch.no_grad()
def evaluate_model(model, encodings, n_batches=20, batch_size=4):
    model.eval()
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        input_ids, labels, attention_mask = get_batch(encodings, batch_size, rng)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1].contiguous()
        targets = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                              targets.view(-1), ignore_index=-100)
        total += loss.item()
    model.train()
    return total / n_batches


def magnitude_prune_lora_state(state, frac):
    """Prune smallest-magnitude LoRA parameters."""
    pruned = {}
    for key, tensor in state.items():
        flat = tensor.abs().reshape(-1)
        n_prune = int(flat.numel() * frac)
        if n_prune > 0:
            threshold = torch.topk(flat, n_prune, largest=False).values[-1]
            mask = (tensor.abs() > threshold).float()
            pruned[key] = tensor * mask
        else:
            pruned[key] = tensor.clone()
    return pruned


def run_seed(seed, tokenizer, domain_data_all, val_data):
    print(f"\n{'='*60}")
    print(f"SEED = {seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    random.seed(seed)

    # Fine-tune 3 LoRA adapters
    lora_states = []
    for i in range(3):
        print(f"\n  Fine-tuning adapter {i}...")
        torch.manual_seed(seed + i + 1000)

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(DEVICE)
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "up_proj", "gate_proj", "down_proj"],
            lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
        rng = random.Random(seed + i)
        model.train()
        for step in range(1, FINETUNE_STEPS + 1):
            input_ids, labels, attention_mask = get_batch(
                domain_data_all[i], BATCH_SIZE, rng)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()

        state = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        lora_states.append(state)
        del model, base_model, optimizer
        torch.cuda.empty_cache()

    # Pipeline A: compose -> prune -> calibrate
    print("\n  Pipeline A: compose -> prune")
    composed_state = {}
    for key in lora_states[0]:
        composed_state[key] = sum(s[key] for s in lora_states) / 3

    pruned_composed = magnitude_prune_lora_state(composed_state, PRUNE_FRAC)

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model_a = get_peft_model(base_model, lora_config)
    model_a.load_state_dict(pruned_composed, strict=False)
    loss_a = evaluate_model(model_a, val_data)
    print(f"    Pipeline A loss: {loss_a:.4f}")
    del model_a, base_model
    torch.cuda.empty_cache()

    # Pipeline B: prune each -> compose
    print("  Pipeline B: prune each -> compose")
    pruned_individuals = [magnitude_prune_lora_state(s, PRUNE_FRAC) for s in lora_states]
    composed_pruned = {}
    for key in pruned_individuals[0]:
        composed_pruned[key] = sum(s[key] for s in pruned_individuals) / 3

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model_b = get_peft_model(base_model, lora_config)
    model_b.load_state_dict(composed_pruned, strict=False)
    loss_b = evaluate_model(model_b, val_data)
    print(f"    Pipeline B loss: {loss_b:.4f}")
    del model_b, base_model
    torch.cuda.empty_cache()

    # No pruning baseline
    print("  No pruning baseline:")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "up_proj", "gate_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model_np = get_peft_model(base_model, lora_config)
    model_np.load_state_dict(composed_state, strict=False)
    loss_np = evaluate_model(model_np, val_data)
    print(f"    No-prune loss: {loss_np:.4f}")
    del model_np, base_model
    torch.cuda.empty_cache()

    # Compute gaps
    gap_ab = (loss_b - loss_a) / loss_a * 100
    gap_a_np = (loss_a - loss_np) / loss_np * 100
    gap_b_np = (loss_b - loss_np) / loss_np * 100

    print(f"\n  B vs A: {gap_ab:+.2f}%")
    print(f"  A vs no-prune: {gap_a_np:+.2f}%")
    print(f"  B vs no-prune: {gap_b_np:+.2f}%")

    return {
        'seed': seed,
        'loss_a': loss_a,
        'loss_b': loss_b,
        'loss_no_prune': loss_np,
        'gap_b_vs_a_pct': gap_ab,
        'gap_a_vs_noprune_pct': gap_a_np,
        'gap_b_vs_noprune_pct': gap_b_np,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Prune-Compose Pipeline — {MODEL_NAME}")
    print(f"Prune fraction: {PRUNE_FRAC*100:.0f}%, Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create domain data
    domain_data_all = []
    for i in range(3):
        dd = create_domain_data(tokenizer, i, n_samples=300)
        domain_data_all.append(dd)
        print(f"  Domain {i}: {len(dd)} samples")

    # Combined val
    val_data = []
    for dd in domain_data_all:
        val_data.extend(dd[:50])

    all_results = []
    for seed in range(42, 42 + N_SEEDS):
        result = run_seed(seed, tokenizer, domain_data_all, val_data)
        all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    gaps = [r['gap_b_vs_a_pct'] for r in all_results]
    mean_gap = statistics.mean(gaps)
    std_gap = statistics.stdev(gaps) if len(gaps) > 1 else 0

    print(f"  Mean B-vs-A gap: {mean_gap:+.2f}% +/- {std_gap:.2f}%")

    # Kill criteria
    if abs(mean_gap) < 2.0:
        print(f"  [PASS] Pipeline B within 2% of A: {mean_gap:+.2f}%")
    else:
        print(f"  [KILL] Pipeline B NOT within 2%: {mean_gap:+.2f}%")

    results = {
        'model': MODEL_NAME,
        'prune_frac': PRUNE_FRAC,
        'n_seeds': N_SEEDS,
        'seed_results': all_results,
        'mean_gap_b_vs_a': mean_gap,
        'std_gap_b_vs_a': std_gap,
        'elapsed_s': time.time() - t0,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
