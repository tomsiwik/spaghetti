"""L2 Norm Stability at Macro Scale — Qwen2.5-0.5B.

Checks if Qwen2.5-0.5B uses L2-normalized QK attention, composes 3 LoRA
adapters, runs 20+ seeds checking for catastrophic failures (>10% degradation).

Output: macro/l2_norm_macro/results.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
LORA_ALPHA = 16
FINETUNE_STEPS = 100
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
LR = 2e-4
N_SEEDS = 25
CATASTROPHIC_THRESHOLD = 0.10  # 10% degradation
OUT_DIR = Path(__file__).parent / "l2_norm_macro"


def create_domain_data(tokenizer, domain_id, n_samples=200, seed=42):
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


def check_l2_norm_attention(model_name):
    """Check if model uses L2-normalized QK attention."""
    config = AutoConfig.from_pretrained(model_name, cache_dir=HF_HOME,
                                         trust_remote_code=True)
    # Qwen2.5 uses RMSNorm on QK (not exactly L2 but similar)
    has_qk_norm = hasattr(config, 'use_qk_norm') and config.use_qk_norm
    # Also check for head_dim-based scaling
    head_dim = getattr(config, 'hidden_size', 896) // getattr(config, 'num_attention_heads', 14)

    print(f"  Config: hidden_size={getattr(config, 'hidden_size', '?')}")
    print(f"  Config: num_attention_heads={getattr(config, 'num_attention_heads', '?')}")
    print(f"  Config: head_dim={head_dim}")
    print(f"  Config: use_qk_norm={has_qk_norm}")

    # Check architecture source code
    return {
        'has_qk_norm': has_qk_norm,
        'head_dim': head_dim,
        'hidden_size': getattr(config, 'hidden_size', None),
        'num_heads': getattr(config, 'num_attention_heads', None),
        'config_keys': [k for k in config.to_dict().keys() if 'norm' in k.lower()],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"L2 Norm Stability — {MODEL_NAME}")
    print(f"Device: {DEVICE}, Seeds: {N_SEEDS}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check L2 norm
    print(f"\n{'='*60}")
    print("L2 NORM ATTENTION CHECK")
    print(f"{'='*60}")
    norm_info = check_l2_norm_attention(MODEL_NAME)

    # Create domain data
    domain_data = []
    for i in range(3):
        dd = create_domain_data(tokenizer, i, n_samples=200)
        domain_data.append(dd)
    val_data = []
    for dd in domain_data:
        val_data.extend(dd[:50])

    # Fine-tune 3 LoRA adapters once
    print(f"\n{'='*60}")
    print("FINE-TUNING 3 LoRA ADAPTERS")
    print(f"{'='*60}")

    lora_states = []
    for i in range(3):
        print(f"\n  Adapter {i}...")
        torch.manual_seed(42 + i + 1000)

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
        rng = random.Random(42 + i)
        model.train()
        for step in range(1, FINETUNE_STEPS + 1):
            input_ids, labels, attention_mask = get_batch(
                domain_data[i], BATCH_SIZE, rng)
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1].contiguous()
            targets = labels[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                  targets.view(-1), ignore_index=-100)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"    step {step} | loss {loss.item():.4f}")

        state = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        lora_states.append(state)
        del model, base_model, optimizer
        torch.cuda.empty_cache()

    # Compose: simple average
    merged_state = {}
    for key in lora_states[0]:
        merged_state[key] = sum(s[key] for s in lora_states) / 3

    # Get baseline loss (base model, no LoRA)
    print(f"\n{'='*60}")
    print("BASELINE")
    print(f"{'='*60}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True,
        torch_dtype=torch.float32
    ).to(DEVICE)
    base_loss = evaluate_model(base_model, val_data)
    print(f"  Base model loss: {base_loss:.4f}")
    del base_model
    torch.cuda.empty_cache()

    # Run N_SEEDS evaluations with different data orderings
    print(f"\n{'='*60}")
    print(f"STABILITY TEST: {N_SEEDS} SEEDS")
    print(f"{'='*60}")

    seed_losses = []
    catastrophic_failures = 0

    for seed_idx in range(N_SEEDS):
        torch.manual_seed(seed_idx)
        random.seed(seed_idx)

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
        model.load_state_dict(merged_state, strict=False)

        # Evaluate with different random batches
        rng = random.Random(seed_idx * 1000)
        total = 0.0
        for _ in range(20):
            input_ids, labels, attention_mask = get_batch(val_data, BATCH_SIZE, rng)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, :-1].contiguous()
                targets = labels[:, 1:].contiguous()
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                      targets.view(-1), ignore_index=-100)
                total += loss.item()
        avg_loss = total / 20
        degradation = (avg_loss - base_loss) / base_loss

        seed_losses.append({
            'seed': seed_idx,
            'loss': avg_loss,
            'degradation_pct': degradation * 100,
            'catastrophic': degradation > CATASTROPHIC_THRESHOLD,
        })

        if degradation > CATASTROPHIC_THRESHOLD:
            catastrophic_failures += 1
            print(f"  Seed {seed_idx}: loss={avg_loss:.4f} [CATASTROPHIC {degradation*100:+.1f}%]")
        elif seed_idx % 5 == 0:
            print(f"  Seed {seed_idx}: loss={avg_loss:.4f} ({degradation*100:+.1f}%)")

        del model, base_model
        torch.cuda.empty_cache()

    # Summary
    losses = [s['loss'] for s in seed_losses]
    degradations = [s['degradation_pct'] for s in seed_losses]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Base loss: {base_loss:.4f}")
    print(f"  Composed mean: {statistics.mean(losses):.4f} +/- {statistics.stdev(losses):.4f}")
    print(f"  Degradation: {statistics.mean(degradations):+.2f}% +/- {statistics.stdev(degradations):.2f}%")
    print(f"  Min/Max: {min(degradations):+.2f}% / {max(degradations):+.2f}%")
    print(f"  Catastrophic failures: {catastrophic_failures}/{N_SEEDS}")

    # Kill criteria
    print(f"\n{'='*60}")
    print("KILL CRITERIA")
    print(f"{'='*60}")

    if catastrophic_failures == 0:
        print(f"  [PASS] 0/{N_SEEDS} catastrophic failures (target: 0%)")
    else:
        print(f"  [KILL] {catastrophic_failures}/{N_SEEDS} catastrophic failures")

    results = {
        'model': MODEL_NAME,
        'lora_rank': LORA_RANK,
        'n_seeds': N_SEEDS,
        'base_loss': base_loss,
        'norm_info': norm_info,
        'seed_results': seed_losses,
        'mean_loss': statistics.mean(losses),
        'std_loss': statistics.stdev(losses),
        'mean_degradation_pct': statistics.mean(degradations),
        'std_degradation_pct': statistics.stdev(degradations),
        'catastrophic_failures': catastrophic_failures,
        'catastrophic_rate': catastrophic_failures / N_SEEDS,
        'elapsed_s': time.time() - t0,
    }

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
