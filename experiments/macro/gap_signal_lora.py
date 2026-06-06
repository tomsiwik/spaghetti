"""Gap-as-Signal Phase 2: Real LoRA on Qwen2.5-0.5B (5 domains).

Validates that the gap-calibration correlation holds with real LLM LoRA adapters.
Fine-tunes 5 rank-16 LoRA adapters on different domains, measures pairwise
orthogonality, gap magnitude, and calibration speed.

Benchmarks: vs joint training, vs simple average, vs TIES, vs DARE.

Output: macro/gap_signal_lora/results.json
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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
LORA_ALPHA = 16
N_SEEDS = 3  # fewer seeds since each is expensive
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
FINETUNE_STEPS = 200
CAL_STEPS = 100
LR = 2e-4
OUT_DIR = Path(__file__).parent / "gap_signal_lora"

# Domains: code-python, code-js, medical, legal, math
DOMAIN_CONFIGS = {
    "python": {"dataset": "bigcode/the-stack-smol", "name": "data/python", "field": "content", "split": "train"},
    "javascript": {"dataset": "bigcode/the-stack-smol", "name": "data/javascript", "field": "content", "split": "train"},
    "medical": {"dataset": "pubmed_qa", "name": "pqa_labeled", "field": "long_answer", "split": "train"},
    "legal": {"dataset": "joelito/eurlex", "name": None, "field": "text", "split": "train"},
    "math": {"dataset": "openai/gsm8k", "name": "main", "field": "question", "split": "train"},
}


def load_domain_data(domain_name, tokenizer, max_samples=500):
    """Load and tokenize domain data."""
    cfg = DOMAIN_CONFIGS[domain_name]
    try:
        if cfg["name"]:
            ds = load_dataset(cfg["dataset"], cfg["name"], split=cfg["split"],
                            cache_dir=HF_HOME, trust_remote_code=True)
        else:
            ds = load_dataset(cfg["dataset"], split=cfg["split"],
                            cache_dir=HF_HOME, trust_remote_code=True)
    except Exception as e:
        print(f"  Warning: Could not load {domain_name}: {e}")
        print(f"  Falling back to synthetic data for {domain_name}")
        return create_synthetic_domain(domain_name, tokenizer, max_samples)

    texts = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        text = item.get(cfg["field"], "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if text and len(text) > 50:
            texts.append(text[:2000])

    if len(texts) < 50:
        print(f"  Warning: only {len(texts)} samples for {domain_name}, augmenting")
        texts = texts * (50 // max(len(texts), 1) + 1)

    # Tokenize
    encodings = []
    for text in texts[:max_samples]:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True,
                              max_length=MAX_SEQ_LEN + 1)
        if len(ids) > 10:
            encodings.append(ids)

    return encodings


def create_synthetic_domain(domain_name, tokenizer, max_samples=500):
    """Create synthetic domain data as fallback."""
    templates = {
        "python": ["def {fn}({args}):\n    return {ret}\n" for _ in range(10)],
        "javascript": ["function {fn}({args}) {{ return {ret}; }}\n" for _ in range(10)],
        "medical": ["The patient presented with {symptom}. Treatment included {treatment}." for _ in range(10)],
        "legal": ["Article {n}. The parties agree to {clause}." for _ in range(10)],
        "math": ["If {x} + {y} = {z}, find the value of {x}." for _ in range(10)],
    }
    rng = random.Random(42)
    texts = []
    for _ in range(max_samples):
        t = rng.choice(templates.get(domain_name, templates["math"]))
        texts.append(t.format(fn="func", args="x, y", ret="x+y",
                             symptom="fever", treatment="rest",
                             n=rng.randint(1,100), clause="cooperation",
                             x=rng.randint(1,100), y=rng.randint(1,100),
                             z=rng.randint(1,200)))
    encodings = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True,
                              max_length=MAX_SEQ_LEN + 1)
        if len(ids) > 5:
            encodings.append(ids)
    return encodings


def get_batch(encodings, batch_size, rng, device=DEVICE):
    """Get a batch of (input_ids, labels) from encodings."""
    seqs = rng.choices(encodings, k=batch_size)
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        input_ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, :len(seq)] = torch.tensor(seq)
        labels[i, 0] = -100  # don't predict first token
        attention_mask[i, :len(seq)] = 1

    return input_ids, labels, attention_mask


def compute_loss(model, input_ids, labels, attention_mask):
    """Compute NTP loss."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                          ignore_index=-100)


@torch.no_grad()
def evaluate_domain(model, encodings, n_batches=10, batch_size=4):
    """Evaluate loss on domain data."""
    model.eval()
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        input_ids, labels, attention_mask = get_batch(encodings, batch_size, rng)
        loss = compute_loss(model, input_ids, labels, attention_mask)
        total += loss.item()
    model.train()
    return total / n_batches


def get_lora_delta(model):
    """Extract LoRA delta as a flat vector."""
    deltas = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            deltas.append(param.data.detach().reshape(-1))
    return torch.cat(deltas) if deltas else torch.tensor([0.0], device=DEVICE)


def get_lora_state(model):
    """Get LoRA adapter state dict."""
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def set_lora_state(model, state):
    """Set LoRA adapter weights."""
    model.load_state_dict(state, strict=False)


# ── Merging Methods ──────────────────────────────────────────────────────

def merge_simple_average(lora_states):
    """Simple averaging of LoRA states."""
    merged = {}
    for key in lora_states[0]:
        merged[key] = sum(s[key] for s in lora_states) / len(lora_states)
    return merged


def merge_ties(lora_states, threshold=0.2):
    """TIES merging: trim, elect sign, merge."""
    merged = {}
    for key in lora_states[0]:
        tensors = [s[key] for s in lora_states]
        # Trim: zero out small values
        trimmed = []
        for t in tensors:
            mask = t.abs() > threshold * t.abs().max()
            trimmed.append(t * mask.float())
        # Elect sign: majority vote
        signs = torch.stack([torch.sign(t) for t in trimmed])
        elected_sign = torch.sign(signs.sum(dim=0))
        # Disjoint merge
        result = torch.zeros_like(tensors[0])
        counts = torch.zeros_like(tensors[0])
        for t in trimmed:
            agree = (torch.sign(t) == elected_sign).float()
            result += t * agree
            counts += agree
        counts = counts.clamp(min=1)
        merged[key] = result / counts
    return merged


def merge_dare(lora_states, p=0.3):
    """DARE merging: random drop + rescale + average."""
    merged = {}
    for key in lora_states[0]:
        tensors = [s[key] for s in lora_states]
        dropped = []
        for t in tensors:
            mask = (torch.rand_like(t.float()) > p).float()
            dropped.append(t * mask / (1 - p))
        merged[key] = sum(dropped) / len(dropped)
    return merged


# ── Main Experiment ──────────────────────────────────────────────────────

def run_experiment(seed=42):
    print(f"\n{'='*70}")
    print(f"GAP-AS-SIGNAL PHASE 2: Real LoRA (seed={seed})")
    print(f"  Model: {MODEL_NAME}, rank={LORA_RANK}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    random.seed(seed)

    # Load tokenizer and base model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load domain data
    print("\nLoading domain data...")
    domain_data = {}
    domain_names = list(DOMAIN_CONFIGS.keys())
    for dname in domain_names:
        print(f"  Loading {dname}...")
        domain_data[dname] = load_domain_data(dname, tokenizer, max_samples=500)
        print(f"    {len(domain_data[dname])} samples")

    # Split each domain into train/val
    domain_train = {}
    domain_val = {}
    for dname, encs in domain_data.items():
        rng = random.Random(seed)
        rng.shuffle(encs)
        n_val = max(20, len(encs) // 5)
        domain_val[dname] = encs[:n_val]
        domain_train[dname] = encs[n_val:]

    # === 1. Fine-tune LoRA adapters ===
    lora_states = {}
    lora_deltas_flat = {}
    expert_val_losses = {}

    for e_idx, dname in enumerate(domain_names):
        print(f"\n--- Fine-tuning LoRA expert {e_idx}: {dname} ---")
        torch.manual_seed(seed + e_idx + 1000)

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
        model.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=LR
        )
        rng = random.Random(seed + e_idx)

        model.train()
        for step in range(1, FINETUNE_STEPS + 1):
            input_ids, labels, attention_mask = get_batch(
                domain_train[dname], BATCH_SIZE, rng)
            optimizer.zero_grad()
            loss = compute_loss(model, input_ids, labels, attention_mask)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"  step {step}/{FINETUNE_STEPS} | loss {loss.item():.4f}")

        # Evaluate
        val_loss = evaluate_domain(model, domain_val[dname])
        expert_val_losses[dname] = val_loss
        print(f"  Expert {dname} val loss: {val_loss:.4f}")

        # Save LoRA state and delta
        lora_states[dname] = get_lora_state(model)
        lora_deltas_flat[dname] = get_lora_delta(model)

        del model, base_model, optimizer
        torch.cuda.empty_cache()

    # === 2. Measure pairwise orthogonality ===
    print(f"\n{'='*70}")
    print("PAIRWISE ORTHOGONALITY")
    print(f"{'='*70}")

    pairwise_cos = {}
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            di, dj = domain_names[i], domain_names[j]
            flat_i = lora_deltas_flat[di]
            flat_j = lora_deltas_flat[dj]
            cos = F.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
            pair_key = f"{di}-{dj}"
            pairwise_cos[pair_key] = cos
            print(f"  cos({di}, {dj}) = {cos:.6f}")

    mean_cos = statistics.mean(pairwise_cos.values())
    print(f"\n  Mean cosine: {mean_cos:.6f}")

    # Theoretical prediction: r/sqrt(D)
    D_total = lora_deltas_flat[domain_names[0]].numel()
    expected_cos = LORA_RANK / math.sqrt(D_total)
    print(f"  Expected (r/sqrt(D)): {expected_cos:.6f}, D={D_total}")

    # === 3. For each pair: compose, measure gap, calibrate ===
    print(f"\n{'='*70}")
    print("GAP-CALIBRATION ANALYSIS")
    print(f"{'='*70}")

    pair_results = []
    for i in range(len(domain_names)):
        for j in range(i + 1, len(domain_names)):
            di, dj = domain_names[i], domain_names[j]
            cos_val = pairwise_cos[f"{di}-{dj}"]
            print(f"\n--- Pair: {di} + {dj} (cos={cos_val:.4f}) ---")

            # Simple average merge
            avg_state = merge_simple_average([lora_states[di], lora_states[dj]])

            # Load base + merged adapter
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
            merged_model = get_peft_model(base_model, lora_config)
            set_lora_state(merged_model, avg_state)

            # Measure gap: eval merged on both domains
            loss_i = evaluate_domain(merged_model, domain_val[di], n_batches=5)
            loss_j = evaluate_domain(merged_model, domain_val[dj], n_batches=5)

            # Compare with individual expert losses
            gap_i = loss_i - expert_val_losses[di]
            gap_j = loss_j - expert_val_losses[dj]
            avg_gap = (gap_i + gap_j) / 2

            print(f"  Merged loss: {di}={loss_i:.4f} (gap={gap_i:+.4f}), "
                  f"{dj}={loss_j:.4f} (gap={gap_j:+.4f})")
            print(f"  Average gap: {avg_gap:.4f}")

            pair_results.append({
                'pair': f"{di}-{dj}",
                'cosine': cos_val,
                'gap_i': gap_i,
                'gap_j': gap_j,
                'avg_gap': avg_gap,
                'merged_loss_i': loss_i,
                'merged_loss_j': loss_j,
            })

            del merged_model, base_model
            torch.cuda.empty_cache()

    # === 4. Correlation analysis ===
    print(f"\n{'='*70}")
    print("CORRELATION: COSINE vs GAP")
    print(f"{'='*70}")

    cosines = [r['cosine'] for r in pair_results]
    gaps = [r['avg_gap'] for r in pair_results]

    n = len(cosines)
    if n >= 3:
        mx_c = sum(cosines) / n
        my_g = sum(gaps) / n
        sxy = sum((x - mx_c) * (y - my_g) for x, y in zip(cosines, gaps))
        sxx = sum((x - mx_c) ** 2 for x in cosines)
        syy = sum((y - my_g) ** 2 for y in gaps)
        r2 = (sxy ** 2) / (sxx * syy) if sxx * syy > 0 else 0
        print(f"  r^2(cosine, gap) = {r2:.4f} (n={n} pairs)")
    else:
        r2 = 0.0
        print(f"  Not enough pairs for correlation (n={n})")

    # === 5. Merging comparison ===
    print(f"\n{'='*70}")
    print("MERGING METHOD COMPARISON")
    print(f"{'='*70}")

    all_states = [lora_states[d] for d in domain_names]
    methods = {
        'average': merge_simple_average(all_states),
        'ties': merge_ties(all_states),
        'dare': merge_dare(all_states),
    }

    method_results = {}
    for method_name, merged_state in methods.items():
        print(f"\n  --- {method_name} ---")
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
        merged_model = get_peft_model(base_model, lora_config)
        set_lora_state(merged_model, merged_state)

        domain_losses = {}
        for dname in domain_names:
            domain_losses[dname] = evaluate_domain(merged_model, domain_val[dname], n_batches=5)
        avg_loss = statistics.mean(domain_losses.values())
        avg_gap = avg_loss - statistics.mean(expert_val_losses.values())

        print(f"    Avg loss: {avg_loss:.4f}, avg gap: {avg_gap:+.4f}")
        method_results[method_name] = {
            'avg_loss': avg_loss,
            'avg_gap': avg_gap,
            'domain_losses': domain_losses,
        }

        del merged_model, base_model
        torch.cuda.empty_cache()

    # === 6. Compile results ===
    results = {
        'seed': seed,
        'model': MODEL_NAME,
        'lora_rank': LORA_RANK,
        'domains': domain_names,
        'expert_val_losses': expert_val_losses,
        'pairwise_cosines': pairwise_cos,
        'mean_cosine': mean_cos,
        'expected_cosine': expected_cos,
        'D_total': D_total,
        'pair_results': pair_results,
        'r2_cosine_gap': r2,
        'method_comparison': method_results,
    }

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_results = []
    for seed in range(42, 42 + N_SEEDS):
        result = run_experiment(seed=seed)
        all_results.append(result)

        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    all_cosines = []
    all_gaps = []
    for res in all_results:
        for pr in res['pair_results']:
            all_cosines.append(pr['cosine'])
            all_gaps.append(pr['avg_gap'])

    if len(all_cosines) >= 3:
        n = len(all_cosines)
        mx = sum(all_cosines) / n
        my = sum(all_gaps) / n
        sxy = sum((x-mx)*(y-my) for x,y in zip(all_cosines, all_gaps))
        sxx = sum((x-mx)**2 for x in all_cosines)
        syy = sum((y-my)**2 for y in all_gaps)
        r2 = (sxy**2)/(sxx*syy) if sxx*syy > 0 else 0
        print(f"\n  Overall r^2(cosine, gap): {r2:.4f}")

    for res in all_results:
        print(f"\n  Seed {res['seed']}:")
        print(f"    Mean cosine: {res['mean_cosine']:.6f}")
        print(f"    r^2: {res['r2_cosine_gap']:.4f}")
        for method, mr in res['method_comparison'].items():
            print(f"    {method}: avg_loss={mr['avg_loss']:.4f}, gap={mr['avg_gap']:+.4f}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Save final
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
