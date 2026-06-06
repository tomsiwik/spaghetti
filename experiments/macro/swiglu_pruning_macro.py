"""SwiGLU Gate Pruning at Macro Scale — Qwen2.5-0.5B.

Tests whether SwiGLU gate-product signal (|gate * up|) identifies dead neurons
that can be pruned after LoRA composition without quality loss.

Previous macro attempt failed because low activation = specialist, not dead.
This time we use the gate-product signal, not activation magnitude.

Output: macro/swiglu_pruning_macro/results.json
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
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
FINETUNE_STEPS = 150
LR = 2e-4
PRUNE_THRESHOLDS = [0.05, 0.01, 0.001]
N_PROFILE_BATCHES = 50
OUT_DIR = Path(__file__).parent / "swiglu_pruning_macro"


def create_domain_data(tokenizer, domain_id, n_samples=300, seed=42):
    """Create simple domain data from wikitext or synthetic."""
    rng = random.Random(seed + domain_id)
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=HF_HOME)
        texts = [t for t in ds["text"] if len(t) > 100]
        rng.shuffle(texts)
        texts = texts[:n_samples]
    except Exception:
        words = ["the", "a", "is", "was", "in", "of", "to", "and", "for", "on",
                 "with", "at", "by", "from", "this", "that", "it", "as", "be", "or"]
        texts = []
        for _ in range(n_samples):
            length = rng.randint(50, 200)
            texts.append(" ".join(rng.choices(words, k=length)))

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


def compute_loss(model, input_ids, labels, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                          ignore_index=-100)


@torch.no_grad()
def evaluate_model(model, encodings, n_batches=20, batch_size=4):
    model.eval()
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        input_ids, labels, attention_mask = get_batch(encodings, batch_size, rng)
        loss = compute_loss(model, input_ids, labels, attention_mask)
        total += loss.item()
    model.train()
    return total / n_batches


def profile_gate_activations(model, encodings, n_batches=50):
    """Profile |gate_output * up_output| per neuron across all MLP layers.

    Qwen2.5 uses SwiGLU: output = down(silu(gate(x)) * up(x))
    We hook into the gate and up projections to measure the product.
    """
    model.eval()
    rng = random.Random(0)

    # Find MLP layers
    gate_activations = {}  # layer_idx -> accumulated |gate * up| per neuron

    hooks = []
    gate_outputs = {}
    up_outputs = {}

    def make_gate_hook(layer_idx):
        def hook(module, input, output):
            gate_outputs[layer_idx] = output.detach()
        return hook

    def make_up_hook(layer_idx):
        def hook(module, input, output):
            up_outputs[layer_idx] = output.detach()
        return hook

    # Register hooks on gate_proj and up_proj
    # After merge_and_unload(), model is plain Qwen2ForCausalLM
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.base_model.model.model.layers
    for idx, layer in enumerate(layers):
        mlp = layer.mlp
        hooks.append(mlp.gate_proj.register_forward_hook(make_gate_hook(idx)))
        hooks.append(mlp.up_proj.register_forward_hook(make_up_hook(idx)))
        gate_activations[idx] = None

    n_tokens_total = 0
    for _ in range(n_batches):
        input_ids, labels, attention_mask = get_batch(encodings, BATCH_SIZE, rng)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        n_tokens = attention_mask.sum().item()
        n_tokens_total += n_tokens

        for idx in gate_outputs:
            gate = gate_outputs[idx]
            up = up_outputs[idx]
            product = (F.silu(gate) * up).abs()
            # Average over batch and sequence dims, keep neuron dim
            per_neuron = product.sum(dim=(0, 1))  # [intermediate_size]
            if gate_activations[idx] is None:
                gate_activations[idx] = per_neuron
            else:
                gate_activations[idx] += per_neuron

    # Normalize by total tokens
    for idx in gate_activations:
        gate_activations[idx] = gate_activations[idx] / n_tokens_total

    # Remove hooks
    for h in hooks:
        h.remove()

    model.train()
    return gate_activations


def prune_neurons(model, gate_activations, threshold):
    """Zero out neurons where |gate*up| < threshold * max."""
    n_pruned_total = 0
    n_total = 0

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.base_model.model.model.layers
    for idx, layer in enumerate(layers):
        acts = gate_activations[idx]
        max_act = acts.max()
        mask = (acts >= threshold * max_act).float()
        n_active = mask.sum().item()
        n_neurons = mask.numel()
        n_pruned = n_neurons - n_active
        n_pruned_total += n_pruned
        n_total += n_neurons

        # Zero out pruned neurons in gate_proj, up_proj, down_proj
        with torch.no_grad():
            # Zero the output of gate and up for pruned neurons
            layer.mlp.gate_proj.weight.data *= mask.unsqueeze(1)
            layer.mlp.up_proj.weight.data *= mask.unsqueeze(1)
            # Zero the input of down_proj for pruned neurons
            layer.mlp.down_proj.weight.data *= mask.unsqueeze(0)

    pct = n_pruned_total / n_total * 100 if n_total > 0 else 0
    return n_pruned_total, n_total, pct


def random_prune_neurons(model, gate_activations, target_pct):
    """Random pruning baseline: prune the same % but random neurons."""
    n_pruned_total = 0
    n_total = 0
    rng = random.Random(42)

    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.base_model.model.model.layers
    for idx, layer in enumerate(layers):
        n_neurons = gate_activations[idx].numel()
        n_to_prune = int(n_neurons * target_pct / 100)
        indices = list(range(n_neurons))
        rng.shuffle(indices)
        prune_set = set(indices[:n_to_prune])

        mask = torch.ones(n_neurons, device=DEVICE)
        for p in prune_set:
            mask[p] = 0.0

        with torch.no_grad():
            layer.mlp.gate_proj.weight.data *= mask.unsqueeze(1)
            layer.mlp.up_proj.weight.data *= mask.unsqueeze(1)
            layer.mlp.down_proj.weight.data *= mask.unsqueeze(0)

        n_pruned_total += n_to_prune
        n_total += n_neurons

    return n_pruned_total, n_total


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"SwiGLU Gate Pruning Macro — {MODEL_NAME}")
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                               trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create 3 domain datasets
    domains = {}
    for i in range(3):
        domains[f"domain_{i}"] = create_domain_data(tokenizer, i, n_samples=300)
        print(f"  Domain {i}: {len(domains[f'domain_{i}'])} samples")

    # Combined val set
    all_val = []
    for d in domains.values():
        all_val.extend(d[:50])

    # Fine-tune 3 LoRA adapters
    lora_states = []
    for i in range(3):
        print(f"\n--- Fine-tuning LoRA adapter {i} ---")
        torch.manual_seed(42 + i)
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
                domains[f"domain_{i}"], BATCH_SIZE, rng)
            optimizer.zero_grad()
            loss = compute_loss(model, input_ids, labels, attention_mask)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"  step {step}/{FINETUNE_STEPS} | loss {loss.item():.4f}")

        state = {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}
        lora_states.append(state)
        del model, base_model, optimizer
        torch.cuda.empty_cache()

    # Compose: simple average of LoRA states
    print("\n--- Composing LoRA adapters (simple average) ---")
    merged_state = {}
    for key in lora_states[0]:
        merged_state[key] = sum(s[key] for s in lora_states) / len(lora_states)

    # Load composed model
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
    composed_model = get_peft_model(base_model, lora_config)
    composed_model.load_state_dict(merged_state, strict=False)

    # Merge LoRA into base weights
    composed_model = composed_model.merge_and_unload()

    # Baseline quality
    baseline_loss = evaluate_model(composed_model, all_val)
    print(f"\nBaseline (composed, no pruning): {baseline_loss:.4f}")

    # Profile gate activations
    print("\n--- Profiling gate activations ---")
    gate_activations = profile_gate_activations(composed_model, all_val, n_batches=N_PROFILE_BATCHES)

    # Print activation distribution
    all_acts = torch.cat([ga for ga in gate_activations.values()])
    print(f"  Total neurons: {all_acts.numel()}")
    print(f"  Min: {all_acts.min().item():.6f}")
    print(f"  Max: {all_acts.max().item():.6f}")
    print(f"  Mean: {all_acts.mean().item():.6f}")
    print(f"  Median: {all_acts.median().item():.6f}")

    for q in [0.01, 0.05, 0.1, 0.25]:
        val = torch.quantile(all_acts.float(), q).item()
        print(f"  {q*100:.0f}th percentile: {val:.6f}")

    # Test pruning at each threshold
    results = {
        'model': MODEL_NAME,
        'baseline_loss': baseline_loss,
        'n_adapters': 3,
        'pruning_results': [],
    }

    for tau in PRUNE_THRESHOLDS:
        print(f"\n--- Pruning at tau={tau} ---")

        # Gate-product pruning
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
        pruned_model = get_peft_model(base_model, lora_config)
        pruned_model.load_state_dict(merged_state, strict=False)
        pruned_model = pruned_model.merge_and_unload()

        n_pruned, n_total, pct = prune_neurons(pruned_model, gate_activations, tau)
        pruned_loss = evaluate_model(pruned_model, all_val)
        quality_delta = (pruned_loss - baseline_loss) / baseline_loss * 100

        print(f"  Gate-product: pruned {n_pruned}/{n_total} ({pct:.1f}%)")
        print(f"  Loss: {pruned_loss:.4f} (delta: {quality_delta:+.2f}%)")

        # Random pruning baseline (same %)
        del pruned_model
        torch.cuda.empty_cache()

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
        random_model = get_peft_model(base_model, lora_config)
        random_model.load_state_dict(merged_state, strict=False)
        random_model = random_model.merge_and_unload()

        random_prune_neurons(random_model, gate_activations, pct)
        random_loss = evaluate_model(random_model, all_val)
        random_delta = (random_loss - baseline_loss) / baseline_loss * 100

        print(f"  Random:       loss={random_loss:.4f} (delta: {random_delta:+.2f}%)")
        print(f"  Gate advantage: {random_delta - quality_delta:+.2f}pp")

        results['pruning_results'].append({
            'threshold': tau,
            'n_pruned': n_pruned,
            'n_total': n_total,
            'pct_pruned': pct,
            'gate_loss': pruned_loss,
            'gate_quality_delta_pct': quality_delta,
            'random_loss': random_loss,
            'random_quality_delta_pct': random_delta,
            'gate_advantage_pp': random_delta - quality_delta,
        })

        del random_model
        torch.cuda.empty_cache()

    # Kill criteria check
    print(f"\n{'='*70}")
    print("KILL CRITERIA")
    print(f"{'='*70}")

    best_prunable = 0
    best_quality = float('inf')
    for pr in results['pruning_results']:
        if pr['gate_quality_delta_pct'] < 3.0 and pr['pct_pruned'] > best_prunable:
            best_prunable = pr['pct_pruned']
            best_quality = pr['gate_quality_delta_pct']

    if best_prunable >= 10:
        print(f"  [PASS] {best_prunable:.1f}% prunable at {best_quality:+.2f}% quality (target: >10% at <3%)")
    else:
        print(f"  [KILL] Only {best_prunable:.1f}% prunable within 3% quality (target: >10%)")

    elapsed = time.time() - t0
    results['elapsed_s'] = elapsed
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
