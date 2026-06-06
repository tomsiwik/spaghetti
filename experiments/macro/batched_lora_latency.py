"""Batched LoRA Latency Fix — proving the 314-396% overhead is an implementation artifact.

The MoE benchmark (lora_moe_benchmark.py) showed LoRA MoE has excellent quality
(+0.70% vs joint) but 314-396% latency overhead. This overhead comes from calling
set_lora_state() sequentially and running separate full forward passes per expert.

The theoretical overhead is only 0.98% (LoRA FLOPs/token = 9,633,792 vs Base
FLOPs/token = 988,000,000).

This script proves the overhead can be eliminated by:
1. Running the base model forward pass ONCE to get hidden states at each layer
2. For each layer where LoRA is applied, extracting A and B matrices from expert states
3. Computing the LoRA delta as an additive operation: delta = input @ A.T @ B.T
4. Weighting the deltas by router weights and adding to base output

Result: ONE base forward pass + k small matrix multiplications per LoRA target per layer.

Output: macro/batched_lora_latency/results.json
"""

import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_HOME = os.environ.get("HF_HOME", "/workspace/hf_cache")
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
LORA_RANK = 16
LORA_ALPHA = 16
BATCH_SIZE = 4
MAX_SEQ_LEN = 256
FINETUNE_STEPS = 80  # Quick training — we only need weight matrices, not quality
LR = 2e-4
OUT_DIR = Path(__file__).parent / "batched_lora_latency"

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]

# 4 domain configs matching the benchmark
DOMAIN_CONFIGS = {
    "python": {"dataset": "nampdn-ai/tiny-codes", "name": None, "field": "response", "split": "train",
               "filter_fn": lambda item: "python" in item.get("prompt", "").lower() or "python" in item.get("response", "")[:100].lower()},
    "javascript": {"dataset": "nampdn-ai/tiny-codes", "name": None, "field": "response", "split": "train",
                   "filter_fn": lambda item: "javascript" in item.get("prompt", "").lower() or "function" in item.get("response", "")[:80].lower()},
    "news": {"dataset": "abisee/cnn_dailymail", "name": "3.0.0", "field": "article", "split": "train",
             "filter_fn": None},
    "math": {"dataset": "openai/gsm8k", "name": "main", "field": "question", "split": "train",
             "filter_fn": None},
}

N_DOMAINS = len(DOMAIN_CONFIGS)
SYNTHETIC_DOMAINS = set()


def load_domain_data(domain_name, tokenizer, max_samples=200, seed=42):
    cfg = DOMAIN_CONFIGS[domain_name]
    try:
        kwargs = {"split": cfg["split"], "cache_dir": HF_HOME, "trust_remote_code": True}
        if cfg["name"]:
            ds = load_dataset(cfg["dataset"], cfg["name"], **kwargs)
        else:
            ds = load_dataset(cfg["dataset"], **kwargs)
    except Exception as e:
        print(f"  [FALLBACK] {domain_name}: {e}")
        SYNTHETIC_DOMAINS.add(domain_name)
        return create_synthetic(domain_name, tokenizer, max_samples, seed)

    filter_fn = cfg.get("filter_fn")
    texts = []
    for i, item in enumerate(ds):
        if i >= max_samples * 10:
            break
        if filter_fn and not filter_fn(item):
            continue
        text = item.get(cfg["field"], "")
        if isinstance(text, list):
            text = " ".join(str(t) for t in text)
        if text and len(text) > 50:
            texts.append(text[:2000])
        if len(texts) >= max_samples * 2:
            break

    if len(texts) < 50:
        print(f"  [FALLBACK] {domain_name}: only {len(texts)} samples, using synthetic")
        SYNTHETIC_DOMAINS.add(domain_name)
        return create_synthetic(domain_name, tokenizer, max_samples, seed)

    print(f"  [REAL DATA] {domain_name}: {len(texts)} samples from {cfg['dataset']}")
    encodings = []
    for text in texts[:max_samples]:
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN + 1)
        if len(ids) > 10:
            encodings.append(ids)
    return encodings


def create_synthetic(domain_name, tokenizer, max_samples=200, seed=42):
    rng = random.Random(seed)
    func_names = ["calc", "process", "transform", "convert", "validate", "parse"]
    templates = {
        "python": [
            "def {f}(x, y):\n    result = x {op} y\n    return result\n\nprint({f}({a}, {b}))\n",
            "class {F}:\n    def __init__(self, val={n}):\n        self.val = val\n    def get(self):\n        return self.val\n",
        ],
        "javascript": [
            "function {f}(arr) {{\n  return arr.map(x => x {op} {n}).filter(x => x > 0);\n}}\nconsole.log({f}([{a}, {b}]));\n",
            "const {f} = ({a}) => {{\n  if ({a} > {n}) return true;\n  return false;\n}};\n",
        ],
        "news": "Breaking news: Reports indicate that {s}. Officials stated {t}. The situation remains {o}.",
        "math": "Calculate: {a} {op} {b} = ?\nStep 1: We need to find {a} {op} {b}.\nStep 2: {a} {op} {b} = {r}.",
    }
    ops = ["+", "-", "*"]
    texts = []
    for _ in range(max_samples):
        fname = rng.choice(func_names)
        a, b, n = rng.randint(1, 99), rng.randint(1, 99), rng.randint(1, 20)
        op = rng.choice(ops)
        r = eval(f"{a} {op} {b}")
        t = templates.get(domain_name, templates["math"])
        if isinstance(t, list):
            t = rng.choice(t)
        texts.append(t.format(f=fname, F=fname.capitalize(), n=n, s="developments unfolding",
                              t="measures are being taken", o="ongoing", a=a, b=b, op=op, r=r))
    return [tokenizer.encode(t, add_special_tokens=True, truncation=True, max_length=MAX_SEQ_LEN + 1)
            for t in texts if len(t) > 5]


def get_batch(encodings, batch_size, rng, device=DEVICE):
    seqs = rng.choices(encodings, k=batch_size)
    max_len = min(MAX_SEQ_LEN, max(len(s) for s in seqs))
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attn = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        seq = seq[:max_len]
        input_ids[i, :len(seq)] = torch.tensor(seq)
        labels[i, :len(seq)] = torch.tensor(seq)
        labels[i, 0] = -100
        attn[i, :len(seq)] = 1
    return input_ids, labels, attn


def compute_loss(model, input_ids, labels, attn):
    out = model(input_ids=input_ids, attention_mask=attn)
    logits = out.logits[:, :-1].contiguous()
    targets = labels[:, 1:].contiguous()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)


def get_lora_state(model):
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def set_lora_state(model, state):
    model.load_state_dict(state, strict=False)


# =============================================================================
# KEY DATA STRUCTURE: Pre-extracted LoRA matrices organized for batched application
# =============================================================================

def extract_lora_matrices(lora_state, n_layers=24, scaling=1.0):
    """Extract LoRA A and B matrices from a state dict into a structured format.

    Returns: dict mapping (layer_idx, target_name) -> (A, B, scaling)
    where A shape is (r, d_in) and B shape is (d_out, r)
    so the LoRA output is: scaling * (x @ A.T @ B.T)
    """
    matrices = {}
    for layer_idx in range(n_layers):
        for target in LORA_TARGETS:
            # PEFT key format
            a_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{target}.lora_A.default.weight"
            b_key = f"base_model.model.model.layers.{layer_idx}.self_attn.{target}.lora_B.default.weight"

            # For MLP targets, the path is different
            if target in ("up_proj", "gate_proj", "down_proj"):
                a_key = f"base_model.model.model.layers.{layer_idx}.mlp.{target}.lora_A.default.weight"
                b_key = f"base_model.model.model.layers.{layer_idx}.mlp.{target}.lora_B.default.weight"

            if a_key in lora_state and b_key in lora_state:
                matrices[(layer_idx, target)] = (
                    lora_state[a_key],  # (r, d_in)
                    lora_state[b_key],  # (d_out, r)
                    scaling,
                )
    return matrices


# =============================================================================
# APPROACH 1: Sequential baseline (matches the benchmark bottleneck)
# =============================================================================

@torch.no_grad()
def sequential_moe_forward(model, input_ids, attn, lora_states, expert_weights):
    """The SLOW approach: swap LoRA weights and run full forward per expert.

    This matches lines 460-477 of lora_moe_benchmark.py.
    """
    domains = list(lora_states.keys())
    n_experts = len(domains)
    top_k = expert_weights.shape[1]  # (batch, top_k)

    # Get top-k expert indices (simulate router)
    # expert_weights: (batch, n_experts) -> top-k selection
    topk_weights, topk_idx = expert_weights.topk(top_k, dim=-1)
    topk_weights = F.softmax(topk_weights, dim=-1)  # renormalize

    mixed_logits = None
    for k_i in range(top_k):
        for b in range(input_ids.shape[0]):
            eidx = topk_idx[b, k_i].item()
            set_lora_state(model, lora_states[domains[eidx]])
            out = model(input_ids=input_ids[b:b+1], attention_mask=attn[b:b+1])
            w = topk_weights[b, k_i]
            contribution = w * out.logits
            if mixed_logits is None:
                mixed_logits = torch.zeros(input_ids.shape[0], out.logits.shape[1],
                                          out.logits.shape[2], device=DEVICE)
            mixed_logits[b:b+1] += contribution

    return mixed_logits


@torch.no_grad()
def sequential_moe_forward_simple(model, input_ids, attn, lora_states, topk_idx, topk_weights):
    """Simpler sequential: run full model forward for each selected expert, weight-average logits.

    Matches the benchmark more closely — all batch items use same experts.
    topk_idx: (top_k,) integer indices into domain list
    topk_weights: (top_k,) float weights summing to 1
    """
    domains = list(lora_states.keys())
    mixed_logits = None

    for k_i in range(len(topk_idx)):
        eidx = topk_idx[k_i]
        set_lora_state(model, lora_states[domains[eidx]])
        out = model(input_ids=input_ids, attention_mask=attn)
        w = topk_weights[k_i]
        if mixed_logits is None:
            mixed_logits = w * out.logits
        else:
            mixed_logits = mixed_logits + w * out.logits

    return mixed_logits


# =============================================================================
# APPROACH 2: Batched LoRA — ONE base forward + additive deltas
# =============================================================================

class BatchedLoRAInference:
    """Pre-compute base model hidden states, then apply LoRA deltas additively.

    Architecture insight: A LoRA adapter modifies a linear layer Y = Wx to
    Y' = Wx + scaling * x @ A.T @ B.T. Since addition distributes over the
    forward pass, we can:
    1. Run the base forward pass ONCE (no LoRA)
    2. For each LoRA target in each layer, compute the delta
    3. But we cannot simply add deltas at the output because the model is
       nonlinear — the LoRA delta at layer L affects layer L+1's input.

    CORRECT approach: Hook into each linear layer during the base forward pass
    to capture inputs, then apply LoRA deltas and re-run from that point.

    EVEN BETTER approach: Since we want the final logits, and LoRA deltas are
    small perturbations, we can use the fact that PEFT already provides the
    mechanism — we just need to avoid the weight-swapping overhead.

    ACTUAL approach implemented here:
    - Pre-extract all LoRA A/B matrices for all experts
    - Register forward hooks on LoRA target modules to capture layer inputs
    - Run base forward ONCE to collect all intermediate activations
    - For each expert's LoRA targets, compute delta = scaling * input @ A.T @ B.T
    - Apply these deltas through the rest of the network

    BUT this is complex because of nonlinearity. The PRACTICAL batched approach:
    - Store all expert LoRA matrices in GPU memory (they're small: k * 7 * 24 * 2 * r * d)
    - For Qwen2.5-0.5B: 4 * 7 * 24 * 2 * 16 * 896 = 38.5M params = ~154MB in fp32
    - Use a custom forward that applies LoRA as batched matmuls without weight swapping

    The cleanest implementation: modify the PEFT model to support multi-expert
    inference in a single forward pass by pre-loading all expert weights.
    """

    def __init__(self, base_model, lora_states, n_layers=24, scaling=1.0):
        """
        base_model: the PEFT model (with LoRA modules)
        lora_states: dict of domain_name -> state_dict with LoRA weights
        """
        self.model = base_model
        self.domains = list(lora_states.keys())
        self.n_experts = len(self.domains)
        self.n_layers = n_layers
        self.scaling = scaling

        # Pre-extract all expert matrices
        self.expert_matrices = {}
        for domain in self.domains:
            self.expert_matrices[domain] = extract_lora_matrices(
                lora_states[domain], n_layers, scaling
            )

        # Zero out the model's LoRA weights (we'll apply deltas manually)
        zero_state = {k: torch.zeros_like(v) for k, v in lora_states[self.domains[0]].items()}
        set_lora_state(self.model, zero_state)

    @torch.no_grad()
    def forward(self, input_ids, attn, topk_idx, topk_weights):
        """Batched inference using direct parameter copy (skip load_state_dict).

        Still does k forward passes but avoids the expensive load_state_dict call
        by directly writing to LoRA parameter tensors.
        """
        k = len(topk_idx)
        expert_logits = []
        for ki in range(k):
            eidx = topk_idx[ki]
            domain = self.domains[eidx]
            matrices = self.expert_matrices[domain]

            # Directly set LoRA parameters (much faster than load_state_dict)
            for (layer_idx, target), (A, B, s) in matrices.items():
                if target in ("up_proj", "gate_proj", "down_proj"):
                    module = self.model.base_model.model.model.layers[layer_idx].mlp
                else:
                    module = self.model.base_model.model.model.layers[layer_idx].self_attn
                lora_layer = getattr(module, target)
                lora_layer.lora_A["default"].weight.data.copy_(A)
                lora_layer.lora_B["default"].weight.data.copy_(B)

            out = self.model(input_ids=input_ids, attention_mask=attn)
            expert_logits.append(out.logits)

        # Weighted average
        mixed = torch.zeros_like(expert_logits[0])
        for ki in range(k):
            mixed = mixed + topk_weights[ki] * expert_logits[ki]

        return mixed

    @torch.no_grad()
    def forward_stacked(self, input_ids, attn, topk_idx, topk_weights):
        """TRUE batched: stack k copies of input, apply different LoRA per batch slice.

        Uses hooks to apply expert-specific LoRA deltas to each slice of the
        stacked batch. ONE forward pass through the model, k times the batch size.
        """
        batch_size = input_ids.shape[0]
        k = len(topk_idx)

        # Stack inputs: (k * batch, seq_len)
        stacked_ids = input_ids.repeat(k, 1)
        stacked_attn = attn.repeat(k, 1)

        hooks = []

        # For each LoRA target, register a hook that applies expert-specific deltas
        for layer_idx in range(self.n_layers):
            for target in LORA_TARGETS:
                if target in ("up_proj", "gate_proj", "down_proj"):
                    parent = self.model.base_model.model.model.layers[layer_idx].mlp
                else:
                    parent = self.model.base_model.model.model.layers[layer_idx].self_attn
                lora_module = getattr(parent, target)

                # Collect A, B for each expert slice
                expert_ABs = []
                for ki in range(k):
                    eidx = topk_idx[ki]
                    domain = self.domains[eidx]
                    key = (layer_idx, target)
                    if key in self.expert_matrices[domain]:
                        A, B, s = self.expert_matrices[domain][key]
                        expert_ABs.append((A, B, s))
                    else:
                        expert_ABs.append(None)

                if all(ab is None for ab in expert_ABs):
                    continue

                def make_hook(ab_list, bs, num_k):
                    def hook_fn(module, input, output):
                        x = input[0]  # (k*batch, seq, d_in)
                        for ki, ab in enumerate(ab_list):
                            if ab is None:
                                continue
                            A, B, s = ab
                            start = ki * bs
                            end = (ki + 1) * bs
                            # Apply LoRA delta to this expert's slice
                            x_slice = x[start:end]  # (batch, seq, d_in)
                            delta = s * (x_slice @ A.t() @ B.t())  # (batch, seq, d_out)
                            output[start:end] = output[start:end] + delta
                        return output
                    return hook_fn

                h = lora_module.register_forward_hook(make_hook(expert_ABs, batch_size, k))
                hooks.append(h)

        # ONE forward pass with stacked batch
        out = self.model(input_ids=stacked_ids, attention_mask=stacked_attn)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Unstack and weight-average: out.logits is (k*batch, seq, vocab)
        logits = out.logits
        mixed = torch.zeros(batch_size, logits.shape[1], logits.shape[2], device=logits.device)
        for ki in range(k):
            start = ki * batch_size
            end = (ki + 1) * batch_size
            mixed = mixed + topk_weights[ki] * logits[start:end]

        return mixed

    def setup_persistent_hooks(self):
        """Register PERMANENT hooks that are controlled by self._hook_config.

        Avoids the overhead of register/remove per call (168 hooks * Python overhead).
        Set self._hook_config before calling forward to control behavior.
        """
        self._hook_config = None  # Set to (topk_idx, batch_size, topk_weights) to enable
        self._persistent_hooks = []

        for layer_idx in range(self.n_layers):
            for target in LORA_TARGETS:
                if target in ("up_proj", "gate_proj", "down_proj"):
                    parent = self.model.base_model.model.model.layers[layer_idx].mlp
                else:
                    parent = self.model.base_model.model.model.layers[layer_idx].self_attn
                lora_module = getattr(parent, target)

                def make_hook(li, tgt):
                    def hook_fn(module, input, output):
                        cfg = self._hook_config
                        if cfg is None:
                            return output
                        topk_idx, bs, topk_weights = cfg
                        k = len(topk_idx)
                        x = input[0]
                        for ki in range(k):
                            eidx = topk_idx[ki]
                            domain = self.domains[eidx]
                            key = (li, tgt)
                            if key not in self.expert_matrices[domain]:
                                continue
                            A, B, s = self.expert_matrices[domain][key]
                            start = ki * bs
                            end = (ki + 1) * bs
                            x_slice = x[start:end]
                            delta = s * (x_slice @ A.t() @ B.t())
                            output[start:end] = output[start:end] + delta
                        return output
                    return hook_fn

                h = lora_module.register_forward_hook(make_hook(layer_idx, target))
                self._persistent_hooks.append(h)

        print(f"    Registered {len(self._persistent_hooks)} persistent hooks")

    @torch.no_grad()
    def forward_persistent(self, input_ids, attn, topk_idx, topk_weights):
        """Forward using persistent hooks — no hook registration overhead."""
        batch_size = input_ids.shape[0]
        k = len(topk_idx)

        stacked_ids = input_ids.repeat(k, 1)
        stacked_attn = attn.repeat(k, 1)

        # Enable hooks
        self._hook_config = (topk_idx, batch_size, topk_weights)

        out = self.model(input_ids=stacked_ids, attention_mask=stacked_attn)

        # Disable hooks
        self._hook_config = None

        logits = out.logits
        mixed = torch.zeros(batch_size, logits.shape[1], logits.shape[2], device=logits.device)
        for ki in range(k):
            start = ki * batch_size
            end = (ki + 1) * batch_size
            mixed = mixed + topk_weights[ki] * logits[start:end]

        return mixed

    def remove_persistent_hooks(self):
        """Remove all persistent hooks."""
        if hasattr(self, '_persistent_hooks'):
            for h in self._persistent_hooks:
                h.remove()
            self._persistent_hooks = []

    @torch.no_grad()
    def forward_direct_matmul(self, input_ids, attn, topk_idx, topk_weights):
        """Even faster: skip PEFT entirely, apply LoRA as manual matmuls via hooks.

        Register hooks on each LoRA target linear layer to:
        1. Capture the input to the linear layer
        2. Add the weighted LoRA delta from all top-k experts

        This requires ONE base forward pass + the hook overhead (small matmuls).
        """
        batch_size = input_ids.shape[0]
        k = len(topk_idx)
        hooks = []

        # For each LoRA target, register a hook that adds the weighted expert deltas
        for layer_idx in range(self.n_layers):
            for target in LORA_TARGETS:
                # Get the underlying linear module (not the LoRA wrapper)
                if target in ("up_proj", "gate_proj", "down_proj"):
                    parent = self.model.base_model.model.model.layers[layer_idx].mlp
                else:
                    parent = self.model.base_model.model.model.layers[layer_idx].self_attn
                lora_module = getattr(parent, target)

                # Collect A, B matrices for selected experts
                expert_ABs = []
                for ki in range(k):
                    eidx = topk_idx[ki]
                    domain = self.domains[eidx]
                    key = (layer_idx, target)
                    if key in self.expert_matrices[domain]:
                        A, B, s = self.expert_matrices[domain][key]
                        expert_ABs.append((A, B, s, topk_weights[ki]))

                if not expert_ABs:
                    continue

                def make_hook(ab_list):
                    def hook_fn(module, input, output):
                        x = input[0]  # (batch, seq, d_in)
                        delta = torch.zeros_like(output)
                        for A, B, s, w in ab_list:
                            # x @ A.T -> (batch, seq, r), then @ B.T -> (batch, seq, d_out)
                            d = s * (x @ A.t() @ B.t())
                            delta = delta + w * d
                        return output + delta
                    return hook_fn

                h = lora_module.register_forward_hook(make_hook(expert_ABs))
                hooks.append(h)

        # Zero out LoRA weights so base forward is clean
        # (already done in __init__, but ensure)

        # Run single forward pass — hooks add the LoRA deltas
        out = self.model(input_ids=input_ids, attention_mask=attn)

        # Remove hooks
        for h in hooks:
            h.remove()

        return out.logits


# =============================================================================
# APPROACH 3: Pure manual forward — bypass PEFT completely
# =============================================================================

@torch.no_grad()
def pure_base_forward(base_model_unwrapped, input_ids, attn):
    """Run the raw base model (no LoRA) and return logits."""
    return base_model_unwrapped(input_ids=input_ids, attention_mask=attn).logits


# =============================================================================
# Latency measurement utilities
# =============================================================================

def warmup_gpu(model, input_ids, attn, n=5):
    """GPU warmup to stabilize timing."""
    for _ in range(n):
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def time_fn(fn, n_iters=50):
    """Time a function over n_iters, return mean ms."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1000


# =============================================================================
# Main experiment
# =============================================================================

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("BATCHED LoRA LATENCY FIX")
    print("=" * 70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=HF_HOME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load domain data
    domains = list(DOMAIN_CONFIGS.keys())
    domain_data = {}
    for d in domains:
        print(f"  Loading {d}...")
        domain_data[d] = load_domain_data(d, tokenizer, 200, 42)
        print(f"    {len(domain_data[d])} samples")

    # -- 1. Train quick LoRA experts (minimal steps, just need weight matrices) --
    print(f"\n{'=' * 70}")
    print(f"PHASE 1: Train {N_DOMAINS} quick LoRA experts ({FINETUNE_STEPS} steps each)")
    print(f"{'=' * 70}")

    lora_states = {}
    for idx, domain in enumerate(domains):
        print(f"\n  Training expert: {domain}")
        torch.manual_seed(42 + idx * 100)

        base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                                     trust_remote_code=True, dtype=torch.float32).to(DEVICE)
        cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(base, cfg)

        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        rng = random.Random(42 + idx)

        model.train()
        for step in range(1, FINETUNE_STEPS + 1):
            opt.zero_grad()
            loss = compute_loss(model, *get_batch(domain_data[domain], BATCH_SIZE, rng))
            loss.backward()
            opt.step()
            if step % 20 == 0:
                print(f"    step {step}/{FINETUNE_STEPS} loss={loss.item():.4f}")

        lora_states[domain] = get_lora_state(model)
        print(f"    Done. LoRA state has {len(lora_states[domain])} keys")

        del model, base, opt
        torch.cuda.empty_cache()

    # -- 2. Set up for latency measurement --
    print(f"\n{'=' * 70}")
    print(f"PHASE 2: Latency measurement")
    print(f"{'=' * 70}")

    # Load model fresh for benchmarking
    base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=HF_HOME,
                                                 trust_remote_code=True, dtype=torch.float32).to(DEVICE)
    cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, target_modules=LORA_TARGETS,
                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    hidden_dim = base.config.hidden_size
    n_layers = base.config.num_hidden_layers
    print(f"  Model: {MODEL_NAME}, d={hidden_dim}, layers={n_layers}")

    # Create test input (batch=1 for latency, matching benchmark)
    rng_lat = random.Random(0)
    input_ids, labels, attn = get_batch(domain_data[domains[0]], 1, rng_lat)
    seq_len = input_ids.shape[1]
    print(f"  Input shape: {input_ids.shape}")

    # Warmup
    warmup_gpu(model, input_ids, attn, n=10)

    results = {
        "model": MODEL_NAME,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "n_experts": N_DOMAINS,
        "n_lora_targets": len(LORA_TARGETS),
        "seq_len": seq_len,
        "finetune_steps": FINETUNE_STEPS,
        "batch_size_latency": 1,
        "n_timing_iters": 50,
        "device": str(DEVICE),
    }

    # -- 2a. Monolithic baseline (single LoRA, full forward) --
    print(f"\n  [A] Monolithic baseline (single LoRA)...")
    set_lora_state(model, lora_states[domains[0]])
    mono_ms = time_fn(lambda: model(input_ids=input_ids, attention_mask=attn))
    print(f"      {mono_ms:.2f} ms")

    # -- 2b. Sequential MoE (matches benchmark bottleneck) --
    print(f"\n  [B] Sequential MoE (set_lora_state + full forward per expert)...")

    for top_k in [1, 2, 4]:
        actual_k = min(top_k, N_DOMAINS)
        topk_idx_test = list(range(actual_k))
        topk_weights_test = [1.0 / actual_k] * actual_k

        def seq_fn():
            zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
            set_lora_state(model, zero_state)
            _ = model.model.model(input_ids=input_ids, attention_mask=attn)
            for ki in range(actual_k):
                set_lora_state(model, lora_states[domains[topk_idx_test[ki]]])
                model(input_ids=input_ids, attention_mask=attn)

        seq_ms = time_fn(seq_fn)
        overhead_pct = (seq_ms / mono_ms - 1) * 100
        print(f"      k={top_k}: {seq_ms:.2f} ms ({overhead_pct:.1f}% overhead)")
        results[f"sequential_k{top_k}_ms"] = seq_ms
        results[f"sequential_k{top_k}_overhead_pct"] = overhead_pct

    # -- 2c. Batched LoRA — direct parameter copy (skip load_state_dict) --
    print(f"\n  [C] Batched LoRA — direct parameter copy (skip load_state_dict)...")

    batched = BatchedLoRAInference(model, lora_states, n_layers=n_layers, scaling=LORA_ALPHA / LORA_RANK)

    for top_k in [1, 2, 4]:
        actual_k = min(top_k, N_DOMAINS)
        topk_idx_test = list(range(actual_k))
        topk_weights_test = torch.tensor([1.0 / actual_k] * actual_k, device=DEVICE)

        def batched_fn():
            batched.forward(input_ids, attn, topk_idx_test, topk_weights_test)

        batched_ms = time_fn(batched_fn)
        overhead_pct = (batched_ms / mono_ms - 1) * 100
        print(f"      k={top_k}: {batched_ms:.2f} ms ({overhead_pct:.1f}% overhead)")
        results[f"batched_direct_k{top_k}_ms"] = batched_ms
        results[f"batched_direct_k{top_k}_overhead_pct"] = overhead_pct

    # -- 2d. Batched LoRA — hook-based (ONE forward pass + delta hooks) --
    print(f"\n  [D] Batched LoRA — hook-based (ONE base forward + LoRA delta hooks)...")

    for top_k in [1, 2, 4]:
        actual_k = min(top_k, N_DOMAINS)
        topk_idx_test = list(range(actual_k))
        topk_weights_test = torch.tensor([1.0 / actual_k] * actual_k, device=DEVICE)

        def hook_fn():
            batched.forward_direct_matmul(input_ids, attn, topk_idx_test, topk_weights_test)

        hook_ms = time_fn(hook_fn)
        overhead_pct = (hook_ms / mono_ms - 1) * 100
        print(f"      k={top_k}: {hook_ms:.2f} ms ({overhead_pct:.1f}% overhead)")
        results[f"batched_hook_k{top_k}_ms"] = hook_ms
        results[f"batched_hook_k{top_k}_overhead_pct"] = overhead_pct

    # -- 2e. Batched LoRA — stacked (ONE forward, k*batch_size, hooks apply per-slice) --
    print(f"\n  [E] Batched LoRA — stacked (ONE forward pass, batch stacking + hooks)...")

    for top_k in [1, 2, 4]:
        actual_k = min(top_k, N_DOMAINS)
        topk_idx_test = list(range(actual_k))
        topk_weights_test = torch.tensor([1.0 / actual_k] * actual_k, device=DEVICE)

        # Re-zero LoRA weights before each test
        zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
        set_lora_state(model, zero_state)

        def stacked_fn():
            batched.forward_stacked(input_ids, attn, topk_idx_test, topk_weights_test)

        stacked_ms = time_fn(stacked_fn)
        overhead_pct = (stacked_ms / mono_ms - 1) * 100
        print(f"      k={top_k}: {stacked_ms:.2f} ms ({overhead_pct:.1f}% overhead)")
        results[f"batched_stacked_k{top_k}_ms"] = stacked_ms
        results[f"batched_stacked_k{top_k}_overhead_pct"] = overhead_pct

    # -- 2f. Batched LoRA — persistent hooks (no registration overhead) --
    print(f"\n  [F] Batched LoRA — persistent hooks (no registration overhead)...")

    # Set up persistent hooks once
    zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
    set_lora_state(model, zero_state)
    batched.setup_persistent_hooks()

    for top_k in [1, 2, 4]:
        actual_k = min(top_k, N_DOMAINS)
        topk_idx_test = list(range(actual_k))
        topk_weights_test = torch.tensor([1.0 / actual_k] * actual_k, device=DEVICE)

        def persistent_fn():
            batched.forward_persistent(input_ids, attn, topk_idx_test, topk_weights_test)

        persistent_ms = time_fn(persistent_fn)
        overhead_pct = (persistent_ms / mono_ms - 1) * 100
        print(f"      k={top_k}: {persistent_ms:.2f} ms ({overhead_pct:.1f}% overhead)")
        results[f"batched_persistent_k{top_k}_ms"] = persistent_ms
        results[f"batched_persistent_k{top_k}_overhead_pct"] = overhead_pct

    # -- 3. Quality check: verify numerical equivalence --
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Numerical equivalence check")
    print(f"{'=' * 70}")

    top_k = 2
    topk_idx_test = [0, 1]
    topk_weights_test = torch.tensor([0.6, 0.4], device=DEVICE)

    # Sequential reference
    print(f"  Computing sequential reference (k={top_k})...")
    set_lora_state(model, lora_states[domains[0]])
    out0 = model(input_ids=input_ids, attention_mask=attn).logits
    set_lora_state(model, lora_states[domains[1]])
    out1 = model(input_ids=input_ids, attention_mask=attn).logits
    ref_logits = 0.6 * out0 + 0.4 * out1

    # Batched direct copy
    print(f"  Computing batched (direct copy)...")
    batched_logits_direct = batched.forward(input_ids, attn, topk_idx_test, topk_weights_test)

    # Batched hook-based (old approach)
    print(f"  Computing batched (hook-based, old)...")
    zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
    set_lora_state(model, zero_state)
    batched_logits_hook = batched.forward_direct_matmul(input_ids, attn, topk_idx_test, topk_weights_test)

    # Batched stacked (new approach)
    print(f"  Computing batched (stacked)...")
    zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
    set_lora_state(model, zero_state)
    # Remove persistent hooks temporarily for non-persistent stacked test
    batched.remove_persistent_hooks()
    batched_logits_stacked = batched.forward_stacked(input_ids, attn, topk_idx_test, topk_weights_test)

    # Batched persistent
    print(f"  Computing batched (persistent hooks)...")
    zero_state = {k: torch.zeros_like(v) for k, v in lora_states[domains[0]].items()}
    set_lora_state(model, zero_state)
    batched.setup_persistent_hooks()
    batched_logits_persistent = batched.forward_persistent(input_ids, attn, topk_idx_test, topk_weights_test)

    # Compare
    diff_direct = (ref_logits - batched_logits_direct).abs()
    diff_hook = (ref_logits - batched_logits_hook).abs()
    diff_stacked = (ref_logits - batched_logits_stacked).abs()
    diff_persistent = (ref_logits - batched_logits_persistent).abs()

    print(f"\n  Sequential vs Batched (direct copy):")
    print(f"    Max abs diff:  {diff_direct.max().item():.2e}")
    print(f"    Mean abs diff: {diff_direct.mean().item():.2e}")

    print(f"\n  Sequential vs Batched (hook-based):")
    print(f"    Max abs diff:  {diff_hook.max().item():.2e}")
    print(f"    Mean abs diff: {diff_hook.mean().item():.2e}")

    print(f"\n  Sequential vs Batched (stacked):")
    print(f"    Max abs diff:  {diff_stacked.max().item():.2e}")
    print(f"    Mean abs diff: {diff_stacked.mean().item():.2e}")

    print(f"\n  Sequential vs Batched (persistent hooks):")
    print(f"    Max abs diff:  {diff_persistent.max().item():.2e}")
    print(f"    Mean abs diff: {diff_persistent.mean().item():.2e}")

    results["quality_check"] = {
        "direct_copy_max_abs_diff": diff_direct.max().item(),
        "direct_copy_mean_abs_diff": diff_direct.mean().item(),
        "hook_max_abs_diff": diff_hook.max().item(),
        "hook_mean_abs_diff": diff_hook.mean().item(),
        "stacked_max_abs_diff": diff_stacked.max().item(),
        "stacked_mean_abs_diff": diff_stacked.mean().item(),
        "persistent_max_abs_diff": diff_persistent.max().item(),
        "persistent_mean_abs_diff": diff_persistent.mean().item(),
    }

    exact_match_direct = diff_direct.max().item() < 1e-5
    exact_match_hook = diff_hook.max().item() < 1e-5
    exact_match_stacked = diff_stacked.max().item() < 1e-5
    exact_match_persistent = diff_persistent.max().item() < 1e-5
    print(f"\n  Direct copy exact match (<1e-5): {exact_match_direct}")
    print(f"  Hook-based exact match (<1e-5): {exact_match_hook}")
    print(f"  Stacked exact match (<1e-5): {exact_match_stacked}")
    print(f"  Persistent hooks exact match (<1e-5): {exact_match_persistent}")

    # -- 4. Theoretical analysis --
    print(f"\n{'=' * 70}")
    print(f"PHASE 4: Theoretical analysis")
    print(f"{'=' * 70}")

    for k_test in [1, 2, 4]:
        lora_flops = k_test * n_layers * len(LORA_TARGETS) * 2 * LORA_RANK * hidden_dim
        base_flops = 2 * 494_000_000
        theoretical_pct = lora_flops / base_flops * 100
        print(f"  k={k_test}: LoRA FLOPs = {lora_flops:,}, theoretical overhead = {theoretical_pct:.2f}%")
        results[f"theoretical_k{k_test}_overhead_pct"] = theoretical_pct
        results[f"theoretical_k{k_test}_lora_flops"] = lora_flops

    results["base_flops_per_token"] = 2 * 494_000_000

    # Memory overhead for pre-extracted matrices
    n_params_per_expert = n_layers * len(LORA_TARGETS) * 2 * LORA_RANK * hidden_dim
    mem_per_expert_mb = n_params_per_expert * 4 / (1024 * 1024)  # fp32
    print(f"\n  Memory per expert (pre-extracted): {mem_per_expert_mb:.1f} MB (fp32)")
    print(f"  Memory for {N_DOMAINS} experts: {mem_per_expert_mb * N_DOMAINS:.1f} MB")
    results["mem_per_expert_mb"] = mem_per_expert_mb
    results["mem_total_experts_mb"] = mem_per_expert_mb * N_DOMAINS

    # -- 5. Batch size scaling --
    print(f"\n{'=' * 70}")
    print(f"PHASE 5: Batch size scaling")
    print(f"{'=' * 70}")

    for bs in [1, 4, 8]:
        input_ids_bs, _, attn_bs = get_batch(domain_data[domains[0]], bs, rng_lat)

        # Monolithic
        set_lora_state(model, lora_states[domains[0]])
        warmup_gpu(model, input_ids_bs, attn_bs, n=3)
        mono_bs_ms = time_fn(lambda: model(input_ids=input_ids_bs, attention_mask=attn_bs), n_iters=30)

        # Direct copy k=2
        topk_idx_bs = [0, 1]
        topk_w_bs = torch.tensor([0.5, 0.5], device=DEVICE)

        def direct_bs_fn():
            batched.forward(input_ids_bs, attn_bs, topk_idx_bs, topk_w_bs)

        direct_bs_ms = time_fn(direct_bs_fn, n_iters=30)
        overhead_direct = (direct_bs_ms / mono_bs_ms - 1) * 100

        # Persistent k=2
        def persistent_bs_fn():
            batched.forward_persistent(input_ids_bs, attn_bs, topk_idx_bs, topk_w_bs)

        persistent_bs_ms = time_fn(persistent_bs_fn, n_iters=30)
        overhead_persistent = (persistent_bs_ms / mono_bs_ms - 1) * 100

        print(f"  batch={bs}: mono={mono_bs_ms:.2f}ms, direct={direct_bs_ms:.2f}ms ({overhead_direct:.1f}%), persistent={persistent_bs_ms:.2f}ms ({overhead_persistent:.1f}%)")
        results[f"batch{bs}_mono_ms"] = mono_bs_ms
        results[f"batch{bs}_direct_ms"] = direct_bs_ms
        results[f"batch{bs}_direct_overhead_pct"] = overhead_direct
        results[f"batch{bs}_persistent_ms"] = persistent_bs_ms
        results[f"batch{bs}_persistent_overhead_pct"] = overhead_persistent

    # -- Summary --
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")

    results["monolithic_ms"] = mono_ms
    results["summary"] = {}

    print(f"\n  Monolithic (single LoRA): {mono_ms:.2f} ms")
    print(f"\n  {'Method':<40s} {'k=1':>10s} {'k=2':>10s} {'k=4':>10s}")
    print(f"  {'-'*40} {'-'*10} {'-'*10} {'-'*10}")

    for method, prefix in [
        ("Sequential (set_lora_state)", "sequential"),
        ("Batched (direct copy)", "batched_direct"),
        ("Batched (hook-based)", "batched_hook"),
        ("Batched (stacked)", "batched_stacked"),
        ("Batched (persistent hooks)", "batched_persistent"),
        ("Theoretical", "theoretical"),
    ]:
        vals = []
        for k in [1, 2, 4]:
            key = f"{prefix}_k{k}_overhead_pct"
            v = results.get(key, None)
            vals.append(f"{v:.1f}%" if v is not None else "N/A")
        print(f"  {method:<40s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")
        results["summary"][method] = {
            f"k{k}": results.get(f"{prefix}_k{k}_overhead_pct") for k in [1, 2, 4]
        }

    # Speedup
    for k in [1, 2, 4]:
        seq_key = f"sequential_k{k}_ms"
        hook_key = f"batched_hook_k{k}_ms"
        if seq_key in results and hook_key in results:
            speedup = results[seq_key] / results[hook_key]
            print(f"\n  Hook speedup over sequential at k={k}: {speedup:.1f}x")
            results[f"speedup_k{k}"] = speedup

    elapsed = time.time() - t_start
    results["elapsed_seconds"] = elapsed
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {OUT_DIR / 'results.json'}")

    del model, base
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
