"""Gap-as-Signal Bridge Experiment — PyTorch/CUDA version for RunPod.

Bridges micro (d=64, r²=0.74) to intermediate scale (d=256, ~2M params).
20 seeds, 7 cosine levels, bootstrap CIs, N=2 and N=4 expert configs.

Self-contained: no MLX imports. Runs on CUDA.
"""

import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Data (self-contained, no MLX) ────────────────────────────────────────

DATA_URL = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
DATA_PATH = Path(__file__).parent.parent.parent.parent / "input.txt"


def load_names(path=None):
    path = path or DATA_PATH
    if not os.path.exists(path):
        urllib.request.urlretrieve(DATA_URL, str(path))
    return [line.strip() for line in open(path) if line.strip()]


class CharTokenizer:
    def __init__(self, docs):
        self.chars = sorted(set("".join(docs)))
        self.bos = len(self.chars)
        self.vocab_size = len(self.chars) + 1
        self._c2i = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self._c2i[c] for c in s]


class CharDataset:
    def __init__(self, docs, tokenizer, block_size=32):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.sequences = []
        for doc in docs:
            tokens = [tokenizer.bos] + tokenizer.encode(doc) + [tokenizer.bos]
            self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def get_batch(self, batch_size, rng=None):
        rng = rng or random
        seqs = rng.choices(self.sequences, k=batch_size)
        inputs, targets = [], []
        for seq in seqs:
            n = min(self.block_size, len(seq) - 1)
            inp = seq[:n] + [0] * (self.block_size - n)
            tgt = seq[1:n + 1] + [0] * (self.block_size - n)
            inputs.append(inp)
            targets.append(tgt)
        return (torch.tensor(inputs, device=DEVICE),
                torch.tensor(targets, device=DEVICE))


def domain_split(docs, method="binary"):
    if method == "binary":
        d0 = [d for d in docs if d[0].lower() <= "m"]
        d1 = [d for d in docs if d[0].lower() > "m"]
        return {"a_m": d0, "n_z": d1}
    if method == "quintary":
        ranges = [("a", "e"), ("f", "j"), ("k", "o"), ("p", "t"), ("u", "z")]
        return {f"{lo}_{hi}": [d for d in docs if lo <= d[0].lower() <= hi]
                for lo, hi in ranges}
    raise ValueError(f"Unknown split: {method}")


def train_val_split(docs, val_frac=0.2, seed=42):
    rng = random.Random(seed)
    docs = list(docs)
    rng.shuffle(docs)
    n_val = int(len(docs) * val_frac)
    return docs[n_val:], docs[:n_val]


# ── Model ────────────────────────────────────────────────────────────────

BRIDGE = dict(n_embd=256, n_head=8, n_layer=6, block_size=64)
LORA_RANK = 16
LORA_ALPHA = 1.0

PRETRAIN_STEPS = 600
FINETUNE_STEPS = 600
MAX_CAL_STEPS = 500
CAL_EVAL_EVERY = 10
CONVERGENCE_THRESHOLD = 0.005
BATCH_SIZE = 32
LR = 1e-3
TARGET_COSINES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
N_SEEDS = 20


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        ms = torch.mean(x * x, dim=-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.wq = nn.Linear(n_embd, n_embd, bias=False)
        self.wk = nn.Linear(n_embd, n_embd, bias=False)
        self.wv = nn.Linear(n_embd, n_embd, bias=False)
        self.wo = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.wo(out)


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size=28, block_size=32, n_embd=64, n_head=4, n_layer=4):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ── LoRA ─────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=16, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.rank = rank
        self.alpha = alpha
        scale = (2.0 / in_dim) ** 0.5
        self.A = nn.Parameter(torch.randn(in_dim, rank) * scale)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        return self.linear(x) + (self.alpha / self.rank) * (x @ self.A @ self.B)

    def get_delta(self):
        return (self.alpha / self.rank) * (self.A @ self.B)


class LoRAMLP(nn.Module):
    def __init__(self, n_embd, rank=16, alpha=1.0):
        super().__init__()
        self.fc1 = LoRALinear(n_embd, 4 * n_embd, rank, alpha)
        self.fc2 = LoRALinear(4 * n_embd, n_embd, rank, alpha)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class LoRABlock(nn.Module):
    def __init__(self, n_embd, n_head, rank=16, alpha=1.0):
        super().__init__()
        self.norm1 = RMSNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.norm2 = RMSNorm(n_embd)
        self.mlp = LoRAMLP(n_embd, rank, alpha)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LoRAGPT(nn.Module):
    def __init__(self, vocab_size=28, block_size=64, n_embd=256, n_head=8,
                 n_layer=6, lora_rank=16, lora_alpha=1.0):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.norm0 = RMSNorm(n_embd)
        self.layers = nn.ModuleList([
            LoRABlock(n_embd, n_head, lora_rank, lora_alpha) for _ in range(n_layer)
        ])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


# ── Delta Manipulation ───────────────────────────────────────────────────

def get_deltas(model):
    deltas = []
    for l_idx, layer in enumerate(model.layers):
        for name, fc in [('fc1', layer.mlp.fc1), ('fc2', layer.mlp.fc2)]:
            delta = (fc.alpha / fc.rank) * (fc.A.data @ fc.B.data)
            deltas.append((l_idx, name, delta.detach()))
    return deltas


def flatten_deltas(deltas):
    return torch.cat([d[2].reshape(-1) for d in deltas])


def unflatten_deltas(flat, template_deltas):
    result = []
    offset = 0
    for l_idx, name, template in template_deltas:
        size = template.numel()
        shape = template.shape
        chunk = flat[offset:offset + size].reshape(shape)
        result.append((l_idx, name, chunk))
        offset += size
    return result


def project_to_target_cosine(deltas_a, deltas_b, target_cos):
    a = flatten_deltas(deltas_a)
    b = flatten_deltas(deltas_b)
    a_norm = torch.norm(a)
    b_norm = torch.norm(b)
    a_hat = a / (a_norm + 1e-12)
    b_parallel = torch.dot(b, a_hat) * a_hat
    b_perp = b - b_parallel
    b_perp_norm = torch.norm(b_perp)
    b_perp_hat = b_perp / (b_perp_norm + 1e-12)
    sin_component = math.sqrt(max(0, 1 - target_cos ** 2))
    b_proj = target_cos * b_norm * a_hat + sin_component * b_norm * b_perp_hat
    actual_cos = (torch.dot(a, b_proj) / (a_norm * torch.norm(b_proj) + 1e-12)).item()
    return unflatten_deltas(b_proj.detach(), deltas_b), actual_cos


def copy_base_to_lora(base_model, lora_model):
    for l_idx in range(len(base_model.layers)):
        bl = base_model.layers[l_idx]
        ll = lora_model.layers[l_idx]
        ll.attn.wq.weight.data.copy_(bl.attn.wq.weight.data)
        ll.attn.wk.weight.data.copy_(bl.attn.wk.weight.data)
        ll.attn.wv.weight.data.copy_(bl.attn.wv.weight.data)
        ll.attn.wo.weight.data.copy_(bl.attn.wo.weight.data)
        ll.mlp.fc1.linear.weight.data.copy_(bl.mlp.fc1.weight.data)
        ll.mlp.fc2.linear.weight.data.copy_(bl.mlp.fc2.weight.data)
    lora_model.wte.weight.data.copy_(base_model.wte.weight.data)
    lora_model.wpe.weight.data.copy_(base_model.wpe.weight.data)
    lora_model.lm_head.weight.data.copy_(base_model.lm_head.weight.data)


def freeze_except_lora(model):
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.layers:
        layer.mlp.fc1.A.requires_grad = True
        layer.mlp.fc1.B.requires_grad = True
        layer.mlp.fc2.A.requires_grad = True
        layer.mlp.fc2.B.requires_grad = True


def apply_deltas_to_base(base_model, deltas, vocab_size):
    model = GPT(vocab_size=vocab_size, **BRIDGE).to(DEVICE)
    model.load_state_dict(base_model.state_dict())
    for l_idx, name, delta in deltas:
        layer = model.layers[l_idx]
        if name == 'fc1':
            layer.mlp.fc1.weight.data += delta.T
        elif name == 'fc2':
            layer.mlp.fc2.weight.data += delta.T
    return model


# ── Routed Model ─────────────────────────────────────────────────────────

class RoutedDeltaGPT(nn.Module):
    def __init__(self, base_model, delta_sets, vocab_size, top_k=2):
        super().__init__()
        self.n_experts = len(delta_sets)
        self.top_k = min(top_k, self.n_experts)
        n_embd = BRIDGE['n_embd']
        n_layer = len(base_model.layers)

        self.wte = base_model.wte
        self.wpe = base_model.wpe
        self.norm0 = base_model.norm0
        self.base_layers = base_model.layers
        self.lm_head = base_model.lm_head

        # Pre-compute expert MLP weights per layer
        self.expert_fc1 = nn.ParameterList()
        self.expert_fc2 = nn.ParameterList()

        for l_idx in range(n_layer):
            base_fc1 = base_model.layers[l_idx].mlp.fc1.weight.data
            base_fc2 = base_model.layers[l_idx].mlp.fc2.weight.data
            fc1_list, fc2_list = [], []
            for deltas in delta_sets:
                for dl_idx, name, delta in deltas:
                    if dl_idx == l_idx and name == 'fc1':
                        fc1_list.append(base_fc1 + delta.T)
                    elif dl_idx == l_idx and name == 'fc2':
                        fc2_list.append(base_fc2 + delta.T)
            self.expert_fc1.append(nn.Parameter(torch.stack(fc1_list), requires_grad=False))
            self.expert_fc2.append(nn.Parameter(torch.stack(fc2_list), requires_grad=False))

        self.routers = nn.ModuleList([
            nn.Linear(n_embd, self.n_experts, bias=False) for _ in range(n_layer)
        ])

    def forward(self, tokens):
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        x = self.wte(tokens) + self.wpe(pos)
        x = self.norm0(x)

        for l_idx, base_layer in enumerate(self.base_layers):
            x = x + base_layer.attn(base_layer.norm1(x))
            h = base_layer.norm2(x)
            scores = self.routers[l_idx](h)
            probs = F.softmax(scores, dim=-1)
            _, top_indices = scores.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(scores)
            mask.scatter_(-1, top_indices, 1.0)
            masked_probs = probs * mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-8)

            delta_out = torch.zeros_like(h)
            for e in range(self.n_experts):
                w_e = masked_probs[..., e:e+1]
                fc1_w = self.expert_fc1[l_idx][e]
                fc2_w = self.expert_fc2[l_idx][e]
                expert_out = F.relu(h @ fc1_w.T) @ fc2_w.T
                delta_out = delta_out + w_e * expert_out
            x = x + delta_out

        return self.lm_head(x)


# ── Training / Eval ──────────────────────────────────────────────────────

def ntp_loss(model, inputs, targets):
    logits = model(inputs)
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))


@torch.no_grad()
def evaluate(model, dataset, batch_size=32, n_batches=10):
    model.eval()
    rng = random.Random(0)
    total = 0.0
    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(batch_size, rng)
        loss = ntp_loss(model, inputs, targets)
        total += loss.item()
    model.train()
    return total / n_batches


# ── Gap Measurement ──────────────────────────────────────────────────────

@torch.no_grad()
def measure_function_space_gap(composed_model, joint_model, dataset, n_batches=20):
    composed_model.eval()
    joint_model.eval()
    rng = random.Random(0)
    total_ce_c, total_ce_j, total_kl, total_l1 = 0.0, 0.0, 0.0, 0.0
    n_tokens = 0

    for _ in range(n_batches):
        inputs, targets = dataset.get_batch(BATCH_SIZE, rng)
        logits_c = composed_model(inputs)
        logits_j = joint_model(inputs)
        B, T, V = logits_c.shape

        ce_c = F.cross_entropy(logits_c.view(B*T, V), targets.view(B*T), reduction='sum')
        ce_j = F.cross_entropy(logits_j.view(B*T, V), targets.view(B*T), reduction='sum')
        total_ce_c += ce_c.item()
        total_ce_j += ce_j.item()

        prob_c = F.softmax(logits_c.view(B*T, V), dim=-1)
        prob_j = F.softmax(logits_j.view(B*T, V), dim=-1)
        kl = (prob_j * (torch.log(prob_j + 1e-10) - torch.log(prob_c + 1e-10))).sum().item()
        l1 = torch.abs(prob_c - prob_j).sum().item()
        total_kl += kl
        total_l1 += l1
        n_tokens += B * T

    composed_model.train()
    joint_model.train()
    return {
        'ce_gap': abs(total_ce_c / n_tokens - total_ce_j / n_tokens),
        'kl_gap': total_kl / n_tokens,
        'prob_l1': total_l1 / n_tokens,
    }


# ── Calibration ──────────────────────────────────────────────────────────

def calibrate_router(model, domain_datasets, val_ds, joint_val_loss,
                     steps=500, lr=1e-3, seed=42):
    # Freeze everything except routers
    for p in model.parameters():
        p.requires_grad = False
    for router in model.routers:
        for p in router.parameters():
            p.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for r in model.routers for p in r.parameters()], lr=lr
    )
    rng = random.Random(seed)
    convergence_target = joint_val_loss * (1 + CONVERGENCE_THRESHOLD)
    loss_curve = []
    steps_to_converge = None
    domains = list(domain_datasets.keys())

    model.train()
    for step in range(1, steps + 1):
        domain = domains[step % len(domains)]
        inputs, targets = domain_datasets[domain].get_batch(BATCH_SIZE, rng)
        optimizer.zero_grad()
        loss = ntp_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()

        if step % CAL_EVAL_EVERY == 0:
            val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=5)
            loss_curve.append((step, val_loss))
            if steps_to_converge is None and val_loss <= convergence_target:
                steps_to_converge = step

    final_val_loss = evaluate(model, val_ds, BATCH_SIZE, n_batches=10)

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    auc = sum(vl for _, vl in loss_curve) / len(loss_curve) if loss_curve else float('inf')
    return {
        'loss_curve': [(s, v) for s, v in loss_curve],
        'steps_to_converge': steps_to_converge,
        'final_val_loss': final_val_loss,
        'auc': auc,
    }


# ── Single Trial ─────────────────────────────────────────────────────────

def run_trial(target_cos, base_model, all_deltas, joint_model,
              domain_train_datasets, domain_val_datasets, joint_val_ds,
              joint_val_loss, V, n_experts, top_k, seed=42):
    # Project experts to target cosine
    projected_deltas = []
    for e_idx in range(n_experts):
        if e_idx == 0:
            projected_deltas.append(all_deltas[e_idx])
        else:
            proj, _ = project_to_target_cosine(all_deltas[0], all_deltas[e_idx], target_cos)
            projected_deltas.append(proj)

    # Task arithmetic: average all deltas
    n_matrices = len(all_deltas[0])
    ta_deltas = []
    for m_idx in range(n_matrices):
        l_idx, name, _ = all_deltas[0][m_idx]
        avg = sum(projected_deltas[e][m_idx][2] for e in range(n_experts)) / n_experts
        ta_deltas.append((l_idx, name, avg))

    ta_model = apply_deltas_to_base(base_model, ta_deltas, V)
    gap = measure_function_space_gap(ta_model, joint_model, joint_val_ds)
    del ta_model

    # Create routed model
    base_copy = GPT(vocab_size=V, **BRIDGE).to(DEVICE)
    base_copy.load_state_dict(base_model.state_dict())
    routed = RoutedDeltaGPT(base_copy, projected_deltas, V, top_k=top_k).to(DEVICE)

    gap_pre = measure_function_space_gap(routed, joint_model, joint_val_ds, n_batches=10)

    cal = calibrate_router(
        routed, domain_train_datasets, joint_val_ds,
        joint_val_loss, steps=MAX_CAL_STEPS, lr=LR, seed=seed
    )

    gap_post = measure_function_space_gap(routed, joint_model, joint_val_ds, n_batches=10)

    domain_vals = {}
    for dname, dval in domain_val_datasets.items():
        domain_vals[dname] = evaluate(routed, dval, BATCH_SIZE, n_batches=5)
    avg_domain_val = statistics.mean(domain_vals.values())
    vs_joint_pct = (avg_domain_val - joint_val_loss) / joint_val_loss * 100

    del routed, base_copy
    torch.cuda.empty_cache()

    return {
        'target_cos': target_cos,
        'n_experts': n_experts,
        'top_k': top_k,
        'ce_gap_ta': gap['ce_gap'],
        'kl_gap_ta': gap['kl_gap'],
        'prob_l1_ta': gap['prob_l1'],
        'ce_gap_pre': gap_pre['ce_gap'],
        'kl_gap_pre': gap_pre['kl_gap'],
        'ce_gap_post': gap_post['ce_gap'],
        'kl_gap_post': gap_post['kl_gap'],
        'final_val_loss': cal['final_val_loss'],
        'avg_domain_val': avg_domain_val,
        'vs_joint_pct': vs_joint_pct,
        'steps_to_converge': cal['steps_to_converge'],
        'auc': cal['auc'],
        'domain_vals': domain_vals,
    }


# ── Full Experiment (one seed) ───────────────────────────────────────────

def run_experiment(seed=42, n_experts=2, top_k=2, verbose=True):
    if verbose:
        print(f"\n{'='*70}")
        print(f"GAP-AS-SIGNAL BRIDGE (seed={seed}, N={n_experts}, k={top_k})")
        print(f"  d={BRIDGE['n_embd']}, n_layer={BRIDGE['n_layer']}, "
              f"rank={LORA_RANK}, block_size={BRIDGE['block_size']}")
        print(f"{'='*70}")

    torch.manual_seed(seed)
    random.seed(seed)

    docs = load_names()
    tokenizer = CharTokenizer(docs)
    V = tokenizer.vocab_size

    if n_experts <= 2:
        splits = domain_split(docs, method="binary")
        domain_names = list(splits.keys())[:n_experts]
    else:
        splits = domain_split(docs, method="quintary")
        domain_names = list(splits.keys())[:n_experts]

    all_train, all_val = train_val_split(docs, seed=seed)
    joint_train = CharDataset(all_train, tokenizer, BRIDGE['block_size'])
    joint_val = CharDataset(all_val, tokenizer, BRIDGE['block_size'])

    domain_train = {}
    domain_val = {}
    for dname in domain_names:
        dtrain, dval = train_val_split(splits[dname], seed=seed)
        domain_train[dname] = CharDataset(dtrain, tokenizer, BRIDGE['block_size'])
        domain_val[dname] = CharDataset(dval, tokenizer, BRIDGE['block_size'])

    # === 1. Joint training baseline ===
    if verbose:
        print("\n--- Joint training ---")
    model_joint = GPT(vocab_size=V, **BRIDGE).to(DEVICE)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(model_joint.parameters(), lr=LR)
    total_steps = FINETUNE_STEPS * n_experts
    for step in range(1, total_steps + 1):
        dname = domain_names[step % n_experts]
        inputs, targets = domain_train[dname].get_batch(BATCH_SIZE, rng)
        optimizer.zero_grad()
        loss = ntp_loss(model_joint, inputs, targets)
        loss.backward()
        optimizer.step()
        if verbose and step % 200 == 0:
            print(f"  step {step}/{total_steps} | loss {loss.item():.4f}")

    joint_domain_vals = {}
    for dname in domain_names:
        joint_domain_vals[dname] = evaluate(model_joint, domain_val[dname], BATCH_SIZE)
    joint_val_loss = statistics.mean(joint_domain_vals.values())
    joint_on_joint = evaluate(model_joint, joint_val, BATCH_SIZE)
    if verbose:
        print(f"  Joint val: avg={joint_val_loss:.4f}, joint_ds={joint_on_joint:.4f}")

    # === 2. Pretrain base model ===
    if verbose:
        print("\n--- Pretraining base ---")
    base_model = GPT(vocab_size=V, **BRIDGE).to(DEVICE)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=LR)
    for step in range(1, PRETRAIN_STEPS + 1):
        inputs, targets = joint_train.get_batch(BATCH_SIZE, rng)
        optimizer.zero_grad()
        loss = ntp_loss(base_model, inputs, targets)
        loss.backward()
        optimizer.step()
    if verbose:
        base_val = evaluate(base_model, joint_val, BATCH_SIZE)
        print(f"  Base val: {base_val:.4f}")

    # === 3. Fine-tune LoRA experts ===
    all_deltas = []
    for e_idx, dname in enumerate(domain_names):
        if verbose:
            print(f"\n--- Fine-tuning LoRA expert {e_idx} ({dname}) ---")
        torch.manual_seed(seed + e_idx + 1000)
        lora_model = LoRAGPT(vocab_size=V, **BRIDGE,
                             lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA).to(DEVICE)
        copy_base_to_lora(base_model, lora_model)
        freeze_except_lora(lora_model)

        rng = random.Random(seed + e_idx)
        lora_params = [p for p in lora_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(lora_params, lr=LR)
        for step in range(1, FINETUNE_STEPS + 1):
            inputs, targets = domain_train[dname].get_batch(BATCH_SIZE, rng)
            optimizer.zero_grad()
            loss = ntp_loss(lora_model, inputs, targets)
            loss.backward()
            optimizer.step()

        deltas = get_deltas(lora_model)
        all_deltas.append(deltas)

        if verbose:
            val_loss = evaluate(lora_model, domain_val[dname], BATCH_SIZE)
            print(f"  Expert {e_idx} val ({dname}): {val_loss:.4f}")
        del lora_model
        torch.cuda.empty_cache()

    # === 4. Natural cosines ===
    natural_cosines = {}
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            flat_i = flatten_deltas(all_deltas[i])
            flat_j = flatten_deltas(all_deltas[j])
            cos = (torch.dot(flat_i, flat_j) /
                   (torch.norm(flat_i) * torch.norm(flat_j) + 1e-12)).item()
            natural_cosines[f"{i}-{j}"] = cos
    if verbose:
        print(f"\n  Natural cosines: {natural_cosines}")

    # === 5. Cosine sweep ===
    if verbose:
        print(f"\n{'='*70}")
        print("RUNNING COSINE SWEEP")
        print(f"{'='*70}")

    trials = []
    for target_cos in TARGET_COSINES:
        if verbose:
            print(f"\n  --- cos = {target_cos:.1f} ---")
        trial = run_trial(
            target_cos, base_model, all_deltas, model_joint,
            domain_train, domain_val, joint_val,
            joint_on_joint, V, n_experts, top_k, seed=seed
        )
        trials.append(trial)
        if verbose:
            print(f"    CE gap: {trial['ce_gap_ta']:.4f}, "
                  f"final VL: {trial['final_val_loss']:.4f}, "
                  f"vs joint: {trial['vs_joint_pct']:+.1f}%")

    del model_joint, base_model
    torch.cuda.empty_cache()

    return {
        'seed': seed,
        'n_experts': n_experts,
        'top_k': top_k,
        'joint_val_loss': joint_val_loss,
        'joint_on_joint': joint_on_joint,
        'joint_domain_vals': joint_domain_vals,
        'natural_cosines': natural_cosines,
        'trials': trials,
    }


# ── Statistical Analysis ────────────────────────────────────────────────

def compute_r_squared(xs, ys):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    ss_yy = sum((y - mean_y) ** 2 for y in ys)
    if ss_xx == 0 or ss_yy == 0:
        return 0.0, 0.0
    r = ss_xy / (ss_xx * ss_yy) ** 0.5
    return r ** 2, r


def bootstrap_ci(xs, ys, n_bootstrap=1000, ci=0.95):
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    rng = random.Random(42)
    r2_samples = []
    for _ in range(n_bootstrap):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        bx = [xs[i] for i in indices]
        by = [ys[i] for i in indices]
        r2, _ = compute_r_squared(bx, by)
        r2_samples.append(r2)
    r2_samples.sort()
    alpha = (1 - ci) / 2
    lo = r2_samples[int(alpha * n_bootstrap)]
    hi = r2_samples[int((1 - alpha) * n_bootstrap)]
    return statistics.mean(r2_samples), lo, hi


def analyze_results(all_experiments, config_label=""):
    print(f"\n\n{'='*80}")
    print(f"GAP-AS-SIGNAL BRIDGE ANALYSIS {config_label}")
    print(f"  d={BRIDGE['n_embd']}, rank={LORA_RANK}, n_seeds={len(all_experiments)}")
    print(f"{'='*80}")

    by_cos = {}
    for exp in all_experiments:
        for trial in exp['trials']:
            cos = trial['target_cos']
            if cos not in by_cos:
                by_cos[cos] = []
            by_cos[cos].append(trial)

    print(f"\n{'Cos':>5} | {'CE Gap':>10} | {'KL Gap':>10} | {'vs Joint':>12} | "
          f"{'AUC':>10} | {'Steps':>8} | n")
    print("-" * 75)

    cos_means_quality = {}
    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        ce_gaps = [t['ce_gap_ta'] for t in trials]
        kl_gaps = [t['kl_gap_ta'] for t in trials]
        vs_joints = [t['vs_joint_pct'] for t in trials]
        aucs = [t['auc'] for t in trials]
        steps_conv = [t['steps_to_converge'] for t in trials if t['steps_to_converge']]

        n = len(trials)
        ce_mean = statistics.mean(ce_gaps)
        ce_std = statistics.stdev(ce_gaps) if n > 1 else 0
        kl_mean = statistics.mean(kl_gaps)
        vj_mean = statistics.mean(vs_joints)
        vj_std = statistics.stdev(vs_joints) if n > 1 else 0
        auc_mean = statistics.mean(aucs)
        steps_str = f"{statistics.mean(steps_conv):.0f}" if steps_conv else "N/C"

        cos_means_quality[cos] = vj_mean
        print(f"{cos:>5.1f} | {ce_mean:>7.4f}+{ce_std:.3f} | {kl_mean:>10.4f} | "
              f"{vj_mean:>+7.1f}%+{vj_std:.1f} | {auc_mean:>10.4f} | {steps_str:>8} | {n}")

    # Correlation analysis
    ce_gaps_all, vs_joint_all, cos_list = [], [], []
    kl_gaps_all, auc_all = [], []
    for exp in all_experiments:
        for trial in exp['trials']:
            ce_gaps_all.append(trial['ce_gap_ta'])
            kl_gaps_all.append(trial['kl_gap_ta'])
            vs_joint_all.append(trial['vs_joint_pct'])
            auc_all.append(trial['auc'])
            cos_list.append(trial['target_cos'])

    mean_ce_by_cos = []
    mean_vj_by_cos = []
    for cos in sorted(by_cos.keys()):
        trials = by_cos[cos]
        mean_ce_by_cos.append(statistics.mean([t['ce_gap_ta'] for t in trials]))
        mean_vj_by_cos.append(statistics.mean([t['vs_joint_pct'] for t in trials]))

    print(f"\n{'='*80}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*80}")

    r2_ce_q_pooled, r_ce_q = compute_r_squared(ce_gaps_all, vs_joint_all)
    r2_ce_q_mean, r2_ce_q_lo, r2_ce_q_hi = bootstrap_ci(ce_gaps_all, vs_joint_all)
    print(f"\n1. CE Gap vs Quality (pooled, n={len(ce_gaps_all)}):")
    print(f"   r^2 = {r2_ce_q_pooled:.4f}, r = {r_ce_q:.4f}")
    print(f"   Bootstrap 95% CI: [{r2_ce_q_lo:.4f}, {r2_ce_q_hi:.4f}]")

    r2_ce_q_mean_curve, _ = compute_r_squared(mean_ce_by_cos, mean_vj_by_cos)
    print(f"\n2. CE Gap vs Quality (mean curve, n=7):")
    print(f"   r^2 = {r2_ce_q_mean_curve:.4f}")

    r2_cos_q, r_cos_q = compute_r_squared(cos_list, vs_joint_all)
    r2_cos_q_mean, r2_cos_q_lo, r2_cos_q_hi = bootstrap_ci(cos_list, vs_joint_all)
    print(f"\n3. Cosine vs Quality (pooled):")
    print(f"   r^2 = {r2_cos_q:.4f}, r = {r_cos_q:.4f}")
    print(f"   Bootstrap 95% CI: [{r2_cos_q_lo:.4f}, {r2_cos_q_hi:.4f}]")

    r2_kl_q, r_kl_q = compute_r_squared(kl_gaps_all, vs_joint_all)
    print(f"\n4. KL Gap vs Quality: r^2 = {r2_kl_q:.4f}")

    r2_cos_ce, _ = compute_r_squared(cos_list, ce_gaps_all)
    print(f"\n5. Cosine vs CE Gap (sanity): r^2 = {r2_cos_ce:.4f}")

    # Monotonicity
    quality_curve = [cos_means_quality[c] for c in sorted(cos_means_quality.keys())]
    is_monotonic = all(quality_curve[i] >= quality_curve[i-1] - 1e-6
                       for i in range(1, len(quality_curve)))
    print(f"\n6. Monotonicity:")
    print(f"   Quality curve: {[f'{v:+.1f}%' for v in quality_curve]}")
    print(f"   Monotonic: {'YES' if is_monotonic else 'NO'}")

    # Effect size
    if 0.0 in cos_means_quality and 0.5 in cos_means_quality:
        effect = cos_means_quality[0.5] - cos_means_quality[0.0]
        print(f"\n7. Effect size (cos=0.0 vs cos=0.5): {effect:+.2f}pp")

    # Kill criteria
    print(f"\n{'='*80}")
    print("KILL CRITERIA")
    print(f"{'='*80}")

    best_r2 = max(r2_ce_q_pooled, r2_cos_q, r2_kl_q)
    kills = 0
    passes = 0

    if best_r2 >= 0.3:
        passes += 1
        print(f"\n  [PASS] r^2 = {best_r2:.4f} >= 0.3")
    else:
        kills += 1
        print(f"\n  [KILL] r^2 = {best_r2:.4f} < 0.3")

    if is_monotonic:
        passes += 1
        print(f"  [PASS] Monotonic")
    else:
        kills += 1
        print(f"  [KILL] Non-monotonic")

    if 0.0 in cos_means_quality and 0.5 in cos_means_quality:
        effect = cos_means_quality[0.5] - cos_means_quality[0.0]
        if effect > 0.5:
            passes += 1
            print(f"  [PASS] Effect size: {effect:+.2f}pp > 0.5pp")
        else:
            kills += 1
            print(f"  [KILL] Effect size: {effect:+.2f}pp < 0.5pp")

    all_steps = [t['steps_to_converge'] for exp in all_experiments
                 for t in exp['trials'] if t['steps_to_converge']]
    if all_steps:
        max_s = max(all_steps)
        if max_s <= 500:
            passes += 1
            print(f"  [PASS] Calibration steps max={max_s}")
        else:
            kills += 1
            print(f"  [KILL] Calibration steps max={max_s} > 500")

    # Natural cosines
    all_nat_cos = []
    for exp in all_experiments:
        all_nat_cos.extend(exp['natural_cosines'].values())
    if all_nat_cos:
        mean_nat = statistics.mean(all_nat_cos)
        expected = LORA_RANK / math.sqrt(BRIDGE['n_embd'] * 4 * BRIDGE['n_embd'] *
                                          2 * BRIDGE['n_layer'])
        print(f"\n  Natural cosine: {mean_nat:.4f} (predicted: {expected:.4f})")

    print(f"\n{'='*80}")
    if kills == 0:
        print(f"  CONFIRMED at d={BRIDGE['n_embd']}: {passes}/{passes} pass")
    elif kills >= 2:
        print(f"  KILLED: {kills} fail")
    else:
        print(f"  PARTIAL: {passes} pass, {kills} fail")
    print(f"{'='*80}")

    return {
        'r2_ce_quality_pooled': r2_ce_q_pooled,
        'r2_ce_quality_ci_lo': r2_ce_q_lo,
        'r2_ce_quality_ci_hi': r2_ce_q_hi,
        'r2_ce_quality_mean_curve': r2_ce_q_mean_curve,
        'r2_cos_quality': r2_cos_q,
        'r2_kl_quality': r2_kl_q,
        'is_monotonic': is_monotonic,
        'best_r2': best_r2,
        'n_seeds': len(all_experiments),
        'kills': kills,
        'passes': passes,
    }


# ── Runner ───────────────────────────────────────────────────────────────

def run_bridge(n_seeds=None, n_experts=2, top_k=2, verbose=True):
    if n_seeds is None:
        n_seeds = N_SEEDS

    t0 = time.time()
    all_experiments = []
    seeds = list(range(42, 42 + n_seeds))
    out_dir = Path(__file__).parent

    for i, seed in enumerate(seeds):
        if verbose:
            print(f"\n\n{'#'*80}")
            print(f"# SEED {i+1}/{n_seeds} (seed={seed})")
            print(f"{'#'*80}")
        result = run_experiment(seed=seed, n_experts=n_experts,
                               top_k=top_k, verbose=verbose)
        all_experiments.append(result)

        # Save intermediate
        with open(out_dir / f"results_N{n_experts}_k{top_k}.json", "w") as f:
            json.dump(all_experiments, f, indent=2, default=str)

    config_label = f"(N={n_experts}, k={top_k})"
    analysis = analyze_results(all_experiments, config_label)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return all_experiments, analysis


def run_full_experiment(verbose=True):
    results = {}
    out_dir = Path(__file__).parent

    # Phase 1a: N=2, k=2
    print("\n" + "="*80)
    print("PHASE 1a: N=2, k=2 (micro-matched)")
    print("="*80)
    exps_2, analysis_2 = run_bridge(n_seeds=N_SEEDS, n_experts=2, top_k=2, verbose=verbose)
    results['N2_k2'] = {'experiments': exps_2, 'analysis': analysis_2}

    # Phase 1b: N=4, k=2
    print("\n" + "="*80)
    print("PHASE 1b: N=4, k=2 (routing selection)")
    print("="*80)
    exps_4, analysis_4 = run_bridge(n_seeds=N_SEEDS, n_experts=4, top_k=2, verbose=verbose)
    results['N4_k2'] = {'experiments': exps_4, 'analysis': analysis_4}

    # Save summary
    summary = {
        'config': {
            'd': BRIDGE['n_embd'],
            'n_head': BRIDGE['n_head'],
            'n_layer': BRIDGE['n_layer'],
            'block_size': BRIDGE['block_size'],
            'lora_rank': LORA_RANK,
            'lora_alpha': LORA_ALPHA,
        },
        'N2_k2': analysis_2,
        'N4_k2': analysis_4,
    }
    with open(out_dir / "results_bridge.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "="*80)
    print("COMPARATIVE SUMMARY")
    print("="*80)
    for label, analysis in [("N=2,k=2", analysis_2), ("N=4,k=2", analysis_4)]:
        print(f"\n  {label}:")
        print(f"    Best r^2: {analysis['best_r2']:.4f}")
        print(f"    CE gap r^2: {analysis['r2_ce_quality_pooled']:.4f} "
              f"[{analysis['r2_ce_quality_ci_lo']:.4f}, "
              f"{analysis['r2_ce_quality_ci_hi']:.4f}]")
        print(f"    Monotonic: {analysis['is_monotonic']}")
        print(f"    Verdict: {analysis['passes']} PASS, {analysis['kills']} KILL")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="3-seed validation")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--n-experts", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    if args.quick:
        run_bridge(n_seeds=3, n_experts=2, top_k=2)
    elif args.seeds:
        run_bridge(n_seeds=args.seeds, n_experts=args.n_experts, top_k=args.top_k)
    else:
        run_full_experiment()
