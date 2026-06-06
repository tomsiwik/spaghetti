#!/usr/bin/env python3
"""M2P with domain conditioning to prevent B-matrix mode collapse.

Based on m2p_distillation_toy rev1 (LoRA-corrected forward path).
Key change: M2PTransformer receives a learned domain embedding (nn.Embedding)
injected additively into memory tokens, providing explicit domain signal.

Theorem 3 (MATH.md): domain embeddings are linearly independent (Lemma 1),
making the B-matrix centroid state unstable. Each embedding receives
domain-private gradient signal, forcing per-domain B-matrix generation.

Kill criteria:
  K855: Median M2P quality >= 25% of SFT across all 5 domains
  K856: No domain has quality below -10% (no catastrophic collapse)
  K857: Grassmannian |cos| <= 1e-5 (structural guarantee — unaffected by conditioning)
"""

import gc
import json
import math
import os
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt
from mlx.utils import tree_flatten, tree_unflatten

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

SEED = 42
D_MODEL = 64
N_LAYERS = 4
N_HEADS = 4
BLOCK_SIZE = 32
VOCAB_SIZE = 95  # printable ASCII
LORA_RANK = 4
N_MEMORY = 32    # memory tokens (doubled in rev1 to accommodate fc1 B-matrices)
M2P_LAYERS = 2   # lightweight M2P
N_DOMAINS = 5

# Training config (identical to baseline for fair comparison)
SFT_STEPS = 300
M2P_PRETRAIN_STEPS = 500
M2P_LR = 1e-3
SFT_LR = 1e-3


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_,)): return bool(o)
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return super().default(o)

def log(m): print(m, flush=True)
def cleanup(*o):
    for x in o: del x
    gc.collect(); mx.clear_cache(); mx.reset_peak_memory()


# ── Toy GPT ──────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + self.eps)

class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h; self.hd = d//h
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
    def __call__(self, x):
        B,T,C = x.shape
        q = self.wq(x).reshape(B,T,self.h,self.hd).transpose(0,2,1,3)
        k = self.wk(x).reshape(B,T,self.h,self.hd).transpose(0,2,1,3)
        v = self.wv(x).reshape(B,T,self.h,self.hd).transpose(0,2,1,3)
        mask = mx.triu(mx.full((T,T), float("-inf")), k=1)
        a = mx.softmax(q @ k.transpose(0,1,3,2) * (self.hd**-0.5) + mask, axis=-1)
        return self.wo((a @ v).transpose(0,2,1,3).reshape(B,T,C))

class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, 4*d, bias=False)
        self.fc2 = nn.Linear(4*d, d, bias=False)
    def __call__(self, x): return self.fc2(nn.relu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n1 = RMSNorm(d); self.attn = Attn(d, h)
        self.n2 = RMSNorm(d); self.mlp = MLP(d)
    def __call__(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))

class ToyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.wpe = nn.Embedding(BLOCK_SIZE, D_MODEL)
        self.layers = [Block(D_MODEL, N_HEADS) for _ in range(N_LAYERS)]
        self.norm = RMSNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
    def __call__(self, tokens):
        B,T = tokens.shape
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        for l in self.layers: x = l(x)
        return self.head(self.norm(x))

    def get_hidden_states(self, tokens):
        """Extract hidden states from all layers (for M2P input)."""
        B,T = tokens.shape
        x = self.wte(tokens) + self.wpe(mx.arange(T))
        states = []
        for l in self.layers:
            x = l(x)
            states.append(x)
        return states  # list of (B, T, D)


# ── M2P Transformer with Domain Conditioning ──────────────────────────────
#
# Key change vs m2p_distillation_toy rev1:
#   - Added domain_embed = nn.Embedding(N_DOMAINS, D_MODEL)
#   - In __call__: memory = memory + domain_embed(domain_id)
#   - domain_id is now a required argument to __call__
#
# Theorem 3 (MATH.md): domain embeddings are linearly independent (Lemma 1),
# making B-matrix centroid state unstable. Each e_d receives domain-private
# gradient: ∂L_d/∂e_d is different per domain → embeddings diverge →
# M2P generates domain-specific B-matrices.
#
# Domain embedding parameters: N_DOMAINS × D_MODEL = 5 × 64 = 320 params
# (1.9% overhead over ~17K M2P body params — negligible per Theorem 3 part 3).

class M2PBlock(nn.Module):
    """One M2P block with alternating row/column attention."""
    def __init__(self, d, n_heads=4, is_column=True):
        super().__init__()
        self.attn = Attn(d, n_heads)
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.mlp = MLP(d)
        self.is_column = is_column

    def __call__(self, x):
        """x: (L, M, H)"""
        L, M, H = x.shape
        if self.is_column:
            x_t = x.transpose(1, 0, 2)  # (M, L, H)
            x_t = x_t + self.attn(self.norm1(x_t))
            x_t = x_t + self.mlp(self.norm2(x_t))
            return x_t.transpose(1, 0, 2)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x


class M2PTransformerMoE(nn.Module):
    """MoE Domain-Conditioned M2P Transformer."""
    def __init__(self, d, n_layers=2, n_heads=4, n_target_modules=5, n_experts=4):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Embedding(N_DOMAINS, n_experts)

        self.memory_embed = mx.random.normal(shape=(1, N_MEMORY, d)) * 0.02
        self.p_layer = mx.zeros((N_LAYERS, 1, d))   
        self.p_token = mx.zeros((1, N_MEMORY, d))    

        self.experts = []
        for _ in range(n_experts):
            blocks = []
            for i in range(n_layers):
                blocks.append(M2PBlock(d, n_heads, is_column=(i % 2 == 0)))
            self.experts.append(blocks)
            
        self.final_norm = RMSNorm(d)
        self.n_target_modules = n_target_modules

    def __call__(self, hidden_states_list, domain_id):
        L = len(hidden_states_list)
        memories = []
        for h in hidden_states_list:
            pooled = mx.mean(h[0], axis=0)
            mem = mx.broadcast_to(pooled[None, :], (N_MEMORY, D_MODEL))
            memories.append(mem)

        memory = mx.stack(memories, axis=0)

        memory = memory + mx.broadcast_to(self.memory_embed, memory.shape)
        memory = memory + self.p_layer + self.p_token

        # === MoE Domain Routing (Prototype fixed centroid trap) ===
        route_weights = nn.softmax(self.router(mx.array(domain_id)), axis=-1)
        
        expert_outputs = []
        for expert_blocks in self.experts:
            expert_mem = memory
            for block in expert_blocks:
                expert_mem = block(expert_mem)
            expert_outputs.append(expert_mem)
            
        # Weighted sum of expert memory states
        memory = sum(route_weights[i] * expert_outputs[i] for i in range(self.n_experts))

        memory = self.final_norm(memory)

        # Generate B-matrices from memory
        # Module output dimensions: [wq, wk, wv, wo] = d_model, [fc1] = 4*d_model
        module_out_dims = [D_MODEL, D_MODEL, D_MODEL, D_MODEL, 4*D_MODEL]

        all_B = []
        for li in range(L):
            flat = memory[li].reshape(-1)  # (M*H,)
            layer_B = {}
            offset = 0
            for mi in range(self.n_target_modules):
                d_out = module_out_dims[mi]
                n_params = LORA_RANK * d_out
                if offset + n_params > flat.shape[0]:
                    b = mx.zeros((LORA_RANK, d_out))
                else:
                    b = flat[offset:offset + n_params].reshape(LORA_RANK, d_out)
                layer_B[mi] = b
                offset += n_params
            all_B.append(layer_B)

        return all_B  # list of L dicts


# ── Grassmannian A-matrices (unchanged from baseline) ─────────────────────

def generate_grassmannian_A(n_domains, n_layers, n_modules, d, rank, seed=42):
    """Generate frozen orthogonal A-matrices via QR decomposition.

    Theorem 1 (m2p_distillation_toy MATH.md): A_i^T A_j = 0 for i ≠ j
    for any B_i, B_j — parameter-space interference = 0 by construction.

    Returns: dict of (domain, layer, module) → A matrix (d, rank)
    """
    rng = np.random.RandomState(seed)
    A_matrices = {}
    for li in range(n_layers):
        for mi in range(n_modules):
            total_rank = n_domains * rank
            assert total_rank <= d, f"Need {total_rank} orthogonal vectors but d={d}"
            random_mat = rng.randn(d, total_rank).astype(np.float32)
            Q, _ = np.linalg.qr(random_mat)
            for di in range(n_domains):
                start = di * rank
                A_matrices[(di, li, mi)] = mx.array(Q[:, start:start+rank])
    return A_matrices


# ── LoRA with Grassmannian A + M2P B ────────────────────────────────────

class GrassmannianLoRA(nn.Module):
    """LoRA with frozen Grassmannian A and externally-set B."""
    def __init__(self, base, A_frozen, scale=2.0):
        super().__init__()
        self.base = base
        self.A = A_frozen  # (d, rank), frozen
        self.B = mx.zeros((LORA_RANK, base.weight.shape[0]))  # (rank, out)
        self.scale = scale
        self.base.freeze()
        self.freeze(keys=["A"], strict=False)

    def __call__(self, x):
        return self.base(x) + (x @ self.A) @ self.B * self.scale


# ── Data generators (unchanged from baseline) ─────────────────────────────

def gen_data(domain_id, n, rng):
    chars = "abcdefgh"
    data = []
    for _ in range(n):
        if domain_id == 0:  # arithmetic
            a, b = rng.randint(0,50), rng.randint(0,50)
            data.append(f"{a}+{b}={a+b}")
        elif domain_id == 1:  # reverse
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2,5)))
            data.append(f"{s}>{''.join(reversed(s))}")
        elif domain_id == 2:  # repeat
            p = "".join(rng.choice(list(chars)) for _ in range(rng.randint(1,3)))
            r = rng.randint(2,4)
            data.append(f"{p}*{r}={p*r}")
        elif domain_id == 3:  # sort
            s = "".join(rng.choice(list(chars)) for _ in range(rng.randint(2,5)))
            data.append(f"{s}>{''.join(sorted(s))}")
        elif domain_id == 4:  # parity
            bits = "".join(str(rng.randint(0,2)) for _ in range(rng.randint(2,6)))
            data.append(f"{bits}>{'even' if bits.count('1')%2==0 else 'odd'}")
    return data

DOMAIN_NAMES = ["arithmetic", "reverse", "repeat", "sort", "parity"]
TARGET_MODULES = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.fc1"]

def encode(text):
    return [max(0, min(ord(c) - 32, VOCAB_SIZE-1)) for c in text]

def make_batches(texts):
    return [mx.array(encode(t)[:BLOCK_SIZE]) for t in texts if len(t) >= 4]


# ── Phase functions (CODING_GUIDELINES: each phase self-contained) ────────

def phase_pretrain_base(domain_data):
    """Pre-train base GPT on all domains. Returns frozen base model + per-domain losses."""
    log("\n=== Phase 1: Pre-train Base Model ===")
    base = ToyGPT()
    mx.eval(base.parameters())
    base_optimizer = opt.Adam(learning_rate=3e-4)
    all_train = []
    for name in DOMAIN_NAMES:
        all_train.extend(domain_data[name]["train"])

    loss_fn = lambda model, tokens: nn.losses.cross_entropy(
        model(tokens[None, :])[:, :-1], tokens[None, 1:], reduction="mean")
    base_loss_grad = nn.value_and_grad(base, loss_fn)

    gc.disable()
    for step in range(1000):
        tokens = all_train[step % len(all_train)]
        loss, grads = base_loss_grad(base, tokens)
        base_optimizer.update(base, grads)
        mx.eval(base.parameters(), base_optimizer.state, loss)
        if (step+1) % 250 == 0:
            log(f"  Step {step+1}: loss={loss.item():.4f}")
    gc.enable()
    base.freeze()

    def eval_loss(model, batches):
        total = 0; n = 0
        for tokens in batches[:50]:
            l = nn.losses.cross_entropy(model(tokens[None, :])[:, :-1], tokens[None, 1:], reduction="mean")
            mx.eval(l); total += l.item(); n += 1
        return total / max(n, 1)

    base_losses = {name: round(eval_loss(base, domain_data[name]["val"]), 4) for name in DOMAIN_NAMES}
    log(f"  Base losses: {base_losses}")

    loss_ratio = max(base_losses.values()) / min(base_losses.values())
    log(f"  Loss ratio (max/min): {loss_ratio:.2f}x (baseline had 4.9x)")

    cleanup(base_optimizer)
    return base, base_losses


def phase_grassmannian(A_matrices, cos_values_list):
    """Verify Grassmannian A-matrix orthogonality (K857)."""
    log("\n=== Phase 2: Grassmannian A-matrices ===")
    for li in range(N_LAYERS):
        for mi in range(len(TARGET_MODULES)):
            for di in range(N_DOMAINS):
                for dj in range(di+1, N_DOMAINS):
                    ai = A_matrices[(di, li, mi)]
                    aj = A_matrices[(dj, li, mi)]
                    cos = mx.abs(mx.sum(ai * aj) / (mx.linalg.norm(ai.reshape(-1)) * mx.linalg.norm(aj.reshape(-1)) + 1e-8)).item()
                    cos_values_list.append(cos)
    log(f"  Grassmannian |cos|: mean={np.mean(cos_values_list):.6f}, max={np.max(cos_values_list):.6f}")


def phase_sft(di, name, base, A_matrices, domain_data, base_losses, sft_losses):
    """Train one SFT adapter. Returns sft_loss for the domain."""
    loss_fn = lambda model, tokens: nn.losses.cross_entropy(
        model(tokens[None, :])[:, :-1], tokens[None, 1:], reduction="mean")

    sft_model = ToyGPT()
    sft_model.update(tree_unflatten(list(zip(
        [n for n, _ in tree_flatten(base.parameters())],
        [p for _, p in tree_flatten(base.parameters())]
    ))))
    mx.eval(sft_model.parameters())

    for li, layer in enumerate(sft_model.layers):
        updates = []
        for mi, mname in enumerate(TARGET_MODULES):
            parts = mname.split(".")
            m = layer
            for p in parts: m = getattr(m, p, None)
            if m is None or not isinstance(m, nn.Linear): continue
            lora = GrassmannianLoRA(m, A_matrices[(di, li, mi)], scale=2.0)
            updates.append((mname, lora))
        if updates:
            layer.update_modules(tree_unflatten(updates))

    sft_model.freeze()
    sft_model.unfreeze(keys=["B"], strict=False)
    sft_opt = opt.Adam(learning_rate=SFT_LR)
    sft_lg = nn.value_and_grad(sft_model, loss_fn)

    gc.disable()
    for step in range(SFT_STEPS):
        tokens = domain_data[name]["train"][step % len(domain_data[name]["train"])]
        loss, grads = sft_lg(sft_model, tokens)
        sft_opt.update(sft_model, grads)
        mx.eval(sft_model.parameters(), sft_opt.state, loss)
    gc.enable()

    def eval_loss(model, batches):
        total = 0; n = 0
        for tokens in batches[:50]:
            l = nn.losses.cross_entropy(model(tokens[None, :])[:, :-1], tokens[None, 1:], reduction="mean")
            mx.eval(l); total += l.item(); n += 1
        return total / max(n, 1)

    sft_loss = round(eval_loss(sft_model, domain_data[name]["val"]), 4)
    log(f"    SFT loss: {sft_loss} (base: {base_losses[name]})")
    sft_losses[name] = sft_loss
    cleanup(sft_model, sft_opt)


def phase_train_m2p(base, A_matrices, domain_data, base_losses):
    """Train domain-conditioned M2P. Returns trained m2p model."""
    log("\n=== Phase 4: Train Domain-Conditioned M2P ===")
    m2p = M2PTransformerMoE(D_MODEL, n_layers=M2P_LAYERS, n_heads=N_HEADS,
                                    n_target_modules=len(TARGET_MODULES))
    mx.eval(m2p.parameters())

    m2p_params = sum(p.size for _, p in tree_flatten(m2p.parameters()))
    domain_embed_params = N_DOMAINS * D_MODEL
    log(f"  M2P parameters: {m2p_params:,} (includes {domain_embed_params} domain embedding params)")
    log(f"  Domain embedding overhead: {domain_embed_params/m2p_params:.1%} of M2P")

    m2p_opt = opt.Adam(learning_rate=M2P_LR)

    def m2p_loss(m2p_model, base_model, tokens, domain_id):
        """M2P loss: generate domain-conditioned adapter, apply, measure task loss.

        Matches SFT GrassmannianLoRA forward path exactly (rev1 corrected path):
          - wq/wk/wv corrections applied INSIDE attention
          - wo correction applied to attention context
          - fc1 correction applied INSIDE MLP (before relu)
        """
        hidden_states = base_model.get_hidden_states(tokens[None, :])

        # Domain-conditioned B-matrix generation (Theorem 3)
        all_B = m2p_model(hidden_states, domain_id)

        scale = 2.0
        x = base_model.wte(tokens[None, :]) + base_model.wpe(mx.arange(tokens.shape[0]))
        for li, layer in enumerate(base_model.layers):
            x_norm = layer.n1(x)
            attn = layer.attn
            Bs, T, C = x_norm.shape
            h, hd = attn.h, attn.hd

            q = attn.wq(x_norm) + scale * (x_norm @ A_matrices[(domain_id, li, 0)]) @ all_B[li][0]
            k = attn.wk(x_norm) + scale * (x_norm @ A_matrices[(domain_id, li, 1)]) @ all_B[li][1]
            v = attn.wv(x_norm) + scale * (x_norm @ A_matrices[(domain_id, li, 2)]) @ all_B[li][2]

            q = q.reshape(Bs, T, h, hd).transpose(0, 2, 1, 3)
            k = k.reshape(Bs, T, h, hd).transpose(0, 2, 1, 3)
            v = v.reshape(Bs, T, h, hd).transpose(0, 2, 1, 3)
            mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
            a_mat = mx.softmax(q @ k.transpose(0, 1, 3, 2) * (hd**-0.5) + mask, axis=-1)
            attn_ctx = (a_mat @ v).transpose(0, 2, 1, 3).reshape(Bs, T, C)

            attn_out = attn.wo(attn_ctx) + scale * (attn_ctx @ A_matrices[(domain_id, li, 3)]) @ all_B[li][3]
            x = x + attn_out

            x_norm2 = layer.n2(x)
            fc1_out = layer.mlp.fc1(x_norm2) + scale * (x_norm2 @ A_matrices[(domain_id, li, 4)]) @ all_B[li][4]
            mlp_out = layer.mlp.fc2(nn.relu(fc1_out))
            x = x + mlp_out

        logits = base_model.head(base_model.norm(x))
        return nn.losses.cross_entropy(logits[:, :-1], tokens[None, 1:], reduction="mean")

    def m2p_train_loss(m2p_model, base_model, tokens, domain_id):
        raw_loss = m2p_loss(m2p_model, base_model, tokens, domain_id)
        # M2P Loss Normalization prevents higher-loss domains from dominating
        name = DOMAIN_NAMES[domain_id]
        scale = base_losses[name]
        return raw_loss / scale

    m2p_lg = nn.value_and_grad(m2p, m2p_train_loss)

    gc.disable()
    for step in range(M2P_PRETRAIN_STEPS):
        di = step % N_DOMAINS
        name = DOMAIN_NAMES[di]
        tokens = domain_data[name]["train"][step % len(domain_data[name]["train"])]
        loss, grads = m2p_lg(m2p, base, tokens, di)
        m2p_opt.update(m2p, grads)
        mx.eval(m2p.parameters(), m2p_opt.state, loss)
        if (step+1) % 100 == 0:
            log(f"  Step {step+1}: loss={loss.item():.4f}")
    gc.enable()

    cleanup(m2p_opt)
    return m2p, m2p_loss, m2p_params


def phase_evaluate_m2p(m2p, m2p_loss_fn, base, A_matrices, domain_data, base_losses, sft_losses):
    """Evaluate M2P adapter quality per domain. Returns m2p_losses dict."""
    log("\n=== Phase 5: M2P Adapter Quality ===")
    m2p_losses = {}
    for di, name in enumerate(DOMAIN_NAMES):
        total = 0; n = 0
        for tokens in domain_data[name]["val"][:30]:
            loss = m2p_loss_fn(m2p, base, tokens, di)
            mx.eval(loss)
            total += loss.item(); n += 1
        m2p_losses[name] = round(total / max(n, 1), 4)

        quality_ratio = (base_losses[name] - m2p_losses[name]) / (base_losses[name] - sft_losses[name]) \
            if (base_losses[name] - sft_losses[name]) > 0.01 else 0
        log(f"  {name}: M2P={m2p_losses[name]} SFT={sft_losses[name]} base={base_losses[name]} quality={quality_ratio:.1%}")

    return m2p_losses


def phase_composition_check(m2p, base, domain_data):
    """Check B-matrix diversity and Grassmannian interference."""
    log("\n=== Phase 6: Composition Check ===")
    context_Bs = {}
    for di, name in enumerate(DOMAIN_NAMES):
        context_tokens = domain_data[name]["train"][0]
        hidden_states = base.get_hidden_states(context_tokens[None, :])
        all_B = m2p(hidden_states, di)
        flat = mx.concatenate([b.reshape(-1) for layer_B in all_B for b in layer_B.values()])
        mx.eval(flat)
        context_Bs[name] = flat

    # Pairwise B cosine (diagnostic: should be substantially less than 0.9956)
    delta_cos = []
    for i in range(N_DOMAINS):
        for j in range(i+1, N_DOMAINS):
            cos = mx.abs(mx.sum(context_Bs[DOMAIN_NAMES[i]] * context_Bs[DOMAIN_NAMES[j]]) /
                        (mx.linalg.norm(context_Bs[DOMAIN_NAMES[i]]) * mx.linalg.norm(context_Bs[DOMAIN_NAMES[j]]) + 1e-8)).item()
            delta_cos.append(cos)
    mean_cos = float(np.mean(delta_cos))
    log(f"  M2P adapter B-matrix |cos|: mean={mean_cos:.4f} (baseline was 0.9956)")
    log(f"  Theorem 3 prediction: substantially below 0.9956 (centroid destabilized)")
    log(f"  (Grassmannian A guarantees parameter-space interference=0 regardless of B content)")
    return mean_cos, delta_cos


def phase_router_check(m2p):
    """K860 (audit-2026-04-17 strict KC): test whether router specialises by domain.

    For each domain d, compute route_weights[d, :] = softmax(router(d)).
    Report:
      - max_route_weight per domain (m_d)
      - mean across domains (m̄) — the K860 test statistic
      - argmax expert per domain (degenerate-collapse diagnostic)
      - entropy per domain (uniform = ln(n_experts) ≈ 1.386)

    K860 PASS iff m̄ ≥ 0.50 (router specialises);  FAIL iff m̄ ≤ 0.50.
    Pre-registered prediction (MATH.md §K.2): FAIL — router falls back to
    uniform allocation (m̄ ≈ 0.25 ± 0.10).
    """
    log("\n=== Phase 7: Router Uniform-Fallback Check (K860) ===")
    n_experts = m2p.n_experts
    uniform = 1.0 / n_experts
    per_domain_route = {}
    per_domain_max = []
    per_domain_argmax = []
    per_domain_entropy = []
    for di, name in enumerate(DOMAIN_NAMES):
        logits = m2p.router(mx.array(di))
        weights = nn.softmax(logits, axis=-1)
        mx.eval(weights)
        w = np.array(weights).astype(np.float64)
        m_d = float(np.max(w))
        am_d = int(np.argmax(w))
        # Stable entropy: -Σ p log p with safe clip
        ent_d = float(-np.sum(w * np.log(np.clip(w, 1e-12, 1.0))))
        per_domain_route[name] = [round(float(x), 4) for x in w.tolist()]
        per_domain_max.append(m_d)
        per_domain_argmax.append(am_d)
        per_domain_entropy.append(ent_d)
        log(f"  {name}: route={per_domain_route[name]} max={m_d:.4f} argmax=expert_{am_d} H={ent_d:.4f}")
    mean_max = float(np.mean(per_domain_max))
    mean_entropy = float(np.mean(per_domain_entropy))
    n_unique_argmax = int(len(set(per_domain_argmax)))
    max_entropy = float(np.log(n_experts))
    log(f"  Summary: m̄={mean_max:.4f} (uniform={uniform:.4f})  H̄={mean_entropy:.4f}/{max_entropy:.4f}  unique_argmax_experts={n_unique_argmax}/{n_experts}")
    return {
        "n_experts": n_experts,
        "uniform_weight": round(uniform, 4),
        "max_entropy": round(max_entropy, 4),
        "per_domain_route_weights": per_domain_route,
        "per_domain_max": [round(x, 4) for x in per_domain_max],
        "per_domain_argmax": per_domain_argmax,
        "per_domain_entropy": [round(x, 4) for x in per_domain_entropy],
        "mean_max_route_weight": round(mean_max, 4),
        "mean_entropy": round(mean_entropy, 4),
        "entropy_uniform_ratio": round(mean_entropy / max_entropy, 4),
        "n_unique_argmax_experts": n_unique_argmax,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log("M2P Domain-Conditioned: Prevents B-matrix mode collapse via learned domain embeddings")
    log("Theorem 3 (MATH.md): domain embeddings are linearly independent → centroid state unstable")
    log("=" * 70)
    mx.random.seed(SEED)
    rng = np.random.RandomState(SEED)

    # Generate domain data
    domain_data = {}
    for di, name in enumerate(DOMAIN_NAMES):
        texts = gen_data(di, 500, rng)
        domain_data[name] = {
            "train": make_batches(texts[:400]),
            "val": make_batches(texts[400:]),
        }
        log(f"  {name}: {len(domain_data[name]['train'])} train, {len(domain_data[name]['val'])} val")

    # Phase 1: Pre-train base model
    base, base_losses = phase_pretrain_base(domain_data)

    # Phase 2: Generate and verify Grassmannian A-matrices (K857)
    A_matrices = generate_grassmannian_A(N_DOMAINS, N_LAYERS, len(TARGET_MODULES), D_MODEL, LORA_RANK, SEED)
    cos_values = []
    phase_grassmannian(A_matrices, cos_values)

    # Phase 3: SFT baselines (one per domain)
    log("\n=== Phase 3: SFT Adapter Baselines ===")
    sft_losses = {}
    for di, name in enumerate(DOMAIN_NAMES):
        log(f"  Training SFT adapter: {name}")
        phase_sft(di, name, base, A_matrices, domain_data, base_losses, sft_losses)

    # Phase 4: Train domain-conditioned M2P
    m2p, m2p_loss_fn, m2p_params = phase_train_m2p(base, A_matrices, domain_data, base_losses)

    # Phase 5: Evaluate M2P quality
    m2p_losses = phase_evaluate_m2p(m2p, m2p_loss_fn, base, A_matrices, domain_data, base_losses, sft_losses)

    # Phase 6: Composition check (B-matrix diversity)
    mean_b_cos, b_cos_values = phase_composition_check(m2p, base, domain_data)

    # Phase 7: Router uniform-fallback check (K860 — DB-tracked KC)
    router_stats = phase_router_check(m2p)

    # Compute quality ratios
    quality_ratios = [
        (base_losses[n] - m2p_losses[n]) / (base_losses[n] - sft_losses[n])
        if (base_losses[n] - sft_losses[n]) > 0.01 else 0
        for n in DOMAIN_NAMES
    ]
    mean_quality = float(np.mean(quality_ratios))
    median_quality = float(np.median(quality_ratios))
    min_quality = float(np.min(quality_ratios))
    per_domain_quality = {n: round(q, 3) for n, q in zip(DOMAIN_NAMES, quality_ratios)}

    # Kill criteria assessment
    # K860 (audit-2026-04-17 strict KC, DB-tracked): router specialises by domain
    #   PASS iff mean_max_route_weight >= 0.50 (router puts ≥50% mass on one expert per domain on average)
    #   FAIL iff mean_max_route_weight <= 0.50 (uniform-fallback collapse)
    k860_stat = router_stats["mean_max_route_weight"]
    k860 = bool(k860_stat >= 0.50)

    # Auxiliary diagnostics (not gating verdict — DB tracks K860 only)
    k855 = bool(median_quality >= 0.25)
    k856 = bool(min_quality >= -0.10)
    k857 = bool(float(np.max(cos_values)) <= 1e-5)

    results = {
        "experiment": "m2p_moe_routing",
        "total_time_s": round(time.time() - t0, 1),
        "is_smoke": False,
        "ran": True,
        "base_losses": base_losses,
        "sft_losses": sft_losses,
        "m2p_losses": m2p_losses,
        "per_domain_quality": per_domain_quality,
        "mean_quality_ratio": round(mean_quality, 3),
        "median_quality_ratio": round(median_quality, 3),
        "min_quality_ratio": round(min_quality, 3),
        "grassmannian_cos": round(float(np.mean(cos_values)), 7),
        "grassmannian_cos_max": round(float(np.max(cos_values)), 7),
        "m2p_b_cos_mean": round(mean_b_cos, 4),
        "m2p_b_cos_values": [round(c, 4) for c in b_cos_values],
        "m2p_params": m2p_params,
        "baseline_b_cos": 0.9956,  # From Finding #341 for comparison
        "router_stats": router_stats,
        "kill_criteria": {
            "K860": {
                "pass": k860,
                "mean_max_route_weight": k860_stat,
                "threshold": 0.50,
                "uniform_baseline": router_stats["uniform_weight"],
                "n_unique_argmax_experts": router_stats["n_unique_argmax_experts"],
                "mean_entropy": router_stats["mean_entropy"],
                "max_entropy": router_stats["max_entropy"],
                "description": "Router specialises by domain (DB-tracked strict KC, audit-2026-04-17)"
            },
            "K855_aux": {
                "pass": k855,
                "median_quality": round(median_quality, 3),
                "threshold": 0.25,
                "description": "AUX: Median M2P quality >= 25% of SFT (not verdict-gating)"
            },
            "K856_aux": {
                "pass": k856,
                "min_quality": round(min_quality, 3),
                "threshold": -0.10,
                "description": "AUX: No domain below -10% (not verdict-gating)",
                "worst_domain": DOMAIN_NAMES[int(np.argmin(quality_ratios))]
            },
            "K857_aux": {
                "pass": k857,
                "grassmannian_cos_max": round(float(np.max(cos_values)), 7),
                "threshold": 1e-5,
                "description": "AUX: Grassmannian A structural orthogonality (not verdict-gating)"
            },
        },
        "all_pass": k860,  # DB tracks K860 only — verdict gated by K860 alone (MATH.md §K.4)
        "verdict": "ALL_PASS" if k860 else "KILLED",
        "theorem3_check": {
            "b_cos_reduction": round(0.9956 - mean_b_cos, 4),
            "centroid_destabilized": bool(mean_b_cos < 0.90),
            "description": "Theorem 3 predicts B-matrix |cos| << 0.9956 after domain conditioning"
        }
    }

    log(f"\n{'='*70}")
    log(f"K860 (DB-tracked): m̄_route={k860_stat:.4f}  {'PASS' if k860 else 'FAIL'} (threshold ≥ 0.50; uniform = {router_stats['uniform_weight']:.4f})")
    log(f"M2P quality: median={median_quality:.1%} mean={mean_quality:.1%} min={min_quality:.1%} of SFT")
    log(f"Per-domain: {per_domain_quality}")
    log(f"B-matrix diversity: |cos|={mean_b_cos:.4f} (was 0.9956 in Finding #341)")
    log(f"Centroid destabilized: {results['theorem3_check']['centroid_destabilized']}")
    for k, v in results["kill_criteria"].items():
        log(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} — {v}")
    log(f"\nVerdict: {results['verdict']} in {results['total_time_s']}s")

    RESULTS_FILE.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))

if __name__ == "__main__":
    main()
