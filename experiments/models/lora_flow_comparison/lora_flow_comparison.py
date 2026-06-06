#!/usr/bin/env python3
"""
SOLE vs LoRA-Flow: Empirical micro-scale comparison.

Compares four LoRA composition methods on a synthetic MLP char-level model:

  1. SOLE: Unit-weight addition of expert deltas. W_s + sum(dW_i). Zero cost.
  2. CAT (LoRA Soups): Learned per-layer static scalar weights.
  3. LoRA-Flow: Dynamic per-layer per-token fusion gate.
     w^l = softmax(W_gate^l @ x_t^l) + b^l  (Wang et al., 2024)
  4. X-LoRA: MLP gating on hidden states for per-layer mixing. (Buehler, 2024)

Architecture: 4-layer MLP, d=64, d_ff=256, rank=8. Pure numpy, CPU-only.
All gate training uses SPSA (2 forward passes per layer per step).

Kill criteria:
  K1: LoRA-Flow achieves >10% quality gain over SOLE at comparable overhead
  K2: LoRA-Flow's per-layer weight learning is feasible at N>10 experts
"""

import json
import time
from pathlib import Path
import numpy as np

# =============================================================================
# Constants
# =============================================================================
VOCAB_SIZE = 32
D_MODEL = 64
D_FF = 256
N_LAYERS = 4
LORA_RANK = 8
LORA_ALPHA = 8
SEEDS = [42, 137, 2024]

BASE_LR = 0.05
EXPERT_LR = 0.02
BASE_STEPS = 500
EXPERT_STEPS = 300
BATCH_SIZE = 64
SEQ_LEN = 16
EVAL_SAMPLES = 256

# Gate training: SPSA steps (each = 2*L forward passes)
GATE_STEPS = 50
GATE_LR = 1e-3
GATE_TRAIN_N = 200

# CAT: finite-difference steps on scalar weights
CAT_STEPS = 50
CAT_LR = 0.1

# 12 domains in 4 clusters (to test K2 at N>10)
DOMAIN_CLUSTERS = {
    'code':      ['code_a', 'code_b', 'code_c'],
    'reasoning': ['reason_a', 'reason_b', 'reason_c'],
    'knowledge': ['know_a', 'know_b', 'know_c'],
    'creative':  ['creat_a', 'creat_b', 'creat_c'],
}
DOMAINS = {}
for cluster, names in DOMAIN_CLUSTERS.items():
    for name in names:
        DOMAINS[name] = {'cluster': cluster}

# =============================================================================
# Data
# =============================================================================
def make_transition_matrix(cluster, domain_idx, rng):
    V = VOCAB_SIZE
    if cluster == 'code':
        base = rng.dirichlet(np.ones(V) * 0.3, size=V); base[:, :8] *= 3.0
    elif cluster == 'reasoning':
        base = rng.dirichlet(np.ones(V) * 0.5, size=V); base[:, 8:16] *= 2.5
    elif cluster == 'knowledge':
        base = rng.dirichlet(np.ones(V) * 0.4, size=V); base[:, 16:24] *= 2.5
    else:
        base = rng.dirichlet(np.ones(V) * 0.6, size=V); base[:, 24:] *= 2.0
    noise = rng.dirichlet(np.ones(V) * 1.0, size=V)
    alpha = 0.1 + 0.05 * domain_idx
    mixed = (1 - alpha) * base + alpha * noise
    return mixed / mixed.sum(axis=1, keepdims=True)

def generate_data(tm, n_samples, seq_len, rng):
    data = np.zeros((n_samples, seq_len), dtype=np.int32)
    data[:, 0] = rng.integers(0, VOCAB_SIZE, size=n_samples)
    for t in range(1, seq_len):
        for i in range(n_samples):
            data[i, t] = rng.choice(VOCAB_SIZE, p=tm[data[i, t - 1]])
    return data

# =============================================================================
# Model
# =============================================================================
def silu(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return x * s

def silu_deriv(x):
    s = 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    return s + x * s * (1 - s)

def softmax(logits):
    m = logits.max(axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=-1, keepdims=True)


class MLP:
    def __init__(self, rng):
        sc = lambda fi, fo: np.sqrt(2.0 / (fi + fo))
        self.embed = rng.standard_normal((VOCAB_SIZE, D_MODEL)).astype(np.float32) * 0.02
        self.W_out = rng.standard_normal((D_MODEL, VOCAB_SIZE)).astype(np.float32) * sc(D_MODEL, VOCAB_SIZE)
        self.b_out = np.zeros(VOCAB_SIZE, dtype=np.float32)
        self.W_up, self.b_up, self.W_down, self.b_down = [], [], [], []
        for _ in range(N_LAYERS):
            self.W_up.append(rng.standard_normal((D_MODEL, D_FF)).astype(np.float32) * sc(D_MODEL, D_FF))
            self.b_up.append(np.zeros(D_FF, dtype=np.float32))
            self.W_down.append(rng.standard_normal((D_FF, D_MODEL)).astype(np.float32) * sc(D_FF, D_MODEL))
            self.b_down.append(np.zeros(D_MODEL, dtype=np.float32))

    def forward(self, x_ids, lora_deltas=None):
        x = self.embed[x_ids]
        self._hs = [x.copy()]
        h = x
        for l in range(N_LAYERS):
            Wu = self.W_up[l] + (lora_deltas[l][0] if lora_deltas else 0)
            Wd = self.W_down[l] + (lora_deltas[l][1] if lora_deltas else 0)
            pre = h @ Wu + self.b_up[l]
            act = silu(pre)
            h = h + act @ Wd + self.b_down[l]
            self._hs.append(h.copy())
        return h @ self.W_out + self.b_out

    def forward_dynamic(self, x_ids, expert_deltas_list, weight_fn):
        """Forward with per-layer per-sample dynamic expert weighting."""
        x = self.embed[x_ids]
        h = x
        B = x_ids.shape[0]
        N = len(expert_deltas_list)
        for l in range(N_LAYERS):
            w = weight_fn(h, l)  # (B, N)
            pre = h @ self.W_up[l] + self.b_up[l]
            # Add weighted expert contributions to pre-activation
            for i in range(N):
                pre += (h @ expert_deltas_list[i][l][0]) * w[:, i:i+1]
            act = silu(pre)
            h_new = act @ self.W_down[l] + self.b_down[l]
            for i in range(N):
                h_new += (act @ expert_deltas_list[i][l][1]) * w[:, i:i+1]
            h = h + h_new
        return h @ self.W_out + self.b_out

    def loss_and_grads(self, x_ids, targets, lora_deltas=None, grad_lora=False, grad_base=True):
        B = x_ids.shape[0]
        logits = self.forward(x_ids, lora_deltas)
        probs = softmax(logits)
        loss = -np.log(np.clip(probs[np.arange(B), targets], 1e-10, 1.0)).mean()

        dlogits = probs.copy()
        dlogits[np.arange(B), targets] -= 1.0
        dlogits /= B

        grads = {}
        if grad_base:
            grads['W_out'] = self._hs[N_LAYERS].T @ dlogits
            grads['b_out'] = dlogits.sum(0)

        dh = dlogits @ self.W_out.T
        lora_grads = []
        for l in range(N_LAYERS - 1, -1, -1):
            Wu = self.W_up[l] + (lora_deltas[l][0] if lora_deltas else 0)
            Wd = self.W_down[l] + (lora_deltas[l][1] if lora_deltas else 0)
            h_in = self._hs[l]
            pre = h_in @ Wu + self.b_up[l]
            act = silu(pre)
            dact = dh @ Wd.T
            dpre = dact * silu_deriv(pre)
            if grad_base:
                grads[f'W_up_{l}'] = h_in.T @ dpre
                grads[f'b_up_{l}'] = dpre.sum(0)
                grads[f'W_down_{l}'] = act.T @ dh
                grads[f'b_down_{l}'] = dh.sum(0)
            if grad_lora and lora_deltas is not None:
                lora_grads.append((h_in.T @ dpre, act.T @ dh))
            else:
                lora_grads.append(None)
            dh = dh + dpre @ Wu.T

        if grad_base:
            grads['embed'] = np.zeros_like(self.embed)
            np.add.at(grads['embed'], x_ids, dh)
        lora_grads.reverse()
        return loss, grads, lora_grads

    def ntp_loss(self, data, lora_deltas=None):
        B, T = data.shape[0], data.shape[1] - 1
        total = 0.0
        for t in range(T):
            logits = self.forward(data[:, t], lora_deltas)
            probs = softmax(logits)
            total += -np.log(np.clip(probs[np.arange(B), data[:, t + 1]], 1e-10, 1.0)).mean()
        return total / T

    def ntp_loss_dynamic(self, data, expert_deltas_list, weight_fn):
        B, T = data.shape[0], data.shape[1] - 1
        total = 0.0
        for t in range(T):
            logits = self.forward_dynamic(data[:, t], expert_deltas_list, weight_fn)
            probs = softmax(logits)
            total += -np.log(np.clip(probs[np.arange(B), data[:, t + 1]], 1e-10, 1.0)).mean()
        return total / T


class LoRA:
    def __init__(self, rng):
        self.A_up, self.B_up, self.A_down, self.B_down = [], [], [], []
        for _ in range(N_LAYERS):
            self.A_up.append(rng.standard_normal((D_MODEL, LORA_RANK)).astype(np.float32) * 0.01)
            self.B_up.append(rng.standard_normal((LORA_RANK, D_FF)).astype(np.float32) * 0.001)
            self.A_down.append(rng.standard_normal((D_FF, LORA_RANK)).astype(np.float32) * 0.01)
            self.B_down.append(rng.standard_normal((LORA_RANK, D_MODEL)).astype(np.float32) * 0.001)

    def get_deltas(self, scale=LORA_ALPHA / LORA_RANK):
        return [(scale * (self.A_up[l] @ self.B_up[l]),
                 scale * (self.A_down[l] @ self.B_down[l])) for l in range(N_LAYERS)]

    def flatten(self):
        parts = []
        for dW_up, dW_down in self.get_deltas():
            parts.extend([dW_up.ravel(), dW_down.ravel()])
        return np.concatenate(parts)

    def train_step(self, model, x_ids, targets, lr):
        deltas = self.get_deltas()
        loss, _, lg = model.loss_and_grads(x_ids, targets, deltas, grad_lora=True, grad_base=False)
        s = LORA_ALPHA / LORA_RANK
        for l in range(N_LAYERS):
            if lg[l] is None: continue
            gu, gd = lg[l]
            self.A_up[l] -= lr * s * (gu @ self.B_up[l].T)
            self.B_up[l] -= lr * s * (self.A_up[l].T @ gu)
            self.A_down[l] -= lr * s * (gd @ self.B_down[l].T)
            self.B_down[l] -= lr * s * (self.A_down[l].T @ gd)
        return loss


# =============================================================================
# Composition Methods
# =============================================================================
def compose_sole(loras):
    deltas = []
    all_d = [l.get_deltas() for l in loras]
    for l in range(N_LAYERS):
        deltas.append((sum(d[l][0] for d in all_d), sum(d[l][1] for d in all_d)))
    return deltas

def compose_avg(loras):
    k = len(loras)
    all_d = [l.get_deltas() for l in loras]
    deltas = []
    for l in range(N_LAYERS):
        deltas.append((sum(d[l][0] for d in all_d) / k, sum(d[l][1] for d in all_d) / k))
    return deltas

def compose_cat(model, loras, data, rng, n_steps=CAT_STEPS, lr=CAT_LR):
    """CAT: per-layer per-expert static scalar weights, optimized via finite diff."""
    k = len(loras)
    w = [np.ones(k, dtype=np.float32) for _ in range(N_LAYERS)]
    ed = [lora.get_deltas() for lora in loras]
    n = len(data)
    eps = 1e-3

    def mk():
        return [(sum(w[l][i] * ed[i][l][p] for i in range(k)) for p in (0, 1)) for l in range(N_LAYERS)]

    def mk_list():
        r = []
        for l in range(N_LAYERS):
            u = sum(w[l][i] * ed[i][l][0] for i in range(k))
            d = sum(w[l][i] * ed[i][l][1] for i in range(k))
            r.append((u, d))
        return r

    for step in range(n_steps):
        idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
        bx, by = data[idx, 0], data[idx, 1]
        for l in range(N_LAYERS):
            for i in range(k):
                w[l][i] += eps
                dd = mk_list()
                lp = model.forward(bx, dd)
                pp = softmax(lp)
                loss_p = -np.log(np.clip(pp[np.arange(len(by)), by], 1e-10, 1)).mean()
                w[l][i] -= 2*eps
                dd = mk_list()
                lm = model.forward(bx, dd)
                pm = softmax(lm)
                loss_m = -np.log(np.clip(pm[np.arange(len(by)), by], 1e-10, 1)).mean()
                w[l][i] += eps
                w[l][i] -= lr * (loss_p - loss_m) / (2*eps)

    final = mk_list()
    return final, {'weights': [ww.tolist() for ww in w]}


def _train_gate_spsa(gate, model, expert_deltas_list, train_data, rng,
                     lr=GATE_LR, n_steps=GATE_STEPS):
    """Train any gate with SPSA. Returns loss curve.

    gate must have: .get_weights(h, l), and lists of param arrays to perturb.
    """
    n = len(train_data)
    eps = 1e-3
    losses = []

    # Collect all parameter arrays from gate
    param_arrays = gate.get_param_arrays()

    for step in range(n_steps):
        idx = rng.integers(0, n, size=min(BATCH_SIZE, n))
        bx, by = train_data[idx, 0], train_data[idx, 1]

        # SPSA: perturb ALL params simultaneously (one +, one -)
        deltas = [rng.choice([-1, 1], size=p.shape).astype(np.float32) for p in param_arrays]

        # Plus perturbation
        for p, d in zip(param_arrays, deltas):
            p += eps * d
        wfn = lambda h, l: gate.get_weights(h, l)
        lp = model.forward_dynamic(bx, expert_deltas_list, wfn)
        pp = softmax(lp)
        loss_p = -np.log(np.clip(pp[np.arange(len(by)), by], 1e-10, 1)).mean()

        # Minus perturbation
        for p, d in zip(param_arrays, deltas):
            p -= 2 * eps * d
        lm = model.forward_dynamic(bx, expert_deltas_list, wfn)
        pm = softmax(lm)
        loss_m = -np.log(np.clip(pm[np.arange(len(by)), by], 1e-10, 1)).mean()

        # Restore and update
        diff = (loss_p - loss_m) / (2 * eps)
        for p, d in zip(param_arrays, deltas):
            p += eps * d  # restore
            p -= lr * diff / d  # SPSA update

        if step % 10 == 0:
            losses.append(float((loss_p + loss_m) / 2))

    return losses


class LoRAFlowGate:
    """LoRA-Flow fusion gate (Wang et al., 2024).
    w^l = softmax(W_gate^l @ x_t^l) + b^l
    Params per layer: k*(d+1). Total: L*k*(d+1).
    """
    def __init__(self, k, rng):
        self.k = k
        self.W_gate = [rng.standard_normal((k, D_MODEL)).astype(np.float32) * 0.01
                       for _ in range(N_LAYERS)]
        self.b_gate = [np.ones(k, dtype=np.float32) * (1.0 - 1.0/k)
                       for _ in range(N_LAYERS)]
        self.n_params = N_LAYERS * k * (D_MODEL + 1)

    def get_weights(self, h, l):
        logits = h @ self.W_gate[l].T
        w = softmax(logits) + self.b_gate[l][np.newaxis, :]
        return w

    def get_param_arrays(self):
        return self.W_gate + self.b_gate

    def train(self, model, edl, td, rng):
        return _train_gate_spsa(self, model, edl, td, rng)


class XLoRAGate:
    """X-LoRA style 2-layer MLP gating (Buehler, 2024).
    w^l = softmax(W2^l @ ReLU(W1^l @ x_t^l))
    Params per layer: h*(d+k). Total: L*h*(d+k).
    """
    def __init__(self, k, rng, h=16):
        self.k = k
        self.hd = h
        self.W1 = [rng.standard_normal((h, D_MODEL)).astype(np.float32) * 0.01
                    for _ in range(N_LAYERS)]
        self.W2 = [rng.standard_normal((k, h)).astype(np.float32) * 0.01
                    for _ in range(N_LAYERS)]
        self.n_params = N_LAYERS * h * (D_MODEL + k)

    def get_weights(self, h, l):
        z = np.maximum(h @ self.W1[l].T, 0)
        return softmax(z @ self.W2[l].T)

    def get_param_arrays(self):
        return self.W1 + self.W2

    def train(self, model, edl, td, rng):
        return _train_gate_spsa(self, model, edl, td, rng)


# =============================================================================
# Metrics
# =============================================================================
def compute_cos_matrix(loras):
    vecs = [l.flatten() for l in loras]
    n = len(vecs)
    norms = [np.linalg.norm(v) for v in vecs]
    mat = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            c = float(np.dot(vecs[i], vecs[j]) / max(norms[i]*norms[j], 1e-10))
            mat[i,j] = mat[j,i] = c
    return mat


# =============================================================================
# Main
# =============================================================================
def run_experiment(seed):
    rng = np.random.default_rng(seed)
    domain_names = list(DOMAINS.keys())
    print(f"\n{'='*60}\nSeed {seed}\n{'='*60}")

    # Data
    print("Generating data...")
    train_data, eval_data = {}, {}
    for idx, (name, info) in enumerate(DOMAINS.items()):
        tm = make_transition_matrix(info['cluster'], idx, rng)
        train_data[name] = generate_data(tm, 500, SEQ_LEN + 1, rng)
        eval_data[name] = generate_data(tm, EVAL_SAMPLES, SEQ_LEN + 1, rng)

    # Base model
    print("Training base model...")
    model = MLP(rng)
    all_train = np.concatenate(list(train_data.values()), axis=0)
    n = len(all_train)
    for step in range(BASE_STEPS):
        idx = rng.integers(0, n, size=BATCH_SIZE)
        t = rng.integers(0, SEQ_LEN)
        loss, grads, _ = model.loss_and_grads(all_train[idx, t], all_train[idx, t+1])
        model.embed -= BASE_LR * grads['embed']
        model.W_out -= BASE_LR * grads['W_out']
        model.b_out -= BASE_LR * grads['b_out']
        for l in range(N_LAYERS):
            model.W_up[l] -= BASE_LR * grads[f'W_up_{l}']
            model.b_up[l] -= BASE_LR * grads[f'b_up_{l}']
            model.W_down[l] -= BASE_LR * grads[f'W_down_{l}']
            model.b_down[l] -= BASE_LR * grads[f'b_down_{l}']

    base_losses = {n: model.ntp_loss(eval_data[n]) for n in domain_names}
    print(f"  Base mean NTP loss: {np.mean(list(base_losses.values())):.4f}")

    # Train experts
    print(f"Training {len(domain_names)} experts...")
    experts = {}
    ind_losses = {}
    for name in domain_names:
        lora = LoRA(rng)
        td = train_data[name]
        for step in range(EXPERT_STEPS):
            idx = rng.integers(0, len(td), size=BATCH_SIZE)
            t = rng.integers(0, SEQ_LEN)
            lora.train_step(model, td[idx, t], td[idx, t+1], EXPERT_LR)
        experts[name] = lora
        ind_losses[name] = model.ntp_loss(eval_data[name], lora.get_deltas())
    print(f"  Expert mean NTP loss: {np.mean(list(ind_losses.values())):.4f}")

    # Orthogonality
    el = [experts[n] for n in domain_names]
    cm = compute_cos_matrix(el)
    off = [abs(cm[i,j]) for i in range(len(el)) for j in range(i+1, len(el))]
    mean_cos = float(np.mean(off))
    print(f"  Mean |cos|: {mean_cos:.6f}")

    # Compositions at different scales
    compositions = [
        ('within_k2', ['code_a', 'code_b']),
        ('cross_k2',  ['code_a', 'reason_a']),
        ('cross_k6',  ['code_a', 'code_b', 'reason_a', 'reason_b', 'know_a', 'know_b']),
        ('all_k12',   domain_names),
    ]

    results = {
        'base_losses': {k: float(v) for k, v in base_losses.items()},
        'individual_losses': {k: float(v) for k, v in ind_losses.items()},
        'orthogonality': {'mean_abs_cos': mean_cos},
        'compositions': {},
    }

    for comp_name, comp_domains in compositions:
        k = len(comp_domains)
        print(f"\n  === {comp_name} (k={k}) ===")
        comp_loras = [experts[d] for d in comp_domains]
        comp_eval = np.concatenate([eval_data[d] for d in comp_domains], axis=0)
        edl = [lora.get_deltas() for lora in comp_loras]

        # SOLE
        t0 = time.perf_counter()
        sole_d = compose_sole(comp_loras)
        loss_sole = model.ntp_loss(comp_eval, sole_d)
        t_sole = time.perf_counter() - t0
        print(f"    SOLE:      {loss_sole:.4f}  ({t_sole:.2f}s)")

        # Avg
        avg_d = compose_avg(comp_loras)
        loss_avg = model.ntp_loss(comp_eval, avg_d)
        print(f"    Avg(1/k):  {loss_avg:.4f}")

        # CAT
        t0 = time.perf_counter()
        cat_d, cat_w = compose_cat(model, comp_loras, comp_eval, rng)
        loss_cat = model.ntp_loss(comp_eval, cat_d)
        t_cat = time.perf_counter() - t0
        print(f"    CAT:       {loss_cat:.4f}  ({t_cat:.2f}s, {t_cat/max(t_sole,.001):.0f}x)")

        # LoRA-Flow
        t0 = time.perf_counter()
        fg = LoRAFlowGate(k, rng)
        flow_td = comp_eval[:GATE_TRAIN_N]
        fl = fg.train(model, edl, flow_td, rng)
        loss_flow = model.ntp_loss_dynamic(comp_eval, edl, lambda h, l: fg.get_weights(h, l))
        t_flow = time.perf_counter() - t0
        print(f"    LoRA-Flow: {loss_flow:.4f}  ({t_flow:.2f}s, {t_flow/max(t_sole,.001):.0f}x, "
              f"params={fg.n_params})")

        # X-LoRA
        t0 = time.perf_counter()
        xg = XLoRAGate(k, rng)
        xl = xg.train(model, edl, flow_td, rng)
        loss_xlora = model.ntp_loss_dynamic(comp_eval, edl, lambda h, l: xg.get_weights(h, l))
        t_xlora = time.perf_counter() - t0
        print(f"    X-LoRA:    {loss_xlora:.4f}  ({t_xlora:.2f}s, {t_xlora/max(t_sole,.001):.0f}x, "
              f"params={xg.n_params})")

        # Base
        loss_base = model.ntp_loss(comp_eval)
        print(f"    Base:      {loss_base:.4f}")

        # Quality gain
        sole_imp = max(0, loss_base - loss_sole)
        flow_imp = max(0, loss_base - loss_flow)
        gain = ((flow_imp - sole_imp) / sole_imp * 100) if sole_imp > 1e-6 else 0.0

        # Analyze gate weight distribution
        sx = comp_eval[:32, 0]
        wstats = {}
        for l in range(N_LAYERS):
            w = fg.get_weights(model.embed[sx], l)
            wstats[f'L{l}'] = {'mean': w.mean(0).tolist(), 'std': w.std(0).tolist()}

        results['compositions'][comp_name] = {
            'k': k, 'domains': comp_domains,
            'sole_loss': float(loss_sole), 'avg_loss': float(loss_avg),
            'cat_loss': float(loss_cat), 'flow_loss': float(loss_flow),
            'xlora_loss': float(loss_xlora), 'base_loss': float(loss_base),
            'sole_time': float(t_sole), 'cat_time': float(t_cat),
            'flow_time': float(t_flow), 'xlora_time': float(t_xlora),
            'flow_params': fg.n_params, 'xlora_params': xg.n_params,
            'flow_gain_pct': float(gain),
            'flow_weight_stats': wstats, 'cat_weights': cat_w,
        }

    return results


def run_all():
    all_results = {}
    t0 = time.perf_counter()
    for seed in SEEDS:
        all_results[seed] = run_experiment(seed)
    elapsed = time.perf_counter() - t0

    # Aggregate
    print(f"\n{'='*70}\nAGGREGATED (3 seeds)\n{'='*70}")
    comp_names = list(all_results[SEEDS[0]]['compositions'].keys())
    summary = {}

    for cn in comp_names:
        vals = {m: [all_results[s]['compositions'][cn][f'{m}_loss'] for s in SEEDS]
                for m in ['sole', 'avg', 'cat', 'flow', 'xlora', 'base']}
        times = {m: [all_results[s]['compositions'][cn][f'{m}_time'] for s in SEEDS]
                 for m in ['sole', 'cat', 'flow', 'xlora']}
        gains = [all_results[s]['compositions'][cn]['flow_gain_pct'] for s in SEEDS]
        k = all_results[SEEDS[0]]['compositions'][cn]['k']

        summary[cn] = {'k': k}
        for m in vals:
            summary[cn][f'{m}_mean'] = float(np.mean(vals[m]))
            summary[cn][f'{m}_std'] = float(np.std(vals[m]))
        for m in times:
            summary[cn][f'{m}_time'] = float(np.mean(times[m]))
        summary[cn]['flow_gain_mean'] = float(np.mean(gains))
        summary[cn]['flow_gain_std'] = float(np.std(gains))
        summary[cn]['flow_params'] = all_results[SEEDS[0]]['compositions'][cn]['flow_params']
        summary[cn]['xlora_params'] = all_results[SEEDS[0]]['compositions'][cn]['xlora_params']

        s = summary[cn]
        print(f"\n{cn} (k={k}):")
        print(f"  Base:      {s['base_mean']:.4f}")
        print(f"  SOLE:      {s['sole_mean']:.4f} +/- {s['sole_std']:.4f}  t={s['sole_time']:.2f}s")
        print(f"  Avg(1/k):  {s['avg_mean']:.4f} +/- {s['avg_std']:.4f}")
        print(f"  CAT:       {s['cat_mean']:.4f} +/- {s['cat_std']:.4f}  t={s['cat_time']:.2f}s")
        print(f"  LoRA-Flow: {s['flow_mean']:.4f} +/- {s['flow_std']:.4f}  t={s['flow_time']:.2f}s  "
              f"p={s['flow_params']}")
        print(f"  X-LoRA:    {s['xlora_mean']:.4f} +/- {s['xlora_std']:.4f}  t={s['xlora_time']:.2f}s  "
              f"p={s['xlora_params']}")
        print(f"  Flow gain: {s['flow_gain_mean']:+.1f}% +/- {s['flow_gain_std']:.1f}%")

    # Ortho
    ortho = [all_results[s]['orthogonality']['mean_abs_cos'] for s in SEEDS]
    print(f"\nOrthogonality: |cos| = {np.mean(ortho):.6f} +/- {np.std(ortho):.6f}")

    # Kill criteria
    print(f"\n{'='*70}\nKILL CRITERIA\n{'='*70}")

    # K1
    print("\nK1: LoRA-Flow >10% quality gain over SOLE?")
    k1_survives = True
    for cn in comp_names:
        g = summary[cn]['flow_gain_mean']
        overhead = summary[cn]['flow_time'] / max(summary[cn]['sole_time'], .001)
        status = "TRIGGERED" if g > 10 else "SURVIVES"
        if g > 10: k1_survives = False
        print(f"  {cn}: gain={g:+.1f}%, overhead={overhead:.0f}x -> {status}")
    print(f"  >> K1 overall: {'SURVIVES' if k1_survives else 'TRIGGERED'}")

    # K2
    print("\nK2: LoRA-Flow feasible at N>10?")
    for cn in comp_names:
        s = summary[cn]
        if s['k'] > 10:
            print(f"  {cn} (k={s['k']}): params={s['flow_params']}, time={s['flow_time']:.1f}s")
            # At production (d=4096, L=32): params = 32*k*(4096+1)
            prod_params = 32 * s['k'] * (4096 + 1)
            print(f"    Production (d=4096, L=32): {prod_params:,} params ({prod_params/1e6:.1f}M)")
            if s['k'] <= 50:
                print(f"    >> K2: feasible at k={s['k']} (params manageable)")
            else:
                print(f"    >> K2: param count scales as O(k*d*L) -- {prod_params/1e6:.1f}M at k={s['k']}")

    # Param scaling
    print(f"\n{'='*70}\nPARAM SCALING (production: d=4096, L=32)\n{'='*70}")
    for k in [2, 10, 50, 100, 500]:
        fp = 32*k*(4096+1); xp = 32*64*(4096+k); cp = 2*k*32
        print(f"  k={k:3d}: SOLE=0  CAT={cp:6d}  Flow={fp:10,}  X-LoRA={xp:10,}")

    # Save
    output = {
        'seeds': SEEDS, 'summary': summary,
        'per_seed': {str(s): _ser(all_results[s]) for s in SEEDS},
        'orthogonality': {'mean': float(np.mean(ortho)), 'std': float(np.std(ortho))},
        'kill_criteria': {'K1_flow_gt_10pct': not k1_survives},
        'total_time_s': elapsed,
    }
    out = Path(__file__).parent / 'results.json'
    with open(out, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    print(f"Total time: {elapsed:.1f}s")
    return output


def _ser(d):
    if isinstance(d, dict): return {k: _ser(v) for k, v in d.items()}
    if isinstance(d, list): return [_ser(v) for v in d]
    if isinstance(d, (np.floating, np.integer)): return float(d)
    if isinstance(d, np.ndarray): return d.tolist()
    return d


if __name__ == '__main__':
    run_all()
