#!/usr/bin/env python3
"""exp_pgolf_random_frame_compose — PGolf replication+ (ref #707) on a new MLX micro substrate.

Phase 1 (K2320, replication): byte-level GPT (d=256, L=4, H=4, ctx=256, tied emb) where every
attn/MLP linear is a FROZEN random matrix + trained rank-32 LoRA, vs a fully-trained dense control
at equal training budget (1500 steps, same data, same seeds for batching).
  KILL K2320 if bpb_random_frame - bpb_dense > 0.08.

Phase 2 (K2321, bet dfa-init extension): merge the trained random-frame model into a frozen base,
then train per-domain adapters delta_i = B_i (A_i x) with B_i a FROZEN orthonormal frame and A_i
trained (zero-init), on two text domains (prose=tinyshakespeare, code=python stdlib). Two arms:
  shared   : B_prose = B_code = Q[:, :r]      (same output subspace)
  disjoint : B_prose = Q[:, :r], B_code = Q[:, r:2r], Q orthonormal => B_p^T B_c = 0 exactly.
Composition is sum_i B_i (A_i x)  — never (sum B)(sum A). Adapter scale 1.0 <= 8.
Interference per arm: I = mean_i [ bpb_composed(val_i) - bpb_solo_i(val_i) ].
  KILL K2321 if (I_shared - I_disjoint) / I_shared < 0.20.
  Validity gate (pre-registered in MATH.md): K2321 decidable only if I_shared >= 0.02 BPB,
  otherwise K2321 is inconclusive and the verdict is PROVISIONAL.

NO MOCKS. Real training, real text, real measured BPB. is_smoke=false. mlx==0.31.1.
"""

import gc
import hashlib
import json
import math
import os
import sys
import sysconfig
import time
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"
RESULTS_FILE = EXP_DIR / "results.json"
DATA_DIR.mkdir(exist_ok=True)

# ---------------- config (pre-registered in MATH.md) ----------------
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
CTX = 256
BATCH = 32
VOCAB = 256                     # raw bytes
RANK_P1 = 32                    # phase-1 trained LoRA rank (ref #707 regime)
RANK_P2 = 16                    # phase-2 frozen-frame rank (2r = 32 <= d)
STEPS_P1 = int(os.environ.get("STEPS_P1", "1500"))
STEPS_P2 = int(os.environ.get("STEPS_P2", "800"))
LR = 1.5e-3
LR_P2 = 1.0e-3
WARMUP = 100
SEED = 1234
ADAPTER_SCALE = 1.0             # <= 8 guard
K2320_THRESH = 0.08             # BPB gap kill threshold (replication)
K2321_CUT = 0.20                # required interference reduction fraction
I_SHARED_MIN = 0.02             # validity gate: interference must be present
EVAL_BATCHES = 24
LN2 = math.log(2.0)

IS_SMOKE = os.environ.get("SMOKE", "0") == "1"
if IS_SMOKE:
    STEPS_P1, STEPS_P2 = 40, 30


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------- data ----------------
def load_prose() -> bytes:
    f = DATA_DIR / "tinyshakespeare.txt"
    if not f.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        log(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=30) as r:
            f.write_bytes(r.read())
    data = f.read_bytes()
    assert len(data) > 500_000, f"prose corpus too small: {len(data)}"
    return data


def load_code(max_bytes=4_000_000) -> bytes:
    stdlib = Path(sysconfig.get_paths()["stdlib"])
    chunks, total = [], 0
    for p in sorted(stdlib.glob("*.py")) + sorted((stdlib / "json").glob("*.py")) + sorted(
        (stdlib / "email").glob("*.py")) + sorted((stdlib / "asyncio").glob("*.py")):
        try:
            b = p.read_bytes()
        except OSError:
            continue
        chunks.append(b)
        total += len(b)
        if total >= max_bytes:
            break
    data = b"\n".join(chunks)[:max_bytes]
    assert len(data) > 1_000_000, f"code corpus too small: {len(data)}"
    return data


def split(data: bytes, frac=0.9):
    n = int(len(data) * frac)
    return np.frombuffer(data[:n], dtype=np.uint8), np.frombuffer(data[n:], dtype=np.uint8)


def mix_bytes(a: np.ndarray, b: np.ndarray, chunk=2048) -> np.ndarray:
    """50/50 interleave by chunks, length-limited by the smaller corpus."""
    n = min(len(a), len(b)) // chunk * chunk
    a, b = a[:n].reshape(-1, chunk), b[:n].reshape(-1, chunk)
    return np.stack([a, b], axis=1).reshape(-1)


def train_batches(arr: np.ndarray, steps: int, rng: np.random.RandomState):
    hi = len(arr) - CTX - 1
    for _ in range(steps):
        idx = rng.randint(0, hi, size=BATCH)
        x = np.stack([arr[i:i + CTX] for i in idx]).astype(np.int32)
        y = np.stack([arr[i + 1:i + CTX + 1] for i in idx]).astype(np.int32)
        yield mx.array(x), mx.array(y)


def val_batches(arr: np.ndarray, max_batches=EVAL_BATCHES):
    """Deterministic non-overlapping windows."""
    span = CTX + 1
    n_win = (len(arr) - 1) // CTX
    out = []
    for b in range(min(max_batches, n_win // BATCH)):
        xs, ys = [], []
        for j in range(BATCH):
            i = (b * BATCH + j) * CTX
            xs.append(arr[i:i + CTX])
            ys.append(arr[i + 1:i + span])
        out.append((mx.array(np.stack(xs).astype(np.int32)),
                    mx.array(np.stack(ys).astype(np.int32))))
    assert out, "val set produced zero batches"
    return out


# ---------------- model ----------------
def layer_key(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


class DenseLinear(nn.Module):
    def __init__(self, d_in, d_out, key):
        super().__init__()
        self.weight = mx.random.normal((d_out, d_in), key=mx.random.key(key)) * (d_in ** -0.5)

    def __call__(self, x):
        return x @ self.weight.T


class RandomFrameLinear(nn.Module):
    """Phase 1: frozen random W0 (underscore => non-parameter) + trained LoRA (A random, B zero)."""

    def __init__(self, d_in, d_out, key):
        super().__init__()
        self._w0 = mx.random.normal((d_out, d_in), key=mx.random.key(key)) * (d_in ** -0.5)
        self.A = mx.random.normal((RANK_P1, d_in), key=mx.random.key(key + 1)) * (d_in ** -0.5)
        self.B = mx.zeros((d_out, RANK_P1))

    def __call__(self, x):
        return x @ self._w0.T + ((x @ self.A.T) @ self.B.T) * ADAPTER_SCALE

    def merged(self):
        return self._w0 + (self.B @ self.A) * ADAPTER_SCALE


class FrameAdapterLinear(nn.Module):
    """Phase 2: frozen merged base + frozen orthonormal frames B_i + trained A_i (zero-init).

    Composition is sum_i B_i (A_i x): each adapter's delta computed independently then summed.
    """

    def __init__(self, w0, frames):
        super().__init__()
        self._w0 = w0                              # frozen (underscore)
        self._frames = frames                      # list of (d_out, r), frozen
        self.As = [mx.zeros((f.shape[1], w0.shape[1])) for f in frames]

    def __call__(self, x):
        y = x @ self._w0.T
        for A, B in zip(self.As, self._frames):
            y = y + ((x @ A.T) @ B.T) * ADAPTER_SCALE
        return y


class Block(nn.Module):
    def __init__(self, mk, prefix):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.q = mk(D_MODEL, D_MODEL, f"{prefix}.q")
        self.k = mk(D_MODEL, D_MODEL, f"{prefix}.k")
        self.v = mk(D_MODEL, D_MODEL, f"{prefix}.v")
        self.o = mk(D_MODEL, D_MODEL, f"{prefix}.o")
        self.fc = mk(D_MODEL, 4 * D_MODEL, f"{prefix}.fc")
        self.proj = mk(4 * D_MODEL, D_MODEL, f"{prefix}.proj")

    def __call__(self, x, mask):
        B, L, _ = x.shape
        h = self.ln1(x)
        hd = D_MODEL // N_HEADS
        q = self.q(h).reshape(B, L, N_HEADS, hd).transpose(0, 2, 1, 3)
        k = self.k(h).reshape(B, L, N_HEADS, hd).transpose(0, 2, 1, 3)
        v = self.v(h).reshape(B, L, N_HEADS, hd).transpose(0, 2, 1, 3)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=hd ** -0.5, mask=mask)
        x = x + self.o(a.transpose(0, 2, 1, 3).reshape(B, L, D_MODEL))
        h = self.ln2(x)
        return x + self.proj(nn.gelu(self.fc(h)))


class ByteGPT(nn.Module):
    def __init__(self, mk):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(CTX, D_MODEL)
        self.blocks = [Block(mk, f"blk{i}") for i in range(N_LAYERS)]
        self.ln_f = nn.LayerNorm(D_MODEL)
        self._mask = nn.MultiHeadAttention.create_additive_causal_mask(CTX)

    def __call__(self, x):
        L = x.shape[1]
        h = self.emb(x) + self.pos(mx.arange(L))
        for b in self.blocks:
            h = b(h, self._mask[:L, :L])
        return self.emb.as_linear(self.ln_f(h))   # tied head (ref #711 anchor)


def loss_fn(model, x, y):
    return nn.losses.cross_entropy(model(x), y, reduction="mean")


def bpb(model, batches) -> float:
    tot, n = 0.0, 0
    for x, y in batches:
        l = loss_fn(model, x, y)
        mx.eval(l)
        tot += l.item() * x.size
        n += x.size
    return tot / n / LN2


def train(model, arr, steps, lr, seed, tag):
    sched = optim.join_schedules(
        [optim.linear_schedule(0.0, lr, WARMUP), optim.cosine_decay(lr, max(1, steps - WARMUP))],
        [WARMUP])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.01)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        opt.update(model, grads)
        return loss

    t0, last = time.time(), 0.0
    rng = np.random.RandomState(seed)
    for i, (x, y) in enumerate(train_batches(arr, steps, rng)):
        loss = step(x, y)
        mx.eval(state)
        last = loss.item()
        if i % 200 == 0:
            log(f"  [{tag}] step {i}/{steps} loss {last:.4f} ({time.time()-t0:.0f}s)")
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    log(f"  [{tag}] done {steps} steps, final loss {last:.4f}, trainable params {n_train:,}")
    return n_train


def frames_for(arm: str, d_out: int, lname: str):
    """Frozen orthonormal frames [B_prose, B_code] at one layer. QR on a per-layer seeded matrix.
    shared: both adapters get the SAME r-dim frame. disjoint: disjoint blocks, B1^T B2 = 0 exact."""
    rng = np.random.RandomState(layer_key(lname) % (2 ** 31))
    Q, _ = np.linalg.qr(rng.randn(d_out, 2 * RANK_P2))
    Q = mx.array(Q.astype(np.float32))
    if arm == "shared":
        return [Q[:, :RANK_P2], Q[:, :RANK_P2]]
    return [Q[:, :RANK_P2], Q[:, RANK_P2:2 * RANK_P2]]


def build_phase2_model(base_weights, base_params, arm, idxs):
    """Frozen base (merged phase-1 random-frame model) + frame adapters `idxs` per linear.
    Solo training uses idxs=[i] (ONLY that adapter exists => no gradient leaks into the other);
    composition uses idxs=[0, 1] (sum of deltas)."""
    def mk(d_in, d_out, lname):
        fr = frames_for(arm, d_out, lname)
        return FrameAdapterLinear(base_weights[lname], [fr[i] for i in idxs])

    m = ByteGPT(mk)
    m.update(base_params)        # frozen emb/pos/norms from the trained base
    m.freeze()
    for blk in m.blocks:
        for lin in (blk.q, blk.k, blk.v, blk.o, blk.fc, blk.proj):
            lin.unfreeze()       # only params are As (w0/frames are underscore => non-params)
    return m


def main():
    mx.random.seed(SEED)
    np.random.seed(SEED)

    log("=== data ===")
    prose_tr, prose_va = split(load_prose())
    code_tr, code_va = split(load_code())
    mix_tr = mix_bytes(prose_tr, code_tr)
    mix_va = mix_bytes(prose_va, code_va)
    log(f"prose train/val {len(prose_tr):,}/{len(prose_va):,}  "
        f"code {len(code_tr):,}/{len(code_va):,}  mix {len(mix_tr):,}/{len(mix_va):,}")
    va_mix = val_batches(mix_va)
    va_prose = val_batches(prose_va)
    va_code = val_batches(code_va)

    results = {
        "experiment_id": "exp_pgolf_random_frame_compose",
        "is_smoke": IS_SMOKE,
        "config": {
            "d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS, "ctx": CTX,
            "batch": BATCH, "rank_p1": RANK_P1, "rank_p2": RANK_P2,
            "steps_p1": STEPS_P1, "steps_p2": STEPS_P2, "lr_p1": LR, "lr_p2": LR_P2,
            "adapter_scale": ADAPTER_SCALE, "seed": SEED, "mlx": mx.__version__,
            "prose": "tinyshakespeare", "code": "python-stdlib-4MB",
        },
    }

    # ---------------- phase 1: replication (K2320) ----------------
    log("=== phase 1: dense control ===")
    dense = ByteGPT(lambda di, do, n: DenseLinear(di, do, layer_key(n)))
    n_dense = train(dense, mix_tr, STEPS_P1, LR, SEED + 1, "dense")
    bpb_dense = bpb(dense, va_mix)
    log(f"dense val BPB = {bpb_dense:.4f}")
    del dense
    gc.collect()
    mx.clear_cache()

    log("=== phase 1: random-frame + trained LoRA ===")
    rf = ByteGPT(lambda di, do, n: RandomFrameLinear(di, do, layer_key(n)))
    n_rf = train(rf, mix_tr, STEPS_P1, LR, SEED + 1, "random-frame")  # same batch seed = same data
    bpb_rf = bpb(rf, va_mix)
    log(f"random-frame val BPB = {bpb_rf:.4f}  (gap {bpb_rf - bpb_dense:+.4f})")

    gap = bpb_rf - bpb_dense
    k2320_fail = gap > K2320_THRESH
    results["phase1"] = {
        "bpb_dense": bpb_dense, "bpb_random_frame": bpb_rf, "gap_bpb": gap,
        "trainable_params_dense": n_dense, "trainable_params_random_frame": n_rf,
        "param_saving_frac": 1.0 - n_rf / n_dense,
    }

    # merge trained random-frame model into a frozen phase-2 base
    base_weights = {}
    for i, blk in enumerate(rf.blocks):
        for nm, lin in [("q", blk.q), ("k", blk.k), ("v", blk.v), ("o", blk.o),
                        ("fc", blk.fc), ("proj", blk.proj)]:
            base_weights[f"blk{i}.{nm}"] = lin.merged()
    mx.eval(list(base_weights.values()))
    base_params = {"emb": rf.emb.parameters(), "pos": rf.pos.parameters(),
                   "ln_f": rf.ln_f.parameters(),
                   "blocks": [{"ln1": b.ln1.parameters(), "ln2": b.ln2.parameters()}
                              for b in rf.blocks]}
    del rf
    gc.collect()
    mx.clear_cache()

    # ---------------- phase 2: shared vs disjoint frames (K2321) ----------------
    domains = [("prose", prose_tr, va_prose), ("code", code_tr, va_code)]
    interf = {}
    for arm in ("shared", "disjoint"):
        log(f"=== phase 2: arm={arm} ===")
        solo_bpb, trained_As = {}, {}
        for di, (dname, dtr, dva) in enumerate(domains):
            # solo model has ONLY this domain's adapter (idxs=[di]) — the other adapter
            # does not exist, so no gradient can leak into it during solo training.
            m = build_phase2_model(base_weights, base_params, arm, idxs=[di])
            train(m, dtr, STEPS_P2, LR_P2, SEED + 7 + di, f"{arm}/{dname}")
            solo_bpb[dname] = bpb(m, dva)
            log(f"  [{arm}/{dname}] solo val BPB = {solo_bpb[dname]:.4f}")
            trained_As[dname] = [
                lin.As[0]
                for blk in m.blocks
                for lin in (blk.q, blk.k, blk.v, blk.o, blk.fc, blk.proj)]
            del m
            gc.collect()
            mx.clear_cache()

        # compose: one model, adapter 0 = prose A, adapter 1 = code A (sum of deltas)
        comp = build_phase2_model(base_weights, base_params, arm, idxs=[0, 1])
        lins = [lin for blk in comp.blocks
                for lin in (blk.q, blk.k, blk.v, blk.o, blk.fc, blk.proj)]
        for li, lin in enumerate(lins):
            lin.As = [trained_As["prose"][li], trained_As["code"][li]]
        mx.eval(comp.parameters())
        comp_bpb = {d: bpb(comp, v) for d, _, v in domains}
        I = {d: comp_bpb[d] - solo_bpb[d] for d in comp_bpb}
        interf[arm] = {
            "solo_bpb": solo_bpb, "composed_bpb": comp_bpb,
            "interference_per_domain": I,
            "interference_mean": sum(I.values()) / len(I),
        }
        log(f"  [{arm}] composed BPB {comp_bpb}  interference {I}  "
            f"mean {interf[arm]['interference_mean']:+.4f}")
        del comp, trained_As
        gc.collect()
        mx.clear_cache()

    I_sh = interf["shared"]["interference_mean"]
    I_dj = interf["disjoint"]["interference_mean"]
    valid = I_sh >= I_SHARED_MIN
    cut = (I_sh - I_dj) / I_sh if I_sh > 0 else float("nan")
    k2321_fail = valid and (cut < K2321_CUT)
    results["phase2"] = {
        "shared": interf["shared"], "disjoint": interf["disjoint"],
        "I_shared": I_sh, "I_disjoint": I_dj, "interference_cut_frac": cut,
        "validity_gate_I_shared_min": I_SHARED_MIN, "interference_present": valid,
    }

    # ---------------- verdict ----------------
    results["kill_criteria"] = {
        "2320": {"text": "random-frame worse than dense by >0.08 BPB at equal budget",
                 "gap_bpb": gap, "threshold": K2320_THRESH,
                 "result": "fail" if k2320_fail else "pass"},
        "2321": {"text": "disjoint-frame interference not >=20% lower than shared-frame",
                 "I_shared": I_sh, "I_disjoint": I_dj, "cut_frac": cut,
                 "threshold_cut": K2321_CUT,
                 "result": ("inconclusive" if not valid else
                            ("fail" if k2321_fail else "pass"))},
    }
    if IS_SMOKE:
        verdict = "PROVISIONAL"
    elif k2320_fail or k2321_fail:
        verdict = "KILLED"
    elif not valid:
        verdict = "PROVISIONAL"  # no interference present; K2321 undecidable (pre-registered)
    else:
        verdict = "SUPPORTED"
    results["all_pass"] = (not IS_SMOKE) and (not k2320_fail) and valid and (not k2321_fail)
    results["verdict"] = verdict
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"verdict={verdict}  K2320 gap={gap:+.4f} (<= {K2320_THRESH})  "
        f"I_shared={I_sh:+.4f} I_disjoint={I_dj:+.4f} cut={cut:+.3f} (>= {K2321_CUT})")
    log(f"results -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
