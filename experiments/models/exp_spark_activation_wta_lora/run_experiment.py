#!/usr/bin/env python3
"""exp_spark_activation_wta_lora — per-token activation-L2 winner-takes-all vs uniform-1/N merge.

Frozen base mlx-community/gemma-4-e4b-it-4bit + N=3 r=6 q_proj LoRA adapters
(data/adapters/{math,python,medical}). On-domain experts: math (GSM8K) and python (HumanEval);
medical is the off-domain distractor for BOTH benchmarks (and, per benchmark, the other on-domain
expert acts as an additional distractor: python is off-domain on GSM8K, math is off-domain on
HumanEval). NOTE: the pre-registered pool named a 4th distractor, but the only adapters in the repo
that are structurally compatible (target q_proj, 42 layers, r=6, shape (2560,6)) are math/python/
medical — `sql` ships no adapters.safetensors and `thinking-openthoughts-universal-v0` targets
o_proj/v_proj on layers 26-41 only. The hypothesis, both magnitude-matched controls, and the kill
criterion are unchanged; only N drops 4->3 (the s/N dilution and argmax routing both adapt to N).

Per q_proj, per token, each adapter x emits delta_hat_i = (h@A_i)@B_i (unscaled), loudness ell_i =
||delta_hat_i||_2. Five arms (single q_proj wrapper, mode-switched):

  base         : y = W h                                       (no adapters)
  sum_uniform  : y = W h + (s/N) * sum_i delta_hat_i           (diluted merge / interference)
  wta_full     : y = W h + s   * delta_hat_{argmax ell}        (routing + FULL magnitude — hypothesis)
  wta_scaled   : y = W h + (s/N)* delta_hat_{argmax ell}       (routing at MATCHED magnitude — control)
  rand_full    : y = W h + s   * delta_hat_{random pick}       (FULL magnitude, WRONG routing — control)

The two controls isolate F#863's magnitude confound: a wta_full>sum_uniform win is only
"loudness=correctness" if ALSO wta_full>rand_full (loudness beats random at matched magnitude) and
wta_scaled>sum_uniform (routing helps at matched magnitude). See MATH.md sec 3/6.

KILL K2303 (target, behavioral): KILL if acc(wta_full) - acc(sum_uniform) < +0.05, where
acc = mean(GSM8K exact-match n40, HumanEval pass@1 n40).

Composition is sum_i B_i A_i (independent deltas), never (sum B)(sum A). LORA_SCALE=6.0 <= 8.
Wrapper attaches via subclass nn.Module + setattr (never __call__ override on instance, F#831).
NO MOCKS. Real model, real adapters, real benchmark execution. is_smoke=False. mlx-lm == 0.31.2.
"""

import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

device_info = mx.device_info()
total_mem = device_info["memory_size"]
mx.set_memory_limit(total_mem - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"
REPO_ROOT = EXP_DIR.parent.parent.parent

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
ADAPTER_NAMES = ["python", "math", "medical"]
ADAPTER_FILES = [REPO_ROOT / "data" / "adapters" / n / "adapters.safetensors" for n in ADAPTER_NAMES]

LORA_SCALE = 6.0          # <= 8 guard OK (matches adapter_config scale)
LORA_RANK = 6
N_ADAPTERS = len(ADAPTER_NAMES)
N_GSM8K = 40
N_HUMANEVAL = 40
MAX_NEW_TOKENS = 1024     # thinking-mode headroom
SEED = 42
RAND_SEED = 1337          # fixed seed for rand_full per-token pick
N_LAYERS_EXPECTED = 42

K_WTA_VS_SUM = 0.05       # K2303: acc(wta_full) - acc(sum_uniform) must be >= +0.05

# Arm modes
MODE_BASE = "base"
MODE_SUM = "sum_uniform"
MODE_WTA_FULL = "wta_full"
MODE_WTA_SCALED = "wta_scaled"
MODE_RAND_FULL = "rand_full"
ARMS = [MODE_BASE, MODE_SUM, MODE_WTA_FULL, MODE_WTA_SCALED, MODE_RAND_FULL]


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Multi-adapter q_proj wrapper (one instance covers all arms via .mode)
# subclass nn.Module + setattr (NEVER __call__ override on instance — F#831)
# ----------------------------------------------------------------------------

class WTAQProj(nn.Module):
    """y = linear(x) + composition over N adapter deltas, selected by self.mode.

    a_list[i]: (d_in, r)   b_list[i]: (r, d_out)   delta_hat_i(x) = (x@a_i)@b_i.
    loudness ell_i = ||delta_hat_i||_2 over last axis (per token). scale s cancels in argmax.
    """
    def __init__(self, base_linear, a_list, b_list, scale, n_adapters):
        super().__init__()
        self.linear = base_linear            # frozen QuantizedLinear
        self.a_list = a_list                 # list of (d_in, r)
        self.b_list = b_list                 # list of (r, d_out)
        self.scale = scale
        self.n = n_adapters
        self.mode = MODE_BASE
        self.rng_counter = [0]               # shared mutable ref for rand_full determinism
        self.linear.freeze()

    def _deltas(self, x):
        # returns stacked deltas: (n, ..., d_out)
        ds = []
        for i in range(self.n):
            ds.append((x @ self.a_list[i]) @ self.b_list[i])
        return mx.stack(ds, axis=0)          # (n, B, T, d_out)

    def __call__(self, x):
        y = self.linear(x)
        if self.mode == MODE_BASE:
            return y

        deltas = self._deltas(x)             # (n, ..., d_out)

        if self.mode == MODE_SUM:
            inj = (self.scale / self.n) * mx.sum(deltas, axis=0)
            return y + inj.astype(x.dtype)

        if self.mode in (MODE_WTA_FULL, MODE_WTA_SCALED):
            # per-token L2 loudness over each adapter, argmax over n
            loud = mx.sqrt(mx.sum(deltas * deltas, axis=-1))   # (n, ..., )
            win = mx.argmax(loud, axis=0)                      # (...,)
            # gather winning delta: one-hot select over axis 0
            onehot = (mx.arange(self.n)[:, None] == win.reshape(-1)[None, :])  # (n, M)
            M = win.size
            d_out = deltas.shape[-1]
            flat = deltas.reshape(self.n, M, d_out)            # (n, M, d_out)
            sel = mx.sum(flat * onehot[:, :, None], axis=0)    # (M, d_out)
            sel = sel.reshape(*win.shape, d_out)
            s = self.scale if self.mode == MODE_WTA_FULL else (self.scale / self.n)
            return y + (s * sel).astype(x.dtype)

        if self.mode == MODE_RAND_FULL:
            # random adapter pick per token, fixed seed + deterministic counter
            M = 1
            shape = deltas.shape[1:-1]        # (..., ) token dims
            for d in shape:
                M *= d
            key = mx.random.key(RAND_SEED + self.rng_counter[0])
            self.rng_counter[0] += 1
            pick = mx.random.randint(0, self.n, (M,), key=key)  # (M,)
            d_out = deltas.shape[-1]
            flat = deltas.reshape(self.n, M, d_out)
            onehot = (mx.arange(self.n)[:, None] == pick[None, :])
            sel = mx.sum(flat * onehot[:, :, None], axis=0)     # (M, d_out)
            sel = sel.reshape(*shape, d_out)
            return y + (self.scale * sel).astype(x.dtype)

        raise ValueError(f"unknown mode {self.mode}")


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach_wta(model, adapters, scale):
    """Wrap q_proj on every layer with WTAQProj holding all N adapters. Returns wrapper list."""
    lm = get_lm(model)
    wrappers = []
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        if ak not in adapters[0] or bk not in adapters[0]:
            continue
        a_list, b_list = [], []
        ref_a, ref_b = adapters[0][ak].shape, adapters[0][bk].shape
        for ad in adapters:
            assert ak in ad and bk in ad, f"adapter missing {ak}"
            assert ad[ak].shape == ref_a and ad[bk].shape == ref_b, \
                f"shape mismatch layer {li}: {ad[ak].shape} vs {ref_a}"
            a_list.append(ad[ak].astype(mx.float32))
            b_list.append(ad[bk].astype(mx.float32))
        base_linear = layer.self_attn.q_proj
        wrapper = WTAQProj(base_linear, a_list, b_list, scale, len(adapters))
        setattr(layer.self_attn, "q_proj", wrapper)   # canonical: setattr
        wrappers.append(wrapper)
    mx.eval(model.parameters())
    assert len(wrappers) == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED} wrapped, got {len(wrappers)}"
    log(f"  Attached {len(wrappers)} WTAQProj (N={len(adapters)} adapters each)")
    return wrappers


def set_mode(wrappers, mode):
    for w in wrappers:
        w.mode = mode
        w.rng_counter[0] = 0
    log(f"  -> arm mode = {mode}")


# ----------------------------------------------------------------------------
# Generation (greedy, thinking mode)
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def generate(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS):
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos
    for _ in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
    del cache
    return tokenizer.decode(out), len(out)


def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ----------------------------------------------------------------------------
# GSM8K (exact-match on final integer)
# ----------------------------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        gold = it["answer"].split("####")[-1].strip().replace(",", "")
        probs.append({"question": it["question"], "gold": gold})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


def gsm8k_prompt(p):
    return (
        "Solve this math problem. Show your reasoning, then give the final answer "
        "on a new line in the form '#### <number>'.\n\n" + p["question"]
    )


def extract_gsm8k_answer(text):
    text = strip_thinking(text)
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""


def num_eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def eval_gsm8k(model, tokenizer, problems):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok = generate(model, tokenizer, prompt)
        pred = extract_gsm8k_answer(text)
        ok = num_eq(pred, p["gold"])
        passed += int(ok)
        details.append({"gold": p["gold"], "pred": pred, "passed": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    GSM8K exact-match = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# HumanEval (real unit-test execution)
# ----------------------------------------------------------------------------

def load_humaneval(n):
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        probs.append({
            "task_id": it["task_id"], "prompt": it["prompt"],
            "test": it["test"], "entry_point": it["entry_point"],
        })
    log(f"  Loaded {len(probs)} HumanEval problems")
    return probs


def humaneval_prompt(p):
    return (
        "Complete this Python function. Return the full function in a "
        "```python code block.\n\n```python\n" + p["prompt"] + "\n```"
    )


def extract_code(text, prompt_code, entry_point):
    text = strip_thinking(text)
    blocks = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
    for blk in blocks:
        if f"def {entry_point}" in blk:
            return blk.strip()
    if blocks:
        return prompt_code + "\n" + blocks[0].strip()
    if f"def {entry_point}" in text:
        return text
    return prompt_code + "\n" + text


def run_humaneval_test(code, test, entry_point, timeout=12):
    full = f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=timeout)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def eval_humaneval(model, tokenizer, problems):
    passed, details = 0, []
    for p in problems:
        prompt = format_chat(tokenizer, humaneval_prompt(p))
        text, ntok = generate(model, tokenizer, prompt)
        code = extract_code(text, p["prompt"], p["entry_point"])
        ok = run_humaneval_test(code, p["test"], p["entry_point"])
        passed += int(ok)
        details.append({"task_id": p["task_id"], "passed": ok, "ntok": ntok})
    acc = passed / len(problems)
    log(f"    HumanEval pass@1 = {acc:.4f} ({passed}/{len(problems)})")
    return acc, details


# ----------------------------------------------------------------------------
# Run one arm: load model, attach, set mode, eval both benchmarks
# ----------------------------------------------------------------------------

def run_arm(mode, gsm8k_probs, he_probs, adapters):
    log(f"\n=== ARM {mode} ===")
    model, tok = load(MODEL_ID)
    if mode != MODE_BASE:
        wrappers = attach_wta(model, adapters, LORA_SCALE)
        set_mode(wrappers, mode)
    else:
        # still wrap so base shares the exact same code path with mode=base
        wrappers = attach_wta(model, adapters, LORA_SCALE)
        set_mode(wrappers, MODE_BASE)
    gc.collect(); mx.clear_cache()

    g_acc, g_det = eval_gsm8k(model, tok, gsm8k_probs)
    h_acc, h_det = eval_humaneval(model, tok, he_probs)
    avg = 0.5 * (g_acc + h_acc)
    log(f"  ARM {mode}: gsm8k={g_acc:.3f} humaneval={h_acc:.3f} avg={avg:.3f}")
    log_mem(f"{mode}-done")
    del model, tok, wrappers
    gc.collect(); mx.clear_cache()
    return {"gsm8k": g_acc, "humaneval": h_acc, "avg": avg,
            "gsm8k_details": g_det, "humaneval_details": h_det}


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_spark_activation_wta_lora")
    log(f"Base: {MODEL_ID}")
    log(f"Adapters (N={N_ADAPTERS}): {ADAPTER_NAMES}")
    log(f"n_gsm8k={N_GSM8K} n_humaneval={N_HUMANEVAL} scale={LORA_SCALE} rank={LORA_RANK}")
    log(f"K2303: kill if acc(wta_full)-acc(sum_uniform) < {K_WTA_VS_SUM}")
    log("=" * 72)
    for f in ADAPTER_FILES:
        assert f.exists(), f"missing adapter {f}"
    log_mem("start")

    log("\n=== Load data + adapters ===")
    gsm8k_probs = load_gsm8k(N_GSM8K)
    he_probs = load_humaneval(N_HUMANEVAL)
    adapters = [mx.load(str(f)) for f in ADAPTER_FILES]

    arm_results = {}
    for mode in ARMS:
        arm_results[mode] = run_arm(mode, gsm8k_probs, he_probs, adapters)

    acc = {m: arm_results[m]["avg"] for m in ARMS}

    # ---- Kill criterion K2303 (target, behavioral) ----
    delta_wta_vs_sum = acc[MODE_WTA_FULL] - acc[MODE_SUM]
    killed = delta_wta_vs_sum < K_WTA_VS_SUM
    verdict = "killed" if killed else "supported"
    all_pass = not killed

    # ---- Confound isolation riders (F#863) ----
    delta_routing_matched = acc[MODE_WTA_SCALED] - acc[MODE_SUM]      # routing at matched magnitude
    delta_routing_vs_random = acc[MODE_WTA_FULL] - acc[MODE_RAND_FULL]  # loudness beats random pick
    confound_magnitude_only = (not killed) and (delta_routing_vs_random <= 0.0)

    results = {
        "experiment_id": "exp_spark_activation_wta_lora",
        "config": {
            "base_model": MODEL_ID,
            "adapters": ADAPTER_NAMES,
            "adapter_files": [str(f) for f in ADAPTER_FILES],
            "n_adapters": N_ADAPTERS,
            "lora_scale": LORA_SCALE,
            "lora_rank": LORA_RANK,
            "n_gsm8k": N_GSM8K,
            "n_humaneval": N_HUMANEVAL,
            "max_new_tokens": MAX_NEW_TOKENS,
            "rand_seed": RAND_SEED,
            "k2303_wta_vs_sum": K_WTA_VS_SUM,
            "mlx_lm": "0.31.2",
        },
        "arms": {m: {k: v for k, v in arm_results[m].items()} for m in ARMS},
        "acc_avg": acc,
        "delta_wta_vs_sum": delta_wta_vs_sum,
        "delta_routing_matched_wta_scaled_minus_sum": delta_routing_matched,
        "delta_routing_vs_random_wta_full_minus_rand_full": delta_routing_vs_random,
        "confound_magnitude_only": bool(confound_magnitude_only),
        "kill_criteria": {
            "2303": {
                "text": "acc(wta_full) - acc(sum_uniform) < +0.05 (avg GSM8K n40 + HumanEval n40)",
                "type": "target_behavioral",
                "measured_delta": delta_wta_vs_sum,
                "threshold": K_WTA_VS_SUM,
                "result": "fail" if killed else "pass",
            }
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "total_wall_clock_sec": time.time() - t0,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    for m in ARMS:
        log(f"acc {m:12s} = {acc[m]:.3f}  (gsm8k={arm_results[m]['gsm8k']:.3f} he={arm_results[m]['humaneval']:.3f})")
    log(f"Delta_wta_vs_sum            = {delta_wta_vs_sum:+.3f}  (K2303 thresh {K_WTA_VS_SUM})")
    log(f"Delta_routing_matched      = {delta_routing_matched:+.3f}  (wta_scaled - sum_uniform)")
    log(f"Delta_routing_vs_random    = {delta_routing_vs_random:+.3f}  (wta_full - rand_full)")
    log(f"confound_magnitude_only    = {confound_magnitude_only}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
