#!/usr/bin/env python3
"""exp_spark_intra_rank_phase_gate — A single LoRA beats itself: tail-ranks @ reasoning +
head-ranks @ answer-emit, norm-held-constant, beats uniform math-solo by >=5pp GSM8K EM.

Frozen base mlx-community/gemma-4-e4b-it-4bit. ONE q_proj LoRA adapter (math, r=6, scale=6.0,
per-layer). Per layer l the delta Delta^l = s * B^l A^l (out x in, rank<=r) is SVD'd:
Delta^l = U^l S^l V^lT. U^l (out x r) are the left singular directions in q-output space.

The per-token injected delta is delta_full(x_t) = (s * B A) x_t  (lives in span(U)).
A sub-rank arm injects the projection onto a chosen set S of singular directions:
  delta_S(x_t) = U_S @ (U_S^T @ delta_full(x_t))
then RENORMALIZES per token to the full norm (MAGNITUDE-MATCH, F#863):
  delta_inject = delta_S * (||delta_full|| / (||delta_S|| + eps)).
=> ||delta_inject|| == ||delta_full|| at every layer & token (asserted, tol 1e-3 rel).

head = top ceil(r/2) singular dirs; tail = bottom floor(r/2).

Reasoning->answer-emit boundary detected at DECODE time from GENERATED tokens by
string-matching the cumulative decoded text. Empirically this tokenizer/template does NOT
open a <|channel>thought block for GSM8K; the model is prompted to end its chain-of-thought
with '#### ' before the numeric answer, so the operative boundary is the '#### ' delimiter
the model itself emits (verified in smoke). Thinking-channel closes ('<channel|>','</think>')
are also honored if a think block ever appears. Not oracle (it is the model's own emitted
delimiter, computed from generated tokens, never from the gold label).

Arms (all delta-norm-matched to uniform-math per token), GSM8K n=80 real EM:
  base               : no adapter (REFERENCE, F#866 underpower guard)
  uniform-math       : full delta every token (ceiling)
  head-only-always   : delta_head renormalized every token
  tail-only-always   : delta_tail renormalized every token
  schedule           : tail @ reasoning, head @ answer-emit (HYPOTHESIS)
  swap               : head @ reasoning, tail @ answer-emit (control)

KILL 2309 (pre-registered, both clauses):
  "Best of {head@answer/tail@reason, its swap} fails to beat uniform-math-solo by >=5pp
   GSM8K EM (n=80), OR any static rank-half arm already matches it (timing irrelevant)"
  clause A: best_schedule_EM - uniform_math_EM >= +0.05
  clause B: max(head_only_EM, tail_only_EM) < uniform_math_EM  (strict)
  SUPPORTED iff A and B. KILLED otherwise.

Composition is Sum_i(B_i @ A_i) low-rank; LORA_SCALE=6.0 <= 8. NO MOCKS. is_smoke=False.
"""

import gc
import json
import os
import re
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

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
MATH_ADAPTER = Path("/Users/tom/Code/tomsiwik/llm/data/adapters/math/adapters.safetensors")

LORA_SCALE = 6.0
LORA_RANK = 6
N_HEAD = (LORA_RANK + 1) // 2      # 3
N_TAIL = LORA_RANK - N_HEAD        # 3
N_LAYERS = 42
N_GSM8K = 80
MAX_NEW_TOKENS = 1024
SEED = 42
EPS = 1e-8
NORM_TOL = 1e-3                    # relative tol for magnitude-match assertion
NORM_FLOOR = 1e-4                  # skip assertion when ||delta_full|| below this

# phase codes
P_OFF = 0      # no injection
P_FULL = 1     # full delta
P_HEAD = 2     # head subspace (renormed)
P_TAIL = 3     # tail subspace (renormed)

# Decode-time reasoning->answer-emit delimiters, detected from GENERATED tokens (not oracle).
# Empirically (this tokenizer/template) gemma-4 does NOT open a <|channel>thought block for
# GSM8K; the model is prompted to end its chain-of-thought with '#### ' before the numeric
# answer, so the operative reasoning->answer-emit boundary is the '#### ' marker the model
# itself emits. We also honor the thinking-channel close delimiters if a think block appears.
ANSWER_DELIMS = ["#### ", "####", "<channel|>", "<|think|>", "</think>", "<|channel|>"]


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


class Ctrl:
    """Shared controller. `phase` selects which subspace is injected this token.

    norm_violations accumulates any magnitude-match breaches (must stay 0).
    norm_checks counts asserted tokens. max_rel_err tracks worst relative norm error.
    """
    def __init__(self):
        self.phase = P_FULL
        self.norm_violations = 0
        self.norm_checks = 0
        self.max_rel_err = 0.0


class RankPhaseLoRALinear(nn.Module):
    """q_proj = W_q x + delta_inject(x) where delta depends on ctrl.phase.

    Precomputes U (out x r) = left singular vectors of Delta = s*B@A.
    delta_full(x) = (s * B A) x  (in span(U)).
    For a sub-rank phase, project delta_full onto chosen columns of U, then renorm to
    ||delta_full|| per token (magnitude-match). Asserts the invariant.
    """
    def __init__(self, base_linear, a, b, scale, ctrl, layer_idx):
        super().__init__()
        self.linear = base_linear
        self.scale = scale
        self._ctrl = ctrl
        self._li = layer_idx
        # a: (in, r)  b: (r, out)   delta = s * (x @ a) @ b  => W_eff = s * a @ b (in x out)
        self.a = a.astype(mx.float32)
        self.b = b.astype(mx.float32)
        # Delta in out-space row convention: for x (..,in), delta = scale*(x@a)@b -> (..,out)
        # Build the out x in matrix M s.t. delta = (M @ x_col); M = scale * b^T @ a^T (out x in)
        M = (scale * (self.b.T @ self.a.T))            # (out, in)
        U, S, Vt = mx.linalg.svd(M, stream=mx.cpu)     # U:(out,out) S:(min,) Vt:(in,in)
        r = self.a.shape[1]
        self.U_head = U[:, :N_HEAD]                    # (out, n_head)
        self.U_tail = U[:, N_HEAD:r]                   # (out, n_tail)
        mx.eval(self.U_head, self.U_tail)
        self.linear.freeze()

    def _delta_full(self, x):
        z = (x @ self.a) @ self.b                      # (.., out)
        return (self.scale * z).astype(mx.float32)

    def _project_renorm(self, delta_full, Usub):
        # delta_full: (B,T,out); Usub: (out, k)
        coeff = delta_full @ Usub                      # (B,T,k)
        delta_s = coeff @ Usub.T                        # (B,T,out)
        nfull = mx.sqrt(mx.sum(delta_full * delta_full, axis=-1, keepdims=True))  # (B,T,1)
        ns = mx.sqrt(mx.sum(delta_s * delta_s, axis=-1, keepdims=True))
        scaled = delta_s * (nfull / (ns + EPS))
        return scaled, nfull, scaled

    def __call__(self, x):
        y = self.linear(x)
        phase = self._ctrl.phase
        if phase == P_OFF:
            return y
        xf = x.astype(mx.float32)
        delta_full = self._delta_full(xf)
        if phase == P_FULL:
            inject = delta_full
            nfull = mx.sqrt(mx.sum(delta_full * delta_full, axis=-1, keepdims=True))
            ninj = nfull
        elif phase == P_HEAD:
            inject, nfull, _ = self._project_renorm(delta_full, self.U_head)
            ninj = mx.sqrt(mx.sum(inject * inject, axis=-1, keepdims=True))
        elif phase == P_TAIL:
            inject, nfull, _ = self._project_renorm(delta_full, self.U_tail)
            ninj = mx.sqrt(mx.sum(inject * inject, axis=-1, keepdims=True))
        else:
            return y
        # magnitude-match assertion (only where full norm is meaningful)
        nf = nfull.reshape(-1)
        ni = ninj.reshape(-1)
        mx.eval(nf, ni)
        nf_l = nf.tolist(); ni_l = ni.tolist()
        for a_, b_ in zip(nf_l, ni_l):
            if a_ > NORM_FLOOR:
                rel = abs(b_ - a_) / a_
                self._ctrl.norm_checks += 1
                if rel > self._ctrl.max_rel_err:
                    self._ctrl.max_rel_err = rel
                if rel > NORM_TOL:
                    self._ctrl.norm_violations += 1
        return y + inject.astype(y.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def attach(model, math_adapter, ctrl):
    lm = get_lm(model)
    count = 0
    for li, layer in enumerate(lm.model.layers):
        base_linear = layer.self_attn.q_proj
        ka = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        kb = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        a = math_adapter[ka]
        b = math_adapter[kb]
        wrapper = RankPhaseLoRALinear(base_linear, a, b, LORA_SCALE, ctrl, li)
        setattr(layer.self_attn, "q_proj", wrapper)
        count += 1
    mx.eval(model.parameters())
    assert count == N_LAYERS, f"expected {N_LAYERS} layers, got {count}"
    log(f"  Attached {count} RankPhaseLoRALinear (head={N_HEAD} tail={N_TAIL})")
    return model


def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def gsm8k_prompt(p):
    return ("Solve this math problem step by step. End with '#### ' followed by "
            "the final numeric answer.\n\n" + p["question"])


def strip_thinking(text):
    if not text:
        return text
    text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|think\|>.*?<\|think\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_gsm8k(text):
    text = strip_thinking(text)
    for pat in (r"####\s*([\-\d,]+(?:\.\d+)?)",
                r"(?:answer\s*(?:is|:)?\s*\$?)([\-\d,]+(?:\.\d+)?)",
                r"([\-\d,]+(?:\.\d+)?)"):
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            try:
                return float(m[-1].replace(",", ""))
            except ValueError:
                continue
    return None


def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    probs = []
    for i in range(min(n, len(ds))):
        it = ds[i]
        m = re.search(r"####\s*([\-\d,]+(?:\.\d+)?)", it["answer"])
        ans = float(m.group(1).replace(",", "")) if m else None
        probs.append({"question": it["question"], "answer_num": ans})
    log(f"  Loaded {len(probs)} GSM8K problems")
    return probs


def _phase_at_token(in_reasoning, arm):
    """Return injection phase code given current phase (reasoning vs answer) and arm."""
    if arm == "base":
        return P_OFF
    if arm == "uniform-math":
        return P_FULL
    if arm == "head-only-always":
        return P_HEAD
    if arm == "tail-only-always":
        return P_TAIL
    if arm == "schedule":   # tail @ reasoning, head @ answer
        return P_TAIL if in_reasoning else P_HEAD
    if arm == "swap":       # head @ reasoning, tail @ answer
        return P_HEAD if in_reasoning else P_TAIL
    raise ValueError(arm)


def generate_arm(model, tokenizer, ctrl, prompt, arm, max_new=MAX_NEW_TOKENS):
    """Greedy decode. Reasoning phase detected at decode time by close-delimiter in text."""
    ids = mx.array(tokenizer.encode(prompt))
    cache = make_prompt_cache(model)
    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eot = eot_enc[-1] if eot_enc else eos

    in_reasoning = True       # chain-of-thought until the model emits the answer delimiter
    decoded = ""
    boundary_tok = None       # token index where reasoning ended

    # prefill
    ctrl.phase = _phase_at_token(in_reasoning, arm)
    logits = model(ids[None], cache=cache)[:, -1, :]
    tok = mx.argmax(logits, axis=-1)
    mx.eval(tok)
    out = [tok.item()]
    decoded = tokenizer.decode(out)
    if in_reasoning and any(d in decoded for d in ANSWER_DELIMS):
        in_reasoning = False
        boundary_tok = 0

    for step in range(max_new - 1):
        if out[-1] in (eos, eot):
            break
        ctrl.phase = _phase_at_token(in_reasoning, arm)
        logits = model(mx.array([[out[-1]]]), cache=cache)[:, -1, :]
        tok = mx.argmax(logits, axis=-1)
        mx.eval(tok)
        out.append(tok.item())
        decoded = tokenizer.decode(out)
        if in_reasoning and any(d in decoded for d in ANSWER_DELIMS):
            in_reasoning = False
            boundary_tok = step + 1
    del cache
    return decoded, len(out), boundary_tok


def eval_arm(model, tokenizer, ctrl, problems, arm):
    correct, details = 0, []
    n_boundary_found = 0
    for p in problems:
        prompt = format_chat(tokenizer, gsm8k_prompt(p))
        text, ntok, btok = generate_arm(model, tokenizer, ctrl, prompt, arm)
        pred = extract_gsm8k(text)
        exp = p["answer_num"]
        ok = pred is not None and exp is not None and abs(pred - exp) < 1e-2
        correct += int(ok)
        if btok is not None:
            n_boundary_found += 1
        details.append({"pred": pred, "exp": exp, "passed": ok, "ntok": ntok,
                        "boundary_tok": btok})
    acc = correct / len(problems)
    log(f"    [{arm}] GSM8K EM = {acc:.4f} ({correct}/{len(problems)})  "
        f"boundary_detected={n_boundary_found}/{len(problems)}")
    return acc, details, n_boundary_found


ARMS = ["base", "uniform-math", "head-only-always", "tail-only-always", "schedule", "swap"]


def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 70)
    log("exp_spark_intra_rank_phase_gate")
    log(f"Base: {MODEL_ID}")
    log(f"math={MATH_ADAPTER}")
    log(f"n_gsm8k={N_GSM8K} scale={LORA_SCALE} rank={LORA_RANK} head={N_HEAD} tail={N_TAIL}")
    log("=" * 70)
    assert MATH_ADAPTER.exists(), f"missing {MATH_ADAPTER}"
    log_mem("start")

    log("\n=== PHASE 0: data + adapter ===")
    gsm8k = load_gsm8k(N_GSM8K)
    math_ad = mx.load(str(MATH_ADAPTER))

    em = {}
    det = {}
    boundary = {}

    model, tok = load(MODEL_ID)
    ctrl = Ctrl()
    attach(model, math_ad, ctrl)

    for arm in ARMS:
        log(f"\n=== ARM: {arm} ===")
        em[arm], det[arm], boundary[arm] = eval_arm(model, tok, ctrl, gsm8k, arm)
        log_mem(f"{arm}-done")

    norm_violations = ctrl.norm_violations
    norm_checks = ctrl.norm_checks
    max_rel_err = ctrl.max_rel_err
    log(f"\n[MAGNITUDE-MATCH] checks={norm_checks} violations={norm_violations} "
        f"max_rel_err={max_rel_err:.2e} (tol={NORM_TOL})")
    assert norm_violations == 0, (
        f"MAGNITUDE-MATCH VIOLATED: {norm_violations}/{norm_checks} tokens exceeded "
        f"rel tol {NORM_TOL} (max_rel_err={max_rel_err:.3e})")

    del model, ctrl, math_ad
    gc.collect(); mx.clear_cache()

    # ---- Kill criteria 2309 ----
    base_em = em["base"]
    uniform = em["uniform-math"]
    head_only = em["head-only-always"]
    tail_only = em["tail-only-always"]
    best_schedule = max(em["schedule"], em["swap"])
    best_schedule_arm = "schedule" if em["schedule"] >= em["swap"] else "swap"

    timing_win_pp = (best_schedule - uniform) * 100.0
    cond_A = (best_schedule - uniform) >= 0.05
    cond_B = max(head_only, tail_only) < uniform        # strict: a match kills
    all_pass = bool(cond_A and cond_B)
    verdict = "SUPPORTED" if all_pass else "KILLED"

    # underpower guard reporting (F#871)
    uniform_vs_base_pp = (uniform - base_em) * 100.0
    win_over_base = best_schedule > base_em

    log("\n" + "=" * 70)
    log("KILL CRITERIA 2309 (pre-registered, both clauses)")
    log("=" * 70)
    log(f"  GSM8K EM: base={base_em:.4f} uniform-math={uniform:.4f} "
        f"head-only={head_only:.4f} tail-only={tail_only:.4f}")
    log(f"            schedule={em['schedule']:.4f} swap={em['swap']:.4f} "
        f"-> best={best_schedule:.4f} ({best_schedule_arm})")
    log(f"  UNDERPOWER GUARD (F#871): uniform-math vs base = {uniform_vs_base_pp:+.1f}pp; "
        f"best_schedule {'>' if win_over_base else '<='} base")
    log(f"  clause A (timing win >=5pp): best_schedule - uniform = {timing_win_pp:+.1f}pp "
        f"-> {'PASS' if cond_A else 'FAIL'}")
    log(f"  clause B (no static half matches): max(head,tail)={max(head_only,tail_only):.4f} "
        f"< uniform={uniform:.4f} -> {'PASS' if cond_B else 'FAIL'}")
    log(f"  VERDICT: {verdict}")
    if all_pass and not win_over_base:
        log("  WARNING: schedule beats uniform but is <= base; win may be recovering "
            "self-inflicted adapter damage (F#866).")

    results = {
        "experiment": "exp_spark_intra_rank_phase_gate",
        "model": MODEL_ID,
        "math_adapter": str(MATH_ADAPTER),
        "lora_scale": LORA_SCALE,
        "lora_rank": LORA_RANK,
        "n_head": N_HEAD,
        "n_tail": N_TAIL,
        "n_layers": N_LAYERS,
        "n_gsm8k": N_GSM8K,
        "enable_thinking": True,
        "greedy": True,
        "is_smoke": False,
        "magnitude_match": {
            "checks": norm_checks,
            "violations": norm_violations,
            "max_rel_err": max_rel_err,
            "tol": NORM_TOL,
            "invariant": "||delta_inject|| == ||delta_full|| per token (renormed)",
        },
        "boundary_detection": {
            "delimiters": ANSWER_DELIMS,
            "n_boundary_found_per_arm": boundary,
            "method": ("decode-time string match of answer-emit delimiter ('#### ' that the "
                       "model is prompted to and does emit, plus thinking-channel closes) in "
                       "cumulative generated text; not oracle"),
        },
        "metrics": {
            "gsm8k_em": {
                "base": base_em,
                "uniform-math": uniform,
                "head-only-always": head_only,
                "tail-only-always": tail_only,
                "schedule": em["schedule"],
                "swap": em["swap"],
            },
            "best_schedule_em": best_schedule,
            "best_schedule_arm": best_schedule_arm,
            "timing_win_vs_uniform_pp": timing_win_pp,
            "uniform_vs_base_pp": uniform_vs_base_pp,
            "best_schedule_beats_base": bool(win_over_base),
        },
        "kill_criteria": {
            "id": 2309,
            "text": ("Best of {head@answer/tail@reason, its swap} fails to beat "
                     "uniform-math-solo by >=5pp GSM8K EM (n=80), OR any static rank-half "
                     "arm already matches it (timing irrelevant)"),
            "metric": "GSM8K EM n=80",
            "clause_A_timing_win_pp": timing_win_pp,
            "clause_A_threshold_pp": 5.0,
            "clause_A_pass": bool(cond_A),
            "clause_B_max_static_half": max(head_only, tail_only),
            "clause_B_pass": bool(cond_B),
            "pass": bool(all_pass),
        },
        "all_pass": all_pass,
        "verdict": verdict,
        "total_time_s": round(time.time() - t0, 1),
        "details": det,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    log(f"\nResults -> {RESULTS_FILE}")
    log(f"Total time: {results['total_time_s']}s")
    log(f"FINAL VERDICT: {verdict}")


if __name__ == "__main__":
    main()
