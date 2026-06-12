#!/usr/bin/env python3
"""exp_jury_r2_answer_class_jury — 3-adapter decorrelated jury vs SC(8) in answer-class space.

BET jury-decode R2 reframed as fresh first rung (R1 selector class closed: F#877 K2316, F#879 K2332).
MECHANISM: no juror ever selects a chain. Each of 3 frozen adapters (math/python/medical, r=6 q_proj,
scale 6 on frozen gemma-4-e4b-it-4bit) scores the 8 chains per question with a judge probe
s = logP(Yes) - logP(No); per question scores are standardized (zero mean, unit std — no free
temperature) and softmaxed; vote mass per answer-equivalence class = sum of softmax weights; jurors
combine log-linearly; argmax class. SC's vote structure is kept and only MODULATED.

R1 did not cache chain texts, so the IDENTICAL 8 chains per question are regenerated with the same
seed schedule / model / adapter / mlx_lm 0.31.2 and validated against R1's cached preds. SC is
recomputed on the regenerated chains as the baseline (pre-registered in MATH.md).

KILL K2333 (pre-registered):
  killed if jury_acc < acc_sc + 0.02, OR mean pairwise juror error-overlap kappa >= math-juror
  bootstrap split-half self-kappa (shared frozen base correlates failures).
SUPPORTED only if jury_acc >= max(acc_sc, best single-juror weighted SC) + 0.02 AND overlap passes
AND math-only control does NOT clear acc_sc + 0.02 (else provisional: weighting, not decorrelation).

NO MOCKS. Real model, real adapters, real GSM8K. is_smoke=False.
Wrapper attaches via subclass nn.Module + setattr (never __call__ override — F#831).
"""

import gc
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

device_info = mx.device_info()
mx.set_memory_limit(device_info["memory_size"] - 6 * 1024**3)

EXP_DIR = Path(__file__).resolve().parent
RESULTS_FILE = EXP_DIR / "results.json"
CHAINS_CACHE = EXP_DIR / "chains_cache.json"
R1_RESULTS = EXP_DIR.parent / "exp_bet_jury_r1_verifier_gain" / "results.json"
ADAPTER_DIR = EXP_DIR.parent.parent.parent / "data" / "adapters"

MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
JURORS = ["math", "python", "medical"]

LORA_SCALE = 6.0          # <= 8 guard
LORA_RANK = 6
N_LAYERS_EXPECTED = 42
N_GSM8K = int(os.environ.get("N_GSM8K", "200"))
N_SAMPLES = 8
TEMPERATURE = 0.8
MAX_NEW_TOKENS = 512
SEED = 42

GATE_GAIN = 0.02          # K2333: jury must clear acc_sc + 2pp
N_BOOTSTRAP = 1000


def log(msg):
    print(msg, flush=True)


def log_mem(label=""):
    log(f"[MEM {label}] active={mx.get_active_memory()/1e9:.2f}GB "
        f"cache={mx.get_cache_memory()/1e9:.2f}GB peak={mx.get_peak_memory()/1e9:.2f}GB")


# ----------------------------------------------------------------------------
# Adapter attach / hot-swap: y = base(x) + s * (x @ A) @ B  (q_proj only)
# ----------------------------------------------------------------------------

class LoRAQProj(nn.Module):
    def __init__(self, base_linear, a, b, scale):
        super().__init__()
        self.linear = base_linear
        self.lora_a = a
        self.lora_b = b
        self.scale = scale
        self.linear.freeze()

    def __call__(self, x):
        y = self.linear(x)
        delta = self.scale * ((x @ self.lora_a) @ self.lora_b)
        return y + delta.astype(x.dtype)


def get_lm(model):
    return model.language_model if hasattr(model, "language_model") else model


def load_adapter_weights(name):
    p = ADAPTER_DIR / name / "adapters.safetensors"
    assert p.exists(), f"missing adapter {p}"
    return mx.load(str(p))


def attach_adapter(model, ad):
    """First attach: wrap every q_proj. Returns list of wrappers for hot-swap."""
    lm = get_lm(model)
    wrappers = []
    for li, layer in enumerate(lm.model.layers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        assert ak in ad and bk in ad, f"adapter missing keys for layer {li}"
        a = ad[ak].astype(mx.float32)
        b = ad[bk].astype(mx.float32)
        assert a.shape[1] == LORA_RANK, f"rank mismatch: {a.shape}"
        w = LoRAQProj(layer.self_attn.q_proj, a, b, LORA_SCALE)
        setattr(layer.self_attn, "q_proj", w)
        wrappers.append(w)
    mx.eval(model.parameters())
    assert len(wrappers) == N_LAYERS_EXPECTED, f"expected {N_LAYERS_EXPECTED}, got {len(wrappers)}"
    return wrappers


def swap_adapter(model, wrappers, ad, name):
    """Hot-swap A/B matrices in existing wrappers (base stays frozen)."""
    for li, w in enumerate(wrappers):
        ak = f"language_model.model.layers.{li}.self_attn.q_proj.lora_a"
        bk = f"language_model.model.layers.{li}.self_attn.q_proj.lora_b"
        w.lora_a = ad[ak].astype(mx.float32)
        w.lora_b = ad[bk].astype(mx.float32)
    mx.eval(model.parameters())
    log(f"  Swapped to '{name}' adapter on {len(wrappers)} q_proj layers")


# ----------------------------------------------------------------------------
# Generation (no-thinking harness) — identical to R1 for bit-exact reproduction
# ----------------------------------------------------------------------------

def format_chat(tokenizer, content):
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )


def generate(model, tokenizer, prompt_ids, max_new, temperature, eos_ids):
    cache = make_prompt_cache(model)
    logits = model(prompt_ids[None], cache=cache)[:, -1, :]
    out = []
    for _ in range(max_new):
        logits32 = logits.astype(mx.float32)
        if temperature <= 0.0:
            tok = mx.argmax(logits32, axis=-1)
        else:
            tok = mx.random.categorical(logits32 / temperature)
        # NB: logprobs computed in R1 too before sampling; keep op order identical
        logprobs = logits32 - mx.logsumexp(logits32, axis=-1, keepdims=True)
        lp = logprobs[0, tok[0]]
        mx.eval(tok, lp)
        tid = tok.item()
        out.append(tid)
        if tid in eos_ids:
            break
        logits = model(mx.array([[tid]]), cache=cache)[:, -1, :]
    del cache
    return tokenizer.decode(out), len(out)


# ----------------------------------------------------------------------------
# Judge probe — identical template to R1
# ----------------------------------------------------------------------------

VERIFIER_TEMPLATE = (
    "Question:\n{q}\n\nProposed solution:\n{sol}\n\n"
    "Is the final answer of this proposed solution correct? "
    "Reply with exactly one word: Yes or No."
)


def juror_score(model, tokenizer, question, solution, yes_id, no_id):
    prompt = format_chat(tokenizer, VERIFIER_TEMPLATE.format(q=question, sol=solution))
    ids = mx.array(tokenizer.encode(prompt))
    logits = model(ids[None])[:, -1, :].astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    s = logprobs[0, yes_id] - logprobs[0, no_id]
    mx.eval(s)
    return float(s.item()), ids.shape[0]


# ----------------------------------------------------------------------------
# GSM8K — identical loading/parsing to R1 (same seed 42 shuffle -> same items)
# ----------------------------------------------------------------------------

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=SEED).select(range(min(n, len(ds))))
    items = [{"question": ds[i]["question"], "answer": ds[i]["answer"]} for i in range(len(ds))]
    log(f"  Loaded {len(items)} GSM8K problems")
    return items


def gsm8k_gt(answer):
    m = re.search(r"####\s*([\-\d,\.]+)", answer)
    return m.group(1).replace(",", "").strip() if m else None


def gsm8k_pred(text):
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return nums[-1] if nums else None


GSM8K_INSTR = "Solve this math problem step by step. End your answer with '#### <number>'.\n\n"


# ----------------------------------------------------------------------------
# Voting math
# ----------------------------------------------------------------------------

def softmax_standardized(scores):
    """Per-question: standardize scores (zero mean, unit std) then softmax. No free temperature."""
    n = len(scores)
    mu = sum(scores) / n
    var = sum((s - mu) ** 2 for s in scores) / n
    sd = var ** 0.5
    z = [(s - mu) / sd if sd > 1e-9 else 0.0 for s in scores]
    mz = max(z)
    e = [pow(2.718281828459045, zi - mz) for zi in z]
    tot = sum(e)
    return [ei / tot for ei in e]


def class_masses(preds, weights):
    """Answer-equivalence class mass from parseable chains only."""
    m = defaultdict(float)
    for p, w in zip(preds, weights):
        if p is not None:
            m[p] += w
    return m


import math as _math


def cohen_kappa(x, y):
    """Binary Cohen's kappa; nan if undefined (zero variance in either marginal)."""
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(1 for a, b in zip(x, y) if a == b) / n
    px1 = sum(x) / n
    py1 = sum(y) / n
    pe = px1 * py1 + (1 - px1) * (1 - py1)
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log("=" * 72)
    log("exp_jury_r2_answer_class_jury — 3-juror answer-class jury vs SC(8), GSM8K")
    log(f"Base: {MODEL_ID}  jurors: {JURORS}  N={N_GSM8K} x {N_SAMPLES} chains")
    log("=" * 72)
    assert R1_RESULTS.exists(), f"missing R1 artifacts {R1_RESULTS}"
    with open(R1_RESULTS) as f:
        r1 = json.load(f)
    r1_details = r1["details"]
    acc_sc_r1 = r1["accuracy"]["self_consistency_8"]
    log(f"  R1 cached: SC(8)={acc_sc_r1:.3f} BoN(8)={r1['accuracy']['bon_8_verifier']:.3f} "
        f"AUC={r1['verifier_auc']:.4f}")
    log_mem("start")

    model, tokenizer = load(MODEL_ID)
    adapters = {name: load_adapter_weights(name) for name in JURORS}
    wrappers = attach_adapter(model, adapters["math"])
    log(f"  Attached math LoRA (generator + juror 1), scale={LORA_SCALE} rank={LORA_RANK}")
    log_mem("model+adapter")

    eos = tokenizer.eos_token_id
    eot_enc = tokenizer.encode("<end_of_turn>")
    eos_ids = {eos, eot_enc[-1] if eot_enc else eos}
    yes_id = tokenizer.encode("Yes")[-1]
    no_id = tokenizer.encode("No")[-1]
    assert yes_id != no_id

    items = load_gsm8k(N_GSM8K)
    assert len(items) == len(r1_details), "R1/R2 item count mismatch"

    # ------------------------------------------------------------------
    # Phase 1: regenerate the 8 chains/question (math adapter, R1 seed
    # schedule) + math-juror scores inline. Restart-safe via chains_cache.
    # ------------------------------------------------------------------
    questions = []          # per q: gt, preds[8], oks[8], texts[8], scores{juror: [8]}
    prefill_tokens = {j: 0 for j in JURORS}
    gen_tokens = 0
    start_q = 0
    if CHAINS_CACHE.exists():
        with open(CHAINS_CACHE) as f:
            cached = json.load(f)
        questions = cached["questions"]
        prefill_tokens["math"] = cached["math_prefill_tokens"]
        gen_tokens = cached["gen_tokens"]
        start_q = len(questions)
        log(f"  Resuming: {start_q} questions loaded from chains_cache.json")

    repro_match = sum(
        1 for qi in range(start_q) for si in range(N_SAMPLES)
        if questions[qi]["preds"][si] == r1_details[qi]["cands"][si]["pred"]
    )
    for qi in range(start_q, len(items)):
        it = items[qi]
        gt = gsm8k_gt(it["answer"])
        prompt = format_chat(tokenizer, GSM8K_INSTR + it["question"])
        prompt_ids = mx.array(tokenizer.encode(prompt))
        texts, preds, oks, mscores = [], [], [], []
        for si in range(N_SAMPLES):
            mx.random.seed(SEED * 100003 + qi * 1009 + si + 1)   # R1 schedule, bit-exact
            text, ntok = generate(model, tokenizer, prompt_ids, MAX_NEW_TOKENS,
                                  TEMPERATURE, eos_ids)
            gen_tokens += ntok
            pred = gsm8k_pred(text)
            ok = (gt is not None and pred is not None and pred == gt)
            s, ptok = juror_score(model, tokenizer, it["question"], text.strip(),
                                  yes_id, no_id)
            prefill_tokens["math"] += ptok
            texts.append(text)
            preds.append(pred)
            oks.append(ok)
            mscores.append(s)
            repro_match += int(pred == r1_details[qi]["cands"][si]["pred"])
        questions.append({"gt": gt, "question": it["question"], "texts": texts,
                          "preds": preds, "oks": oks, "scores": {"math": mscores}})
        if (qi + 1) % 10 == 0:
            done = (qi + 1) * N_SAMPLES
            log(f"  [gen {qi+1}/{len(items)}] repro={repro_match/done:.3f} "
                f"({time.time()-t0:.0f}s)")
            mx.clear_cache()
            with open(CHAINS_CACHE, "w") as f:
                json.dump({"questions": questions, "gen_tokens": gen_tokens,
                           "math_prefill_tokens": prefill_tokens["math"]}, f)
    with open(CHAINS_CACHE, "w") as f:
        json.dump({"questions": questions, "gen_tokens": gen_tokens,
                   "math_prefill_tokens": prefill_tokens["math"]}, f)
    repro_rate = repro_match / (len(items) * N_SAMPLES)
    log(f"  Phase 1 done. Chain-pred reproduction vs R1 cache: {repro_rate:.3f}")
    mx.clear_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Phases 2-3: hot-swap juror adapters, prefill-only scoring of the
    # SAME chains. Zero generated tokens.
    # ------------------------------------------------------------------
    for juror in ["python", "medical"]:
        if all(juror in q["scores"] for q in questions):
            log(f"  Juror '{juror}' already scored (cache)")
            continue
        swap_adapter(model, wrappers, adapters[juror], juror)
        for qi, q in enumerate(questions):
            if juror in q["scores"]:
                continue
            ss = []
            for text in q["texts"]:
                s, ptok = juror_score(model, tokenizer, q["question"], text.strip(),
                                      yes_id, no_id)
                prefill_tokens[juror] += ptok
                ss.append(s)
            q["scores"][juror] = ss
            if (qi + 1) % 25 == 0:
                log(f"  [{juror} {qi+1}/{len(questions)}] ({time.time()-t0:.0f}s)")
                mx.clear_cache()
        with open(CHAINS_CACHE, "w") as f:
            json.dump({"questions": questions, "gen_tokens": gen_tokens,
                       "math_prefill_tokens": prefill_tokens["math"]}, f)
    mx.clear_cache()
    gc.collect()

    # ------------------------------------------------------------------
    # Phase 4: voting arms + overlap analysis (pure CPU)
    # ------------------------------------------------------------------
    n = len(questions)
    n_sc = 0
    n_single = {j: 0 for j in JURORS}
    n_jury = 0
    n_pass8 = 0
    per_q = []
    for q in questions:
        gt, preds, oks = q["gt"], q["preds"], q["oks"]
        n_pass8 += int(any(oks))

        votes = Counter(p for p in preds if p is not None)
        sc_pred = votes.most_common(1)[0][0] if votes else None
        sc_ok = (gt is not None and sc_pred == gt)
        n_sc += int(sc_ok)

        weights = {j: softmax_standardized(q["scores"][j]) for j in JURORS}
        masses = {j: class_masses(preds, weights[j]) for j in JURORS}

        single_pred = {}
        for j in JURORS:
            sp = max(masses[j], key=masses[j].get) if masses[j] else None
            single_pred[j] = sp
            n_single[j] += int(gt is not None and sp == gt)

        classes = set(masses["math"])
        jury_pred = None
        if classes:
            jury_pred = max(classes,
                            key=lambda a: sum(_math.log(masses[j][a]) for j in JURORS))
        jury_ok = (gt is not None and jury_pred == gt)
        n_jury += int(jury_ok)

        per_q.append({"gt": gt, "preds": preds, "oks": oks,
                      "scores": q["scores"], "sc_pred": sc_pred, "sc_ok": sc_ok,
                      "single_pred": single_pred, "jury_pred": jury_pred,
                      "jury_ok": jury_ok})

    acc_sc = n_sc / n
    acc_jury = n_jury / n
    acc_single = {j: n_single[j] / n for j in JURORS}
    pass_at_8 = n_pass8 / n
    best_single = max(acc_single.values())

    # --- overlap: split-half misranking indicators on mixed questions ---
    halves = [list(range(0, 4)), list(range(4, 8))]

    def half_indicator(q, juror, half):
        """1 iff juror's top-scored parseable chain in this half is wrong; None if half not mixed."""
        idx = [i for i in half if q["preds"][i] is not None]
        if not idx:
            return None
        cor = [i for i in idx if q["oks"][i]]
        wro = [i for i in idx if not q["oks"][i]]
        if not cor or not wro:
            return None
        top = max(idx, key=lambda i: q["scores"][juror][i])
        return 0 if q["oks"][top] else 1

    mixed = []
    for q in questions:
        inds = {(j, h): half_indicator(q, j, halves[h]) for j in JURORS for h in (0, 1)}
        if all(v is not None for v in inds.values()):
            mixed.append(inds)
    log(f"  Mixed questions (both halves contain correct+wrong): {len(mixed)}")

    def pair_kappa(j, k):
        k1 = cohen_kappa([m[(j, 0)] for m in mixed], [m[(k, 1)] for m in mixed])
        k2 = cohen_kappa([m[(j, 1)] for m in mixed], [m[(k, 0)] for m in mixed])
        vals = [v for v in (k1, k2) if not _math.isnan(v)]
        return sum(vals) / len(vals) if vals else 0.0   # nan -> 0 (pre-registered)

    pairwise = {f"{a}|{b}": pair_kappa(a, b)
                for a, b in [("math", "python"), ("math", "medical"), ("python", "medical")]}
    mean_pairwise_kappa = sum(pairwise.values()) / len(pairwise)

    rng = random.Random(SEED)
    boots = []
    for _ in range(N_BOOTSTRAP):
        sample = [mixed[rng.randrange(len(mixed))] for _ in range(len(mixed))]
        k1 = cohen_kappa([m[("math", 0)] for m in sample], [m[("math", 1)] for m in sample])
        if not _math.isnan(k1):
            boots.append(k1)
    self_kappa = sum(boots) / len(boots) if boots else float("nan")

    # ------------------------------------------------------------------
    # Verdict (pre-registered K2333, MATH.md)
    # ------------------------------------------------------------------
    gain_jury = acc_jury - acc_sc
    overlap_kill = (not _math.isnan(self_kappa)) and mean_pairwise_kappa >= self_kappa
    if _math.isnan(self_kappa):
        overlap_kill = True   # cannot demonstrate decorrelation -> conservative kill
    acc_kill = acc_jury < acc_sc + GATE_GAIN
    control_pass = acc_single["math"] >= acc_sc + GATE_GAIN

    if acc_kill or overlap_kill:
        verdict, all_pass = "killed", False
    elif acc_jury >= max(acc_sc, best_single) + GATE_GAIN and not control_pass:
        verdict, all_pass = "supported", True
    else:
        verdict, all_pass = "provisional", False

    results = {
        "experiment_id": "exp_jury_r2_answer_class_jury",
        "config": {
            "base_model": MODEL_ID,
            "jurors": JURORS,
            "adapter_dir": str(ADAPTER_DIR),
            "lora_scale": LORA_SCALE, "lora_rank": LORA_RANK,
            "n_gsm8k": n, "n_samples": N_SAMPLES, "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED,
            "no_thinking_harness": True,
            "weighting": "per-question standardized scores -> softmax (no free temperature)",
            "combination": "log-linear over answer-class masses, argmax class",
            "chains": "regenerated bit-exact via R1 seed schedule (R1 cached preds only)",
            "mlx_lm": "0.31.2",
        },
        "chain_reproduction_vs_r1": repro_rate,
        "accuracy": {
            "self_consistency_8": acc_sc,
            "self_consistency_8_r1_reference": acc_sc_r1,
            "jury_3_weighted": acc_jury,
            "single_weighted": acc_single,
            "best_single_weighted": best_single,
            "pass_at_8_ceiling": pass_at_8,
        },
        "gain_jury_minus_sc": gain_jury,
        "control_math_only_clears_gate": control_pass,
        "overlap": {
            "n_mixed_questions": len(mixed),
            "pairwise_kappa": pairwise,
            "mean_pairwise_kappa": mean_pairwise_kappa,
            "self_kappa_math_bootstrap": self_kappa,
            "n_bootstrap": N_BOOTSTRAP,
        },
        "kill_criteria": {
            "2333": {
                "text": "jury < SC+2pp OR pairwise juror error-overlap >= single-verifier "
                        "bootstrap self-overlap",
                "jury_acc": acc_jury, "acc_sc": acc_sc, "gate_gain": GATE_GAIN,
                "acc_kill": acc_kill,
                "mean_pairwise_kappa": mean_pairwise_kappa,
                "self_kappa": self_kappa, "overlap_kill": overlap_kill,
                "result": "fail" if (acc_kill or overlap_kill) else "pass",
            },
        },
        "token_budget": {
            "candidates_generated": gen_tokens,
            "juror_prefill_tokens": prefill_tokens,
            "note": "Jury and SC share the identical 8 chains -> equal generation budget; "
                    "jurors add prefill-only cost, zero generated tokens.",
        },
        "verdict": verdict,
        "all_pass": all_pass,
        "is_smoke": False,
        "details": per_q,
        "total_wall_clock_sec": time.time() - t0,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    log("\n" + "=" * 72)
    log(f"SC(8)={acc_sc:.3f} (R1 ref {acc_sc_r1:.3f}, chain repro {repro_rate:.3f})")
    log(f"jury={acc_jury:.3f} ({gain_jury:+.3f} vs SC, gate >= +{GATE_GAIN})  "
        f"singles: " + " ".join(f"{j}={acc_single[j]:.3f}" for j in JURORS))
    log(f"overlap: mean pairwise kappa={mean_pairwise_kappa:.3f} vs self={self_kappa:.3f} "
        f"-> {'KILL' if overlap_kill else 'pass'}  control_math_clears_gate={control_pass}")
    log(f"VERDICT: {verdict}  all_pass={all_pass}")
    log(f"Wrote {RESULTS_FILE}  ({time.time()-t0:.0f}s)")

    del model
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
