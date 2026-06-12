#!/usr/bin/env python3
"""exp_spark_drafter_disposes — adapter-as-drafter + frozen-base verifier.

Frame-break: never put off-domain adapter weight in the OUTPUT path. The frozen
4-bit base is the LOSSLESS speculative-decode VERIFIER; a domain q_proj LoRA is
only the cheap DRAFTER. Under the exact greedy acceptance rule (accept drafted
token iff it equals the base-greedy argmax at that position, else reject and emit
the base argmax), verified output == base-only greedy for ANY drafter (MATH.md
Theorem 1). So off-domain (medical) drafting gives EXACTLY 0pp accuracy delta
(K1). Drafter choice changes only acceptance length (K2) / speed (K3).

REAL MLX, NO mocks. K1 is a MEASURED correctness property of the decoding loop,
never asserted/hardcoded: we independently compute base-only greedy and compare
the verified sequence token-by-token.

Conditions: math drafter, medical drafter, null random-logit drafter (Gaussian
noise magnitude-matched IN-RUN to the math adapter's mean logit-shift), and the
base-only greedy reference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from safetensors import safe_open

EXPERIMENT_DIR = Path(__file__).parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
RESULTS_FILE = EXPERIMENT_DIR / "results.json"

MODEL = "mlx-community/gemma-4-e4b-it-4bit"
MATH_ADAPTER = REPO_ROOT / "data" / "adapters" / "math" / "adapters.safetensors"
MED_ADAPTER = REPO_ROOT / "data" / "adapters" / "medical" / "adapters.safetensors"
GSM8K = REPO_ROOT / "experiments" / "models" / "exp_p9_ttlora_polar_hybrid" / "data" / "gsm8k_test.jsonl"

N_ITEMS = 200
MAX_NEW = 96          # tokens to generate per item
BLOCK = 4             # speculative draft block size k
LORA_R = 6
LORA_SCALE = 6.0      # adapter-trained scale, <= 8 per guardrail
SEED = 42

# ---------------------------------------------------------------------------
# Model / adapter plumbing
# ---------------------------------------------------------------------------

def lm_layers(model):
    lm = model.language_model if hasattr(model, "language_model") else model
    return lm.model.layers


def load_adapter_tensors(path: Path):
    """Return {layer_idx: (lora_a, lora_b)} for q_proj from a saved adapter."""
    out = {}
    with safe_open(str(path), "np") as f:
        keys = list(f.keys())
        for k in keys:
            if "self_attn.q_proj.lora_a" not in k:
                continue
            li = int(k.split(".layers.")[1].split(".")[0])
            a = mx.array(f.get_tensor(k))
            b = mx.array(f.get_tensor(k.replace("lora_a", "lora_b")))
            out[li] = (a, b)
    return out


def capture_base_qprojs(model):
    """Save the original q_proj (QuantizedLinear) modules — the verifier path."""
    base = {}
    for i, layer in enumerate(lm_layers(model)):
        base[i] = layer.self_attn.q_proj
    return base


def install_base(model, base_qprojs):
    """Restore frozen base q_proj on every layer (verifier configuration)."""
    for i, layer in enumerate(lm_layers(model)):
        setattr(layer.self_attn, "q_proj", base_qprojs[i])


def install_lora_drafter(model, base_qprojs, adapter):
    """setattr submodule replacement: wrap each base q_proj in a LoRALinear
    carrying this adapter's weights. NOT __call__-on-instance."""
    for i, layer in enumerate(lm_layers(model)):
        base_lin = base_qprojs[i]
        lora = LoRALinear.from_base(base_lin, r=LORA_R, dropout=0.0, scale=LORA_SCALE)
        a, b = adapter[i]
        lora.lora_a = a.astype(mx.float32)
        lora.lora_b = b.astype(mx.float32)
        setattr(layer.self_attn, "q_proj", lora)


# ---------------------------------------------------------------------------
# Greedy helpers
# ---------------------------------------------------------------------------

def greedy_token(logits_row):
    # logits_row: (vocab,) -> scalar argmax (ties -> lowest index, mx default)
    return int(mx.argmax(logits_row).item())


def base_only_greedy(model, base_qprojs, prompt_ids, max_new, eos_ids):
    """Reference: pure base-greedy decode. Returns (tokens, wall_seconds)."""
    install_base(model, base_qprojs)
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(prompt_ids, cache=cache)[:, -1, :]
    tok = greedy_token(logits[0])
    out = [tok]
    for _ in range(max_new - 1):
        if tok in eos_ids:
            break
        logits = model(mx.array([[tok]]), cache=cache)[:, -1, :]
        tok = greedy_token(logits[0])
        out.append(tok)
    mx.eval(logits)
    return out, time.perf_counter() - t0


def draft_block_lora(model, base_qprojs, adapter, prefix_tok, cache_offset_ids, k, eos_ids):
    """Drafter greedy rollout of up to k tokens using a fresh drafter cache over
    the full running context. Returns list of drafted token ids.
    `cache_offset_ids` is the entire context (prompt + accepted) as a 1xT array."""
    install_lora_drafter(model, base_qprojs, adapter)
    dcache = make_prompt_cache(model)
    logits = model(cache_offset_ids, cache=dcache)[:, -1, :]
    drafts = []
    tok = greedy_token(logits[0])
    drafts.append(tok)
    for _ in range(k - 1):
        if tok in eos_ids:
            break
        logits = model(mx.array([[tok]]), cache=dcache)[:, -1, :]
        tok = greedy_token(logits[0])
        drafts.append(tok)
    mx.eval(logits)
    return drafts


def draft_block_null(base_logits_seq):
    """Null drafter handled inline in spec loop (needs base logits + noise)."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Lossless greedy speculative decode (verifier = frozen base)
# ---------------------------------------------------------------------------

def spec_decode(model, base_qprojs, adapter, prompt_ids, max_new, k, eos_ids,
                null_sigma=None):
    """Greedy lossless speculative decoding.

    adapter: dict {layer:(a,b)} for a LoRA drafter, OR None for the null drafter
    (in which case null_sigma is the per-token noise std and drafts come from the
    base logits + Gaussian noise).

    Returns (emitted_tokens, accept_lengths, wall_seconds).
    accept_lengths = number of drafted tokens accepted per verifier call.

    The verifier ALWAYS uses the frozen base (install_base before each verify),
    and we trim its KV cache back on rejection so emitted == base-greedy.
    """
    install_base(model, base_qprojs)
    vcache = make_prompt_cache(model)
    # context tracks prompt + all accepted tokens (1 x T), used to (re)seed drafter.
    context = prompt_ids
    # prefill verifier on prompt; last-pos logits give first base-greedy token.
    t0 = time.perf_counter()
    vlogits = model(prompt_ids, cache=vcache)[:, -1, :]  # (1,vocab) at first gen pos
    mx.eval(vlogits)

    emitted = []
    accept_lengths = []
    pending_base_logits = vlogits  # base-greedy logits for the NEXT position

    while len(emitted) < max_new:
        # ---- 1. produce a draft block of k tokens for upcoming positions ----
        if adapter is not None:
            drafts = draft_block_lora(model, base_qprojs, adapter, None, context, k, eos_ids)
            install_base(model, base_qprojs)  # restore verifier path
        else:
            # NULL: greedy over base logits + matched Gaussian noise.
            install_base(model, base_qprojs)
            dcache = make_prompt_cache(model)
            dl = model(context, cache=dcache)[:, -1, :]
            drafts = []
            noisy = dl + null_sigma * mx.random.normal(dl.shape)
            tok = greedy_token(noisy[0])
            drafts.append(tok)
            for _ in range(k - 1):
                if tok in eos_ids:
                    break
                dl = model(mx.array([[tok]]), cache=dcache)[:, -1, :]
                noisy = dl + null_sigma * mx.random.normal(dl.shape)
                tok = greedy_token(noisy[0])
                drafts.append(tok)
            mx.eval(dl)

        kk = len(drafts)

        # ---- 2. verify: run base over [g_for_pos0 already have] + drafts ----
        # pending_base_logits gives base-greedy for position 0 of this block.
        base_first = greedy_token(pending_base_logits[0])
        # feed the drafts to the verifier to get base-greedy at subsequent positions.
        # We feed drafts[0..kk-1] as a single forward; verifier cache currently sits
        # right before block position 0.
        draft_arr = mx.array([drafts])  # (1,kk)
        vblock = model(draft_arr, cache=vcache)  # (1,kk,vocab)
        mx.eval(vblock)
        # vblock[:, i, :] = base-greedy logits for position (i+1) of this block,
        # i.e. conditioned on prefix + drafts[0..i].
        # base-greedy token at block position 0 is base_first (from pending logits);
        # at position i>=1 it is argmax(vblock[:, i-1, :]).

        # ---- 3. left-to-right accept ----
        n_accept = 0
        block_emitted = []
        verifier_consumed = 0  # how many of the kk forwarded tokens we keep in cache
        for i in range(kk):
            base_tok = base_first if i == 0 else greedy_token(vblock[0, i - 1, :])
            if drafts[i] == base_tok and base_tok not in eos_ids:
                block_emitted.append(base_tok)
                n_accept += 1
                verifier_consumed += 1
                if len(emitted) + len(block_emitted) >= max_new:
                    break
            else:
                # reject: emit base_tok (the correct base-greedy token), stop.
                block_emitted.append(base_tok)
                # we KEEP this position's draft slot? No: drafts[i] != base_tok,
                # so the cache entry for drafts[i] is wrong. Keep only the i
                # accepted draft positions; the emitted base_tok occupies the
                # (i)th block slot, which equals drafts[0..i-1] accepted +
                # this corrected token. We trim the verifier cache to drop the
                # kk-i mis-forwarded draft tokens, then re-advance with base_tok.
                break
        else:
            # all kk drafts accepted -> bonus token from last vblock position
            bonus = greedy_token(vblock[0, kk - 1, :])
            if bonus not in eos_ids and len(emitted) + len(block_emitted) < max_new:
                block_emitted.append(bonus)
            verifier_consumed = kk  # all draft forwards kept; bonus not yet in cache

        accept_lengths.append(n_accept)

        # ---- 4. fix verifier cache to reflect EXACTLY the emitted prefix ----
        # The verifier forward put kk draft tokens into vcache. We must leave the
        # cache holding (prompt + previously-emitted + block_emitted) so the next
        # iteration's pending_base_logits is correct.
        # Currently cache holds prompt+prev_emitted + drafts[0..kk-1].
        # We want it to hold prompt+prev_emitted + block_emitted.
        # Accepted draft tokens == block_emitted[:n_accept] (they matched). The
        # remaining block_emitted (the corrected base_tok and/or bonus) are NOT
        # yet in the verifier cache as the *emitted* token, so:
        #   - drop all kk draft tokens we forwarded that are NOT accepted.
        n_to_trim = kk - n_accept
        if n_to_trim > 0:
            trim_prompt_cache(vcache, n_to_trim)
        # Now cache holds prompt+prev_emitted + accepted drafts (== block_emitted[:n_accept]).
        # The extra emitted tokens beyond n_accept (corrected token / bonus) need a
        # verifier forward to (a) enter the cache and (b) yield next pending logits.
        emitted.extend(block_emitted)
        # advance verifier over the emitted-but-not-yet-cached tail to get next logits
        tail = block_emitted[n_accept:]
        # update context for next drafter seed
        context = mx.concatenate([context, mx.array([block_emitted])], axis=1)
        if block_emitted and block_emitted[-1] in eos_ids:
            break
        if len(emitted) >= max_new:
            break
        if tail:
            vlogits = model(mx.array([tail]), cache=vcache)[:, -1, :]
        else:
            # n_accept == kk and bonus was appended above as block_emitted[-1]
            # but bonus is not yet in cache; feed it.
            vlogits = model(mx.array([[block_emitted[-1]]]), cache=vcache)[:, -1, :]
        mx.eval(vlogits)
        pending_base_logits = vlogits

    install_base(model, base_qprojs)
    return emitted[:max_new], accept_lengths, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Logit-shift magnitude measurement (for null-control matching)
# ---------------------------------------------------------------------------

def measure_logit_shift(model, base_qprojs, math_adapter, prompts, eos_ids, n_pos=400):
    """Mean L2 logit-shift magnitude || logits_math - logits_base || on the
    base-greedy trajectory. Used to magnitude-match the null drafter IN-RUN."""
    mags = []
    for pid in prompts:
        if len(mags) >= n_pos:
            break
        install_base(model, base_qprojs)
        c_base = make_prompt_cache(model)
        bl = model(pid, cache=c_base)[:, -1, :]
        tok = greedy_token(bl[0])
        install_lora_drafter(model, base_qprojs, math_adapter)
        c_m = make_prompt_cache(model)
        ml = model(pid, cache=c_m)[:, -1, :]
        diff = float(mx.sqrt(mx.sum((ml[0].astype(mx.float32) - bl[0].astype(mx.float32)) ** 2)).item())
        mags.append(diff)
        # a few more positions along base-greedy
        for _ in range(3):
            if len(mags) >= n_pos or tok in eos_ids:
                break
            install_base(model, base_qprojs)
            bl = model(mx.array([[tok]]), cache=c_base)[:, -1, :]
            tok2 = greedy_token(bl[0])
            install_lora_drafter(model, base_qprojs, math_adapter)
            ml = model(mx.array([[tok]]), cache=c_m)[:, -1, :]
            diff = float(mx.sqrt(mx.sum((ml[0].astype(mx.float32) - bl[0].astype(mx.float32)) ** 2)).item())
            mags.append(diff)
            tok = tok2
    install_base(model, base_qprojs)
    return sum(mags) / len(mags), len(mags)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_prompts(tokenizer, n):
    items = []
    with open(GSM8K) as f:
        for line in f:
            if len(items) >= n:
                break
            q = json.loads(line)["question"]
            msgs = [{"role": "user",
                     "content": q + "\nReason step by step, then give the final answer."}]
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True)
            items.append(mx.array([ids]))
    return items


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main():
    mx.random.seed(SEED)
    print("[load] model + tokenizer")
    model, tokenizer = load(MODEL)
    model.eval() if hasattr(model, "eval") else None
    base_qprojs = capture_base_qprojs(model)

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    for t in ("<end_of_turn>", "<eos>"):
        try:
            tid = tokenizer.convert_tokens_to_ids(t)
            if tid is not None and tid >= 0:
                eos_ids.add(int(tid))
        except Exception:
            pass

    print("[load] adapters")
    math_ad = load_adapter_tensors(MATH_ADAPTER)
    med_ad = load_adapter_tensors(MED_ADAPTER)

    print("[data] building", N_ITEMS, "GSM8K prompts")
    prompts = build_prompts(tokenizer, N_ITEMS)

    # ---- magnitude-match the null drafter IN-RUN ----
    print("[null] measuring math-adapter mean logit-shift magnitude")
    sigma_total, n_meas = measure_logit_shift(model, base_qprojs, math_ad, prompts[:60], eos_ids)
    # per-coordinate std so that E||noise|| over vocab matches sigma_total:
    # ||noise|| ~ sigma_coord * sqrt(vocab)  => sigma_coord = sigma_total / sqrt(vocab)
    vocab = 262144
    null_sigma = sigma_total / (vocab ** 0.5)
    print(f"[null] mean math logit-shift L2 = {sigma_total:.4f} over {n_meas} pos -> noise std/coord = {null_sigma:.6g}")

    # ---- reference: base-only greedy ----
    print("[ref] base-only greedy decode")
    base_seqs = []
    base_secs = 0.0
    base_ntok = 0
    for i, p in enumerate(prompts):
        seq, sec = base_only_greedy(model, base_qprojs, p, MAX_NEW, eos_ids)
        base_seqs.append(seq)
        base_secs += sec
        base_ntok += len(seq)
        mx.clear_cache()
        if (i + 1) % 25 == 0:
            print(f"  base {i+1}/{N_ITEMS}")
    base_toks = base_ntok / base_secs

    conditions = {
        "math": math_ad,
        "medical": med_ad,
        "null": None,
    }
    results_cond = {}
    for name, adapter in conditions.items():
        print(f"[spec] condition={name}")
        match = 0
        all_accept = []
        secs = 0.0
        ntok = 0
        mismatches = []
        for i, p in enumerate(prompts):
            seq, acc, sec = spec_decode(
                model, base_qprojs, adapter, p, MAX_NEW, BLOCK, eos_ids,
                null_sigma=null_sigma if adapter is None else None,
            )
            ref = base_seqs[i]
            L = min(len(seq), len(ref))
            exact = (seq[:L] == ref[:L]) and (len(seq) == len(ref))
            if exact:
                match += 1
            else:
                if len(mismatches) < 5:
                    # find first divergence index
                    div = next((j for j in range(L) if seq[j] != ref[j]), L)
                    mismatches.append({"item": i, "first_div": div,
                                       "len_spec": len(seq), "len_ref": len(ref)})
            all_accept.extend(acc)
            secs += sec
            ntok += len(seq)
            mx.clear_cache()
            if (i + 1) % 25 == 0:
                print(f"  {name} {i+1}/{N_ITEMS} match_so_far={match}")
        results_cond[name] = {
            "exact_match_count": match,
            "exact_match_pct": 100.0 * match / N_ITEMS,
            "mean_accept_length": mean(all_accept),
            "n_blocks": len(all_accept),
            "tok_per_s": ntok / secs,
            "total_tokens": ntok,
            "wall_seconds": round(secs, 2),
            "first_mismatches": mismatches,
        }
        print(f"  {name}: match%={results_cond[name]['exact_match_pct']:.3f} "
              f"meanL={results_cond[name]['mean_accept_length']:.3f} "
              f"tok/s={results_cond[name]['tok_per_s']:.2f}")

    # ---- evaluate kill clauses ----
    k1_math = results_cond["math"]["exact_match_pct"] == 100.0
    k1_med = results_cond["medical"]["exact_match_pct"] == 100.0
    K1 = k1_math and k1_med

    L_math = results_cond["math"]["mean_accept_length"]
    L_med = results_cond["medical"]["mean_accept_length"]
    L_null = results_cond["null"]["mean_accept_length"]
    ratio_vs_med = L_math / L_med if L_med > 0 else float("inf")
    ratio_vs_null = L_math / L_null if L_null > 0 else float("inf")
    K2 = (ratio_vs_med >= 1.5) and (ratio_vs_null >= 1.3)

    speedup = results_cond["math"]["tok_per_s"] / base_toks if base_toks > 0 else 0.0
    K3 = speedup >= 1.25

    all_pass = K1 and K2 and K3
    verdict = "supported" if all_pass else "killed"

    results = {
        "experiment_id": "exp_spark_drafter_disposes",
        "is_smoke": False,
        "verdict": verdict,
        "all_pass": all_pass,
        "model": MODEL,
        "n_items": N_ITEMS,
        "max_new_tokens": MAX_NEW,
        "block_size": BLOCK,
        "lora": {"rank": LORA_R, "scale": LORA_SCALE,
                 "drafter_install": "setattr submodule replacement (LoRALinear.from_base on q_proj)"},
        "null_control": {
            "math_mean_logit_shift_L2": sigma_total,
            "measured_positions": n_meas,
            "noise_std_per_coord": null_sigma,
            "matched_in_run": True,
            "note": "Gaussian noise std/coord = mean_math_logit_shift_L2 / sqrt(vocab); measured in-run, not eyeballed.",
        },
        "base_only_greedy": {"tok_per_s": base_toks, "total_tokens": base_ntok,
                             "wall_seconds": round(base_secs, 2)},
        "conditions": results_cond,
        "kill_criteria": {
            "2297": {
                "K1_accuracy_invariance": {
                    "result": "pass" if K1 else "fail",
                    "math_match_pct": results_cond["math"]["exact_match_pct"],
                    "medical_match_pct": results_cond["medical"]["exact_match_pct"],
                    "null_match_pct": results_cond["null"]["exact_match_pct"],
                    "threshold": "100.0% exact base-greedy match in BOTH math and medical",
                },
                "K2_discrimination": {
                    "result": "pass" if K2 else "fail",
                    "L_math": L_math, "L_medical": L_med, "L_null": L_null,
                    "ratio_math_over_medical": ratio_vs_med,
                    "ratio_math_over_null": ratio_vs_null,
                    "threshold": "L_math >= 1.5*L_medical AND L_math >= 1.3*L_null",
                },
                "K3_net_speedup": {
                    "result": "pass" if K3 else "fail",
                    "math_tok_per_s": results_cond["math"]["tok_per_s"],
                    "base_tok_per_s": base_toks,
                    "speedup": speedup,
                    "threshold": "math tok/s >= 1.25 * base-only greedy tok/s",
                },
                "overall": "pass" if all_pass else "fail",
            }
        },
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"[done] verdict={verdict} K1={K1} K2={K2} K3={K3}")
    print(f"  L_math={L_math:.3f} L_med={L_med:.3f} L_null={L_null:.3f} speedup={speedup:.3f}")


if __name__ == "__main__":
    main()
