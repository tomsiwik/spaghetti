#!/usr/bin/env python3
"""PROD: Privacy Redesign via Null-Space Isolation.

Reuses P7.A1 null-space machinery; adds privacy-specific phases:
  Phase 1: Null-space bases for v_proj at layers 16-23.
  Phase 2: Train user A standard LoRA (baseline for K1643 quality ratio).
  Phase 3: Train user A null-space LoRA on medical data (for K1642 + K1643).
  Phase 4: Train user B null-space LoRA on legal data (for K1644).
  Phase 5: K1642 MIA (member vs legal OOD non-member) + K1643 quality + K1644 cos.

Kill criteria (pre-registered in MATH.md; see MATH.md for assumptions):
  K1642: MIA recovery <= 5% (<=1 of 20 members caught at 5th-percentile tau)
  K1643: PPL_null / PPL_std <= 1.05 on 5 held-out user-A texts
  K1644: mean over 8 layers of max|cos(B_A, B_B)| < 0.30

Platform: Apple M5 Pro 48GB, MLX only. mlx-lm 0.31.2.
"""
import gc
import json
import math
import os
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 6
LORA_SCALE = 8.0  # safe scale per mem-antipattern-003 (NOT 20)
SEED = 42

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
TRAIN_ITERS = 30 if IS_SMOKE else 500
LR = 1e-4
MAX_SEQ_LEN = 256
N_TARGET_LAYERS = 4 if IS_SMOKE else 8


def log(msg: str):
    print(msg, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def log_memory(label: str = ""):
    active = mx.get_active_memory() / 1e9
    cache = mx.get_cache_memory() / 1e9
    log(f"[MEM {label}] active={active:.2f}GB cache={cache:.2f}GB")


# ─── Training data (2 users × 20 QA texts each; 5 held-out per user) ────────────
#
# USER A: medical-style QA. USER B: legal-style QA. Disjoint vocabularies;
# OOD-qualifies each as non-member for the other's MIA.

USER_A_TRAIN = [
    "Q: What is the function of red blood cells?\nA: Red blood cells transport oxygen from the lungs to body tissues via the hemoglobin protein, which binds oxygen in high-oxygen environments and releases it in low-oxygen ones. A typical adult has roughly 25 trillion red blood cells.",
    "Q: Describe the structure of a neuron.\nA: A neuron consists of a cell body (soma) containing the nucleus, dendrites that receive signals from other neurons, and a single axon that transmits signals to other cells. The axon is often insulated by a myelin sheath.",
    "Q: What happens during systole?\nA: During systole, the ventricles of the heart contract, pumping oxygenated blood from the left ventricle to the body and deoxygenated blood from the right ventricle to the lungs. Blood pressure peaks at this phase.",
    "Q: Explain how antibiotics kill bacteria.\nA: Antibiotics disrupt essential bacterial processes. Beta-lactams prevent cell-wall synthesis, tetracyclines inhibit ribosomal protein synthesis, and fluoroquinolones block DNA gyrase required for replication.",
    "Q: What is a cytokine?\nA: Cytokines are small signaling proteins secreted by immune cells that coordinate responses to infection and injury. Interleukins, interferons, and tumor-necrosis factors are major classes with distinct functions.",
    "Q: How does insulin regulate blood glucose?\nA: Insulin, released by pancreatic beta cells after a meal, binds to receptors on muscle and fat cells, triggering GLUT4 transporter translocation and driving glucose uptake from the bloodstream.",
    "Q: What distinguishes viral from bacterial pneumonia?\nA: Viral pneumonia typically presents with diffuse interstitial infiltrates and lymphocytic inflammation, while bacterial pneumonia shows lobar consolidation and neutrophilic infiltration on imaging and pathology.",
    "Q: Describe the function of the glomerulus.\nA: The glomerulus is a capillary tuft in the kidney's nephron where blood is filtered under pressure, producing a plasma-like ultrafiltrate that enters Bowman's capsule for further renal processing.",
    "Q: What is a monoclonal antibody?\nA: A monoclonal antibody is an immunoglobulin derived from a single B-cell clone, binding a specific epitope with uniform affinity. Therapeutic monoclonals target receptors in oncology and autoimmunity.",
    "Q: Explain the mechanism of chemotherapy.\nA: Cytotoxic chemotherapy preferentially kills rapidly dividing cells by damaging DNA, inhibiting microtubule dynamics, or interfering with nucleotide synthesis, exploiting tumor cells' higher division rate.",
    "Q: What is the blood-brain barrier?\nA: The blood-brain barrier is a selective permeability interface formed by tight junctions between brain capillary endothelial cells, restricting passage of most large or polar molecules while permitting lipophilic agents.",
    "Q: How does an MRI distinguish tissue types?\nA: MRI uses radiofrequency pulses to perturb hydrogen proton spins in a strong magnetic field. Differences in T1 and T2 relaxation times among tissues generate contrast in the reconstructed image.",
    "Q: Describe mitochondrial ATP production.\nA: Mitochondria oxidize acetyl-CoA through the tricarboxylic acid cycle, generating NADH that feeds the electron transport chain. Proton pumping builds a gradient that ATP synthase uses to phosphorylate ADP.",
    "Q: What is atherosclerosis?\nA: Atherosclerosis is a chronic inflammatory disease of large arteries in which lipid-laden macrophages and smooth-muscle cells accumulate in the intima, forming plaques that narrow the lumen and may rupture.",
    "Q: Explain how vaccines produce immunity.\nA: Vaccines present attenuated or inactivated antigens to the immune system, priming B and T cells that generate memory. Re-exposure triggers a rapid, amplified response that prevents overt disease.",
    "Q: What is meant by allele dominance?\nA: An allele is dominant if a single copy produces the associated phenotype, masking the effect of a recessive allele on the homologous chromosome. Dominance arises from protein functionality rather than copy count alone.",
    "Q: Describe the renin-angiotensin system.\nA: Renin cleaves angiotensinogen to angiotensin I, which ACE converts to angiotensin II. Angiotensin II raises blood pressure via vasoconstriction and stimulates aldosterone secretion, promoting sodium retention.",
    "Q: What is a cerebral infarct?\nA: A cerebral infarct is a zone of brain tissue death caused by interrupted arterial blood supply, usually from thrombotic or embolic occlusion. Hypoxia triggers a cascade of excitotoxic injury and necrosis.",
    "Q: How does the liver metabolize drugs?\nA: Hepatocytes perform phase I reactions (oxidation, reduction, hydrolysis) via cytochrome P450 enzymes, and phase II reactions (conjugation with glucuronide, sulfate, or glutathione) to increase water solubility.",
    "Q: What distinguishes type 1 from type 2 diabetes?\nA: Type 1 diabetes is caused by autoimmune destruction of pancreatic beta cells and absolute insulin deficiency. Type 2 diabetes involves peripheral insulin resistance with relative insulin insufficiency.",
]

USER_A_HOLDOUT = [
    "Q: What is a lymph node?\nA: A lymph node is a small bean-shaped organ of the lymphatic system containing B cells, T cells, and macrophages. Nodes filter lymph fluid and coordinate adaptive immune responses to infection and tumor antigens.",
    "Q: How do beta blockers work?\nA: Beta blockers competitively inhibit beta-adrenergic receptors, reducing sympathetic stimulation of the heart. They decrease heart rate and contractility, lowering oxygen demand in ischemic and hypertensive conditions.",
    "Q: Describe the stages of mitosis.\nA: Mitosis comprises prophase (chromatin condenses), metaphase (chromosomes align at the spindle equator), anaphase (sister chromatids separate), and telophase (nuclei reform), followed by cytokinesis dividing the cytoplasm.",
    "Q: What is the role of surfactant in the lungs?\nA: Pulmonary surfactant is a lipoprotein complex secreted by type II pneumocytes that reduces alveolar surface tension, preventing alveolar collapse at end-expiration and easing lung expansion.",
    "Q: How does a PCR test amplify DNA?\nA: PCR uses cyclic heating and cooling with a heat-stable polymerase and primers flanking a target sequence. Each cycle doubles the target copy number, enabling detection of trace DNA through exponential amplification.",
]

USER_B_TRAIN = [
    "Q: What is stare decisis?\nA: Stare decisis is the common-law doctrine that courts should follow precedent when the same points arise again in litigation, promoting predictability and stability in the application of legal principles across jurisdictions.",
    "Q: Define actus reus.\nA: Actus reus is the physical act or unlawful omission that constitutes the external element of a crime. Criminal liability generally requires proof of actus reus contemporaneous with the mental element, mens rea.",
    "Q: What is a tort of negligence?\nA: Negligence is a tort in which a defendant breaches a duty of care owed to the plaintiff, causing foreseeable damage. Proof requires duty, breach, causation (factual and legal), and measurable harm.",
    "Q: Explain the Fourth Amendment.\nA: The Fourth Amendment to the United States Constitution protects against unreasonable searches and seizures and requires warrants issued upon probable cause, particularly describing the place, persons, or things to be searched or seized.",
    "Q: What is consideration in contract law?\nA: Consideration is the bargained-for exchange of value that renders a promise legally enforceable. It may be an act, forbearance, or return promise, and must generally be of some value in the eyes of the law.",
    "Q: Describe judicial review.\nA: Judicial review is the power of courts to examine legislative and executive acts and invalidate those inconsistent with the constitution. Marbury v. Madison (1803) established the doctrine in American federal jurisprudence.",
    "Q: What is res judicata?\nA: Res judicata is the doctrine that a final judgment on the merits by a competent court precludes relitigation of the same claim between the same parties. It promotes finality and judicial efficiency.",
    "Q: Explain mens rea levels.\nA: The Model Penal Code recognizes four culpability levels: purposeful (conscious object), knowing (practical certainty), reckless (conscious disregard of substantial risk), and negligent (unreasonable failure to perceive risk).",
    "Q: What is habeas corpus?\nA: Habeas corpus is a writ requiring a detainee to be brought before a court to determine the lawfulness of the detention. It is a fundamental safeguard against arbitrary imprisonment in common-law systems.",
    "Q: Describe promissory estoppel.\nA: Promissory estoppel enforces a promise that induces reasonable reliance by the promisee to their detriment, even absent consideration. It prevents injustice where strict contract formalities would otherwise bar recovery.",
    "Q: What is diversity jurisdiction?\nA: Diversity jurisdiction is federal-court authority over civil suits between citizens of different states where the amount in controversy exceeds the statutory threshold, currently seventy-five thousand dollars exclusive of interest and costs.",
    "Q: Explain the hearsay rule.\nA: Hearsay is an out-of-court statement offered to prove the truth of its contents and is generally inadmissible unless a recognized exception applies, such as present-sense impression, excited utterance, or business record.",
    "Q: What is specific performance?\nA: Specific performance is an equitable remedy ordering a breaching party to perform contractual obligations. Courts grant it when damages are inadequate, typically in unique-goods or real-estate transactions.",
    "Q: Describe the doctrine of corporate veil piercing.\nA: Courts disregard the corporate form and hold shareholders personally liable when the corporation is used to commit fraud, evade obligations, or functions as an alter ego of its owners.",
    "Q: What is the exclusionary rule?\nA: The exclusionary rule prohibits evidence obtained in violation of the Fourth Amendment from being used in the prosecutor's case-in-chief, deterring unconstitutional police conduct by removing its evidentiary value.",
    "Q: Explain fair-use doctrine.\nA: Fair use permits limited unauthorized use of copyrighted works for purposes such as criticism, commentary, news reporting, teaching, or research, evaluated through a four-factor statutory balance test.",
    "Q: What is the Chevron doctrine?\nA: Chevron deference requires courts to uphold a federal agency's reasonable interpretation of an ambiguous statute it administers, reflecting respect for agency expertise and legislative delegation of gap-filling authority.",
    "Q: Describe force majeure clauses.\nA: A force majeure clause excuses contractual performance when extraordinary events beyond the parties' control make performance impracticable or impossible, such as war, natural disaster, or certain governmental actions.",
    "Q: What is substantive due process?\nA: Substantive due process is the doctrine that the Due Process Clauses protect certain fundamental rights from government interference regardless of the procedural fairness of the deprivation process itself.",
    "Q: Explain the doctrine of sovereign immunity.\nA: Sovereign immunity is the legal principle that a state or federal government may not be sued without its consent, rooted in the common-law rule that the sovereign cannot be subjected to suit in its own courts.",
]


# ─── Null-space LoRA (from P7.A1 — copy-verified, not blindly copied) ──────────

class NullSpaceLoRALinear(nn.Module):
    """LoRA in null(W_base).  Forward: linear(x) + scale·((x @ Q) @ a) @ b.

    Q is frozen null-space basis; a/b are trained.  Matches mlx_lm LoRALinear
    parameter naming (lora_a, lora_b) for save/load compatibility.
    """

    def __init__(self, base_linear, Q: mx.array, r: int = 6, scale: float = 8.0):
        super().__init__()
        self.linear = base_linear
        self._Q = Q  # underscore-prefixed → not part of parameters
        self.scale = scale

        d_null = Q.shape[1]
        d_out = base_linear.weight.shape[0]

        init_scale = 1 / math.sqrt(d_null)
        self.lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(d_null, r)
        )
        self.lora_b = mx.zeros(shape=(r, d_out))

    def __call__(self, x):
        y = self.linear(x)
        x_null = x @ self._Q
        z = (x_null @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def dequantize_weight(linear):
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


def compute_null_basis(W: mx.array) -> mx.array:
    d_out, d_in = W.shape
    _, S, Vt = mx.linalg.svd(W, stream=mx.cpu)
    mx.eval(S, Vt)

    sigma_max = S[0].item()
    threshold = 1e-3 * sigma_max
    eff_rank = int(mx.sum(S > threshold).item())

    Q = Vt[eff_rank:, :].T  # (d_in, d_null)
    mx.eval(Q)

    log(f"    shape ({d_out},{d_in}) rank={eff_rank} null_dim={d_in - eff_rank}")
    return Q


def get_non_shared_layers(model):
    text_model = model.language_model.model if hasattr(model, "language_model") else model.model
    prev = text_model.previous_kvs
    return [i for i in range(len(prev)) if prev[i] == i]


def tokenize_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    out = []
    for text in texts:
        toks = tokenizer.encode(text)
        if len(toks) > max_len:
            toks = toks[:max_len]
        if len(toks) > 10:
            out.append(mx.array(toks))
    return out


def compute_ppl(model, token_arrays):
    total_loss, total_tokens = 0.0, 0
    for tokens in token_arrays:
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        mx.eval(logits)
        loss = nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="sum")
        mx.eval(loss)
        total_loss += loss.item()
        total_tokens += targets.shape[0]
        del logits, loss, x
    avg_loss = total_loss / max(total_tokens, 1)
    return {"avg_loss": round(avg_loss, 4),
            "perplexity": round(float(mx.exp(mx.array(avg_loss)).item()), 2),
            "n_tokens": total_tokens}


def compute_per_text_losses(model, token_arrays):
    """Per-text mean cross-entropy. For MIA: each scalar loss represents one text."""
    per_text = []
    for tokens in token_arrays:
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        mx.eval(logits)
        loss = nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")
        mx.eval(loss)
        per_text.append(float(loss.item()))
        del logits, loss, x
    return per_text


def load_model_and_layers(model_id):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    layers = (model.language_model.model.layers
              if hasattr(model, "language_model")
              else model.model.layers)
    return model, tokenizer, layers


# ─── Phases ────────────────────────────────────────────────────────────────────

def phase_null_bases():
    log("=== Phase 1: Null-space bases ===")
    model, tokenizer, layers = load_model_and_layers(MODEL_ID)
    non_shared = get_non_shared_layers(model)
    log(f"Non-shared layers: {non_shared}")

    target_indices = non_shared[-N_TARGET_LAYERS:]
    log(f"Target layers: {target_indices}")

    null_bases = {}
    t0 = time.time()
    for idx in target_indices:
        log(f"  Layer {idx} v_proj SVD...")
        W = dequantize_weight(layers[idx].self_attn.v_proj)
        mx.eval(W)
        Q = compute_null_basis(W)
        null_bases[idx] = Q
        del W
    log(f"SVD total: {time.time() - t0:.1f}s")

    mx.save_safetensors(str(EXPERIMENT_DIR / "null_bases.safetensors"),
                        {f"layer_{k}": v for k, v in null_bases.items()})

    result = {"target_layers": target_indices,
              "null_dims": {str(k): int(v.shape[1]) for k, v in null_bases.items()}}
    cleanup(model, tokenizer)
    for k in list(null_bases):
        del null_bases[k]
    del null_bases
    gc.collect()
    mx.clear_cache()
    return result


def _train_lora(model, layers, target_layers, null_mode, train_texts, tokenizer, null_bases=None):
    """Shared train loop. null_mode in {'std', 'null'}."""
    from mlx_lm.tuner.lora import LoRALinear

    model.freeze()
    for idx in target_layers:
        base = layers[idx].self_attn.v_proj
        if null_mode == "std":
            layers[idx].self_attn.v_proj = LoRALinear.from_base(
                base, r=LORA_RANK, scale=LORA_SCALE
            )
        else:
            Q = null_bases[idx]
            layers[idx].self_attn.v_proj = NullSpaceLoRALinear(
                base, Q, r=LORA_RANK, scale=LORA_SCALE
            )

    trainable = dict(tree_flatten(model.trainable_parameters()))
    n_trainable = sum(v.size for v in trainable.values())
    log(f"  Trainable params: {n_trainable:,} ({null_mode})")

    train_tokens = tokenize_texts(tokenizer, train_texts)
    log(f"  Training sequences: {len(train_tokens)}")

    optimizer = optim.AdamW(learning_rate=LR)

    def loss_fn(model, tokens):
        x = tokens[None, :-1]
        targets = tokens[1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits.squeeze(0), targets, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    t0 = time.time()
    losses = []
    gc.disable()
    for step in range(TRAIN_ITERS):
        tokens = train_tokens[step % len(train_tokens)]
        loss, grads = loss_and_grad(model, tokens)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
        if step % 100 == 0 or step == TRAIN_ITERS - 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            log(f"    step {step:4d}/{TRAIN_ITERS}: loss={losses[-1]:.4f} avg50={avg:.4f}")
    gc.enable()
    train_time = time.time() - t0
    final_loss = sum(losses[-20:]) / len(losses[-20:])
    log(f"  Train done in {train_time:.1f}s final_loss={final_loss:.4f}")

    trainable = dict(tree_flatten(model.trainable_parameters()))
    mx.eval(trainable)
    return trainable, losses, train_time, final_loss, n_trainable


def phase_train_user_adapter(user_label, null_mode, train_texts, target_layers):
    """Train one adapter (standard or null). Save to disk."""
    log(f"\n=== Phase: Train user={user_label} mode={null_mode} ===")
    model, tokenizer, layers = load_model_and_layers(MODEL_ID)

    null_bases = None
    if null_mode == "null":
        bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
        null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    trainable, losses, tt, final_loss, n_trainable = _train_lora(
        model, layers, target_layers, null_mode, train_texts, tokenizer, null_bases
    )

    adapter_path = EXPERIMENT_DIR / f"adapter_{user_label}_{null_mode}.safetensors"
    mx.save_safetensors(str(adapter_path), trainable)

    result = {
        "user": user_label,
        "mode": null_mode,
        "n_trainable": n_trainable,
        "train_time_s": round(tt, 1),
        "final_loss": round(final_loss, 4),
        "loss_first10": [round(l, 4) for l in losses[:10]],
        "loss_last10": [round(l, 4) for l in losses[-10:]],
    }
    cleanup(model, tokenizer)
    if null_bases is not None:
        for k in list(null_bases):
            del null_bases[k]
        del null_bases
    gc.collect()
    mx.clear_cache()
    return result


def _reapply_adapter(model, layers, target_layers, null_mode, adapter_path, null_bases=None):
    """Reload model + reapply adapter weights from file."""
    from mlx_lm.tuner.lora import LoRALinear

    model.freeze()
    for idx in target_layers:
        base = layers[idx].self_attn.v_proj
        if null_mode == "std":
            layers[idx].self_attn.v_proj = LoRALinear.from_base(
                base, r=LORA_RANK, scale=LORA_SCALE
            )
        else:
            Q = null_bases[idx]
            layers[idx].self_attn.v_proj = NullSpaceLoRALinear(
                base, Q, r=LORA_RANK, scale=LORA_SCALE
            )

    weights = mx.load(str(adapter_path))
    # load_weights expects a list of (key, value) tuples; tree_unflatten them
    from mlx.utils import tree_unflatten
    weights_list = list(weights.items())
    model.update(tree_unflatten(weights_list))
    mx.eval(model.parameters())


def phase_kc1643_quality(target_layers):
    """K1643: quality of null-space LoRA vs standard LoRA on held-out user-A texts."""
    log("\n=== Phase: K1643 quality ===")
    from mlx.utils import tree_unflatten

    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    results = {}
    for mode in ["std", "null"]:
        log(f"  Loading user_A/{mode} adapter...")
        model, tokenizer, layers = load_model_and_layers(MODEL_ID)
        _reapply_adapter(
            model, layers, target_layers, mode,
            EXPERIMENT_DIR / f"adapter_user_A_{mode}.safetensors",
            null_bases if mode == "null" else None,
        )
        holdout_tokens = tokenize_texts(tokenizer, USER_A_HOLDOUT)
        ppl = compute_ppl(model, holdout_tokens)
        log(f"  user_A holdout PPL ({mode}): {ppl['perplexity']:.2f}")
        results[mode] = ppl
        cleanup(model, tokenizer)

    for k in list(null_bases):
        del null_bases[k]
    del null_bases, bases_raw
    gc.collect()
    mx.clear_cache()
    return results


def phase_kc1642_mia(target_layers):
    """K1642: MIA — member losses vs OOD non-member (legal) losses under user_A null adapter."""
    log("\n=== Phase: K1642 MIA ===")
    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    model, tokenizer, layers = load_model_and_layers(MODEL_ID)
    _reapply_adapter(
        model, layers, target_layers, "null",
        EXPERIMENT_DIR / "adapter_user_A_null.safetensors",
        null_bases,
    )

    member_tokens = tokenize_texts(tokenizer, USER_A_TRAIN)
    nonmember_tokens = tokenize_texts(tokenizer, USER_B_TRAIN)  # OOD: legal
    log(f"  Members: {len(member_tokens)}, OOD non-members: {len(nonmember_tokens)}")

    member_losses = compute_per_text_losses(model, member_tokens)
    nonmember_losses = compute_per_text_losses(model, nonmember_tokens)

    import statistics
    log(f"  member_loss: mean={statistics.mean(member_losses):.4f} min={min(member_losses):.4f}")
    log(f"  nonmember_loss: mean={statistics.mean(nonmember_losses):.4f} min={min(nonmember_losses):.4f}")

    # τ* = 5th percentile of non-member losses. Recovery = fraction of members with loss <= τ*.
    sorted_nm = sorted(nonmember_losses)
    k5 = max(1, int(0.05 * len(sorted_nm)))
    tau = sorted_nm[k5 - 1]
    recovered = sum(1 for l in member_losses if l <= tau)
    recovery_fraction = recovered / len(member_losses)
    log(f"  tau*={tau:.4f}  recovered={recovered}/{len(member_losses)}  fraction={recovery_fraction:.4f}")

    cleanup(model, tokenizer)
    for k in list(null_bases):
        del null_bases[k]
    del null_bases, bases_raw
    gc.collect()
    mx.clear_cache()

    return {
        "member_losses": [round(l, 4) for l in member_losses],
        "nonmember_losses": [round(l, 4) for l in nonmember_losses],
        "member_mean": round(statistics.mean(member_losses), 4),
        "nonmember_mean": round(statistics.mean(nonmember_losses), 4),
        "tau_5pct": round(tau, 4),
        "recovered": recovered,
        "n_members": len(member_losses),
        "recovery_fraction": round(recovery_fraction, 4),
    }


def phase_kc1644_cross_user(target_layers):
    """K1644: mean over layers of max|cos(B_A, B_B)| where B = lora_b."""
    log("\n=== Phase: K1644 cross-user cosine on lora_b ===")

    weights_A = mx.load(str(EXPERIMENT_DIR / "adapter_user_A_null.safetensors"))
    weights_B = mx.load(str(EXPERIMENT_DIR / "adapter_user_B_null.safetensors"))

    per_layer_max_cos = {}
    all_max = []
    for idx in target_layers:
        # find lora_b keys for this layer
        key_A = next((k for k in weights_A if f"layers.{idx}.self_attn.v_proj.lora_b" in k), None)
        key_B = next((k for k in weights_B if f"layers.{idx}.self_attn.v_proj.lora_b" in k), None)
        if key_A is None or key_B is None:
            log(f"  Layer {idx}: missing lora_b (A={key_A}, B={key_B}), skip")
            continue

        B_A = weights_A[key_A]  # (r, d_out)
        B_B = weights_B[key_B]
        mx.eval(B_A, B_B)

        # normalize each column (direction) then compute max|cos|
        # lora_b rows are the r directions; we want cos between directions
        # B shape = (r, d_out) → rows are r directions in d_out
        def normalize_rows(M):
            norms = mx.sqrt(mx.sum(M * M, axis=1, keepdims=True))
            return M / mx.maximum(norms, 1e-10)

        B_A_n = normalize_rows(B_A)
        B_B_n = normalize_rows(B_B)
        cos_mat = B_A_n @ B_B_n.T  # (r, r)
        mx.eval(cos_mat)
        max_cos = float(mx.max(mx.abs(cos_mat)).item())
        per_layer_max_cos[str(idx)] = round(max_cos, 6)
        all_max.append(max_cos)
        log(f"  Layer {idx}: max|cos(B_A, B_B)| = {max_cos:.4f}")

    mean_max_cos = sum(all_max) / len(all_max) if all_max else 0.0
    return {
        "per_layer_max_cos": per_layer_max_cos,
        "mean_max_cos": round(mean_max_cos, 4),
        "n_layers": len(all_max),
    }


def phase_orthogonality_sanity(target_layers):
    """Sanity: max|W_v @ A_eff^T| < 1e-4 (confirms null-space construction is sound)."""
    log("\n=== Phase: orthogonality sanity check ===")
    model, tokenizer, layers = load_model_and_layers(MODEL_ID)

    bases_raw = mx.load(str(EXPERIMENT_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}
    adapter = mx.load(str(EXPERIMENT_DIR / "adapter_user_A_null.safetensors"))

    max_viol = 0.0
    per_layer = {}
    for idx in target_layers:
        Q = null_bases[idx]
        a_key = next((k for k in adapter if f"layers.{idx}.self_attn.v_proj.lora_a" in k), None)
        if a_key is None:
            continue
        A_small = adapter[a_key]  # (d_null, r)
        A_eff = Q @ A_small  # (d_in, r)
        mx.eval(A_eff)

        W_v = dequantize_weight(layers[idx].self_attn.v_proj)
        mx.eval(W_v)
        product = W_v @ A_eff  # (d_out, r)
        mx.eval(product)
        v = float(mx.max(mx.abs(product)).item())
        per_layer[str(idx)] = v
        max_viol = max(max_viol, v)
        log(f"  Layer {idx}: max|W_v @ A_eff| = {v:.2e}")
        del W_v, A_eff, product

    cleanup(model, tokenizer)
    for k in list(null_bases):
        del null_bases[k]
    del null_bases, bases_raw, adapter
    gc.collect()
    mx.clear_cache()

    return {"max_violation": max_viol, "per_layer": per_layer}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    mx.random.seed(SEED)
    log_memory("start")
    log(f"IS_SMOKE={IS_SMOKE}  iters={TRAIN_ITERS}  layers={N_TARGET_LAYERS}  r={LORA_RANK} scale={LORA_SCALE}")

    # Phase 1: null bases
    bases = phase_null_bases()
    target_layers = bases["target_layers"]
    log_memory("post-phase1")

    # Phase 2: user_A standard LoRA
    a_std = phase_train_user_adapter("user_A", "std", USER_A_TRAIN, target_layers)
    log_memory("post-phase2")

    # Phase 3: user_A null-space LoRA
    a_null = phase_train_user_adapter("user_A", "null", USER_A_TRAIN, target_layers)
    log_memory("post-phase3")

    # Phase 4: user_B null-space LoRA
    b_null = phase_train_user_adapter("user_B", "null", USER_B_TRAIN, target_layers)
    log_memory("post-phase4")

    # Phase 5: K1643 quality
    quality = phase_kc1643_quality(target_layers)
    log_memory("post-phase5")

    # Phase 6: K1642 MIA
    mia = phase_kc1642_mia(target_layers)
    log_memory("post-phase6")

    # Phase 7: K1644 cross-user cosine
    cross = phase_kc1644_cross_user(target_layers)
    log_memory("post-phase7")

    # Phase 8: Orthogonality sanity
    orth = phase_orthogonality_sanity(target_layers)
    log_memory("post-phase8")

    # ── Kill-criterion evaluation ──

    # K1642: recovery <= 5%
    k1642_pass = mia["recovery_fraction"] <= 0.05
    k1642 = {"pass": k1642_pass, "threshold": 0.05, **mia}

    # K1643: PPL_null / PPL_std <= 1.05
    ppl_std = quality["std"]["perplexity"]
    ppl_null = quality["null"]["perplexity"]
    ratio = ppl_null / ppl_std if ppl_std > 0 else float("inf")
    k1643_pass = ratio <= 1.05
    k1643 = {
        "pass": k1643_pass,
        "ppl_std": ppl_std,
        "ppl_null": ppl_null,
        "ratio": round(ratio, 4),
        "threshold": 1.05,
    }

    # K1644: mean max|cos| < 0.30
    k1644_pass = cross["mean_max_cos"] < 0.30
    k1644 = {"pass": k1644_pass, "threshold": 0.30, **cross}

    all_pass = k1642_pass and k1643_pass and k1644_pass

    # Smoke gates 'supported' → provisional; verdict field records intent.
    if IS_SMOKE:
        verdict = "PROVISIONAL"
    elif all_pass:
        verdict = "SUPPORTED"
    else:
        verdict = "KILLED"

    results = {
        "is_smoke": IS_SMOKE,
        "verdict": verdict,
        "all_pass": all_pass,
        "model": MODEL_ID,
        "config": {"rank": LORA_RANK, "scale": LORA_SCALE, "lr": LR,
                   "iters": TRAIN_ITERS, "n_target_layers": N_TARGET_LAYERS,
                   "mlx_lm_version": "0.31.2"},
        "bases": bases,
        "a_std": a_std,
        "a_null": a_null,
        "b_null": b_null,
        "k1642": k1642,
        "k1643": k1643,
        "k1644": k1644,
        "orthogonality_sanity": orth,
        "total_time_min": round((time.time() - t0) / 60, 2),
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2))

    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"K1642 (MIA recovery <= 5%):     {'PASS' if k1642_pass else 'FAIL'}  recovery={mia['recovery_fraction']:.4f}  {mia['recovered']}/{mia['n_members']}")
    log(f"K1643 (PPL_null/PPL_std <= 1.05): {'PASS' if k1643_pass else 'FAIL'}  ratio={ratio:.4f}  (std={ppl_std:.2f} null={ppl_null:.2f})")
    log(f"K1644 (mean max|cos B_A,B_B| < 0.30): {'PASS' if k1644_pass else 'FAIL'}  mean={cross['mean_max_cos']:.4f}")
    log(f"Orth sanity (max|W_v @ A_eff| < 1e-4): max={orth['max_violation']:.2e}")
    log(f"VERDICT: {verdict}  total_time={results['total_time_min']:.1f}min")


if __name__ == "__main__":
    main()
