#!/usr/bin/env python3
"""PROD: Red-team malicious adapter cannot extract other adapters' data.

Reuses adapters from `exp_prod_privacy_redesign` — no retraining.
Three attacks against the composed (user_A_null + user_B_null) model on Gemma-4 E4B v_proj layers 16-23:

  K1667 — activation probe: can a linear probe on composed hidden states
          distinguish member (user_A training) from non-member (user_A holdout +
          padding) above chance?
  K1668 — parameter extraction: how much of user_A's B@A subspace is recoverable
          from the composed delta via rank-r SVD?
  K1669 — canary extraction: can greedy-decode from 30-token prefix reproduce
          >=50% of the victim training text?

See MATH.md for theorems, predictions, and scope notes.

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
from mlx.utils import tree_unflatten
import numpy as np

mx.set_memory_limit(mx.device_info()["memory_size"] - 8 * 1024**3)
mx.set_cache_limit(2 * 1024**3)

EXPERIMENT_DIR = Path(__file__).parent
PRIVACY_DIR = EXPERIMENT_DIR.parent / "exp_prod_privacy_redesign"
RESULTS_FILE = EXPERIMENT_DIR / "results.json"
MODEL_ID = "mlx-community/gemma-4-e4b-it-4bit"
LORA_RANK = 6
LORA_SCALE = 8.0
TARGET_LAYERS = [16, 17, 18, 19, 20, 21, 22, 23]

IS_SMOKE = os.environ.get("SMOKE_TEST", "0") == "1"
if IS_SMOKE:
    TARGET_LAYERS = TARGET_LAYERS[:4]

SEED = 42


def log(msg: str):
    print(msg, flush=True)


def cleanup(*objects):
    for obj in objects:
        del obj
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


# ─── Data (must match dependency exactly for MIA/probe/canary validity) ─────

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

# Padding for probe balance: neutral medical queries (never seen in training).
PROBE_PADDING_NONMEMBER = [
    "Q: Define physiological homeostasis.\nA: Homeostasis refers to the maintenance of stable internal conditions by dynamic regulatory mechanisms, such as body temperature, blood pH, and electrolyte concentrations.",
    "Q: What is a reflex arc?\nA: A reflex arc is the neural pathway mediating an involuntary response: sensory neuron, interneuron in the spinal cord, and motor neuron, bypassing higher brain centers for speed.",
    "Q: Describe the thyroid gland's role.\nA: The thyroid gland secretes T3 and T4 hormones that regulate basal metabolic rate, growth, and thermogenesis, under feedback control by pituitary TSH.",
    "Q: What is cardiac output?\nA: Cardiac output is the product of heart rate and stroke volume, representing the total volume of blood pumped by the heart per minute, typically five liters at rest.",
    "Q: Explain the role of the pancreas.\nA: The pancreas performs exocrine secretion of digestive enzymes into the duodenum and endocrine secretion of insulin and glucagon to regulate blood glucose.",
    "Q: What is respiratory acidosis?\nA: Respiratory acidosis is a condition of decreased blood pH caused by inadequate ventilation that raises arterial carbon dioxide tension and shifts bicarbonate buffering.",
    "Q: Describe lymphatic drainage.\nA: Lymphatic vessels collect interstitial fluid, filter it through lymph nodes, and return it to venous circulation via the thoracic duct, also transporting dietary lipids.",
    "Q: What is hemostasis?\nA: Hemostasis is the process that halts bleeding after vascular injury, comprising vasoconstriction, platelet plug formation, coagulation cascade, and eventual fibrinolytic resolution.",
    "Q: Explain glycogenolysis.\nA: Glycogenolysis is the enzymatic breakdown of glycogen into glucose-1-phosphate, mobilizing stored glucose between meals under glucagon and epinephrine signaling.",
    "Q: What is a sarcomere?\nA: A sarcomere is the functional contractile unit of striated muscle, bounded by Z-discs, containing overlapping thin actin and thick myosin filaments that slide during contraction.",
    "Q: Describe the function of bile.\nA: Bile is secreted by the liver and stored in the gallbladder; its bile salts emulsify dietary fats in the small intestine to facilitate lipase digestion and lipid absorption.",
    "Q: What is proprioception?\nA: Proprioception is the sense of body position and movement, mediated by mechanoreceptors in muscles, tendons, and joints, integrated with vestibular and visual input centrally.",
    "Q: Explain action potential propagation.\nA: An action potential propagates along an axon through sequential depolarization of adjacent membrane regions via voltage-gated sodium channels, followed by repolarizing potassium efflux.",
    "Q: What is the Frank-Starling mechanism?\nA: The Frank-Starling mechanism describes how increased ventricular filling stretches myocardial fibers, enhancing the force of subsequent contraction and matching output to venous return.",
    "Q: Describe erythropoiesis.\nA: Erythropoiesis is the bone-marrow production of red blood cells, stimulated by renal erythropoietin in response to hypoxia and requiring iron, vitamin B12, and folate as substrates.",
]


# ─── Null-space LoRA (must match dependency construction exactly) ────────────

class NullSpaceLoRALinear(nn.Module):
    """Single null-space LoRA wrapping one base linear. Used for dep-compat unwrapping."""

    def __init__(self, base_linear, Q: mx.array, r: int = 6, scale: float = 8.0):
        super().__init__()
        self.linear = base_linear
        self._Q = Q
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


class ComposedNullSpaceLoRALinear(nn.Module):
    """Two null-space LoRA deltas (user A and user B) additively composed on the same base."""

    def __init__(self, base_linear, Q: mx.array, r: int = 6, scale: float = 8.0):
        super().__init__()
        self.linear = base_linear
        self._Q = Q
        self.scale = scale

        d_null = Q.shape[1]
        d_out = base_linear.weight.shape[0]

        init_scale = 1 / math.sqrt(d_null)
        # user A
        self.a_lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(d_null, r)
        )
        self.a_lora_b = mx.zeros(shape=(r, d_out))
        # user B
        self.b_lora_a = mx.random.uniform(
            low=-init_scale, high=init_scale, shape=(d_null, r)
        )
        self.b_lora_b = mx.zeros(shape=(r, d_out))

    def __call__(self, x):
        y = self.linear(x)
        x_null = x @ self._Q
        z_a = (x_null @ self.a_lora_a) @ self.a_lora_b
        z_b = (x_null @ self.b_lora_a) @ self.b_lora_b
        return y + (self.scale * (z_a + z_b)).astype(x.dtype)


def dequantize_weight(linear):
    if isinstance(linear, nn.QuantizedLinear):
        W = mx.dequantize(
            linear.weight, linear.scales, linear.biases,
            linear.group_size, linear.bits,
        )
    else:
        W = linear.weight
    return W.astype(mx.float32)


def load_model_with_composed_adapters():
    """Load Gemma-4 E4B + null bases + apply BOTH user_A_null + user_B_null to each target layer."""
    from mlx_lm import load
    model, tokenizer = load(MODEL_ID)
    text_model = model.language_model.model if hasattr(model, "language_model") else model.model
    layers = text_model.layers

    bases_raw = mx.load(str(PRIVACY_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    weights_A = mx.load(str(PRIVACY_DIR / "adapter_user_A_null.safetensors"))
    weights_B = mx.load(str(PRIVACY_DIR / "adapter_user_B_null.safetensors"))

    model.freeze()

    for idx in TARGET_LAYERS:
        base = layers[idx].self_attn.v_proj
        Q = null_bases[idx]
        layers[idx].self_attn.v_proj = ComposedNullSpaceLoRALinear(
            base, Q, r=LORA_RANK, scale=LORA_SCALE
        )

    # Remap weight keys: dependency wrote them as ".v_proj.lora_a" / ".v_proj.lora_b".
    # Our composed wrapper expects "a_lora_a/b" for user A and "b_lora_a/b" for user B.
    merged = {}
    for k, v in weights_A.items():
        new_k = k.replace(".v_proj.lora_a", ".v_proj.a_lora_a").replace(
            ".v_proj.lora_b", ".v_proj.a_lora_b"
        )
        merged[new_k] = v
    for k, v in weights_B.items():
        new_k = k.replace(".v_proj.lora_a", ".v_proj.b_lora_a").replace(
            ".v_proj.lora_b", ".v_proj.b_lora_b"
        )
        merged[new_k] = v

    weights_list = list(merged.items())
    model.update(tree_unflatten(weights_list))
    mx.eval(model.parameters())

    return model, tokenizer, layers, null_bases, weights_A, weights_B


def tokenize_texts(tokenizer, texts, max_len=256):
    out = []
    for text in texts:
        toks = tokenizer.encode(text)
        if len(toks) > max_len:
            toks = toks[:max_len]
        if len(toks) > 10:
            out.append(mx.array(toks))
    return out


# ─── K1667 — activation probe ────────────────────────────────────────────────

def collect_mean_hidden_states(model, token_arrays):
    """Forward each text, return mean over sequence of last hidden state (before lm_head)."""
    text_model = model.language_model.model if hasattr(model, "language_model") else model.model
    embeds = []
    for tokens in token_arrays:
        x = tokens[None, :]
        # Gemma-4 model.__call__ returns logits; we need hidden states before norm+lm_head.
        # Use the underlying text_model which returns hidden states.
        h = text_model(x)  # (1, T, d)
        mx.eval(h)
        mean_h = mx.mean(h.squeeze(0), axis=0)  # (d,)
        mx.eval(mean_h)
        embeds.append(np.array(mean_h.astype(mx.float32)))
        del h, mean_h, x
        mx.clear_cache()
    return np.stack(embeds, axis=0)  # (N, d)


def phase_k1667_activation_probe():
    log("\n=== Phase K1667: activation probe ===")
    model, tokenizer, layers, null_bases, wA, wB = load_model_with_composed_adapters()

    member_tokens = tokenize_texts(tokenizer, USER_A_TRAIN)
    holdout_tokens = tokenize_texts(tokenizer, USER_A_HOLDOUT + PROBE_PADDING_NONMEMBER)

    log(f"  Members: {len(member_tokens)}  Non-members (holdout+padding): {len(holdout_tokens)}")

    log("  Extracting member hidden states...")
    member_emb = collect_mean_hidden_states(model, member_tokens)
    log(f"    member_emb shape: {member_emb.shape}")

    log("  Extracting non-member hidden states...")
    nonmember_emb = collect_mean_hidden_states(model, holdout_tokens)
    log(f"    nonmember_emb shape: {nonmember_emb.shape}")

    cleanup(model, tokenizer)
    for k in list(null_bases):
        del null_bases[k]

    # Fit logistic regression probe
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_m = member_emb.shape[0]
    n_nm = nonmember_emb.shape[0]
    # Balance: trim to min(n_m, n_nm) = 20
    n_per_class = min(n_m, n_nm)
    X = np.concatenate([member_emb[:n_per_class], nonmember_emb[:n_per_class]], axis=0)
    y = np.concatenate([np.ones(n_per_class), np.zeros(n_per_class)])

    # Sanitize: replace NaN/inf with zeros, then standardize to prevent sklearn overflow warnings.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    X = StandardScaler().fit_transform(X)

    # L2-regularized logistic regression, stratified 2-fold CV
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    fold_accs = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000,
                                  random_state=SEED + fold_idx)
        clf.fit(X[train_idx], y[train_idx])
        acc = clf.score(X[test_idx], y[test_idx])
        log(f"    fold {fold_idx}: probe accuracy = {acc:.4f} (n_test={len(test_idx)})")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    adv = mean_acc - 0.5  # accuracy above chance
    passed = adv < 0.01
    log(f"  K1667 probe mean accuracy: {mean_acc:.4f}  advantage above chance: {adv:+.4f}  → pass={passed}")

    return {
        "pass": bool(passed),
        "threshold": 0.01,
        "metric": "accuracy_above_chance",
        "mean_accuracy": round(mean_acc, 4),
        "advantage_above_chance": round(adv, 4),
        "fold_accuracies": [round(a, 4) for a in fold_accs],
        "n_per_class": n_per_class,
        "n_folds": 2,
    }


# ─── K1668 — parameter extraction subspace overlap ──────────────────────────

def phase_k1668_param_extraction():
    log("\n=== Phase K1668: parameter extraction subspace overlap ===")

    weights_A = mx.load(str(PRIVACY_DIR / "adapter_user_A_null.safetensors"))
    weights_B = mx.load(str(PRIVACY_DIR / "adapter_user_B_null.safetensors"))
    bases_raw = mx.load(str(PRIVACY_DIR / "null_bases.safetensors"))
    null_bases = {int(k.split("_")[1]): v for k, v in bases_raw.items()}

    per_layer = {}
    overlaps_A = []
    overlaps_B = []

    for idx in TARGET_LAYERS:
        # find lora_a/b for A and B at layer idx
        ka_a = next((k for k in weights_A if f"layers.{idx}.self_attn.v_proj.lora_a" in k), None)
        ka_b = next((k for k in weights_A if f"layers.{idx}.self_attn.v_proj.lora_b" in k), None)
        kb_a = next((k for k in weights_B if f"layers.{idx}.self_attn.v_proj.lora_a" in k), None)
        kb_b = next((k for k in weights_B if f"layers.{idx}.self_attn.v_proj.lora_b" in k), None)
        if None in (ka_a, ka_b, kb_a, kb_b):
            log(f"  layer {idx}: missing keys, skip")
            continue

        A_A = weights_A[ka_a]  # (d_null, r)
        B_A = weights_A[ka_b]  # (r, d_out)
        A_B = weights_B[kb_a]
        B_B = weights_B[kb_b]
        Q = null_bases[idx]  # (d_in, d_null)
        mx.eval(A_A, B_A, A_B, B_B, Q)

        # True per-user delta: ΔW_u = s * B_u.T @ A_u.T @ Q.T  ? Let's be careful with shapes.
        # Forward is y + scale * ((x @ Q) @ lora_a) @ lora_b.
        # So as a linear map on x: Δy_u = scale * x @ Q @ lora_a @ lora_b
        #                        = x @ (scale * Q @ A_u @ B_u)
        # Effective weight: ΔW_u (d_in, d_out) = scale * Q @ A_u @ B_u
        # For a conventional (d_out, d_in) weight convention, transpose.
        sc = mx.array(LORA_SCALE, dtype=A_A.dtype)
        dW_A = sc * (Q @ A_A @ B_A)  # (d_in, d_out)
        dW_B = sc * (Q @ A_B @ B_B)
        dW_composed = dW_A + dW_B
        mx.eval(dW_A, dW_B, dW_composed)

        # Attacker's best rank-r approximation of composed delta
        dW_c_np = np.array(dW_composed.astype(mx.float32))
        U, S, Vt = np.linalg.svd(dW_c_np, full_matrices=False)
        # top-r output-direction subspace: left singular vectors columns → but dW is (d_in,d_out)
        # Output directions live in V (columns), i.e., Vt.T columns are d_out directions.
        U_attack_out = Vt[:LORA_RANK, :].T  # (d_out, r) attacker-recovered output dirs

        # True per-user output directions: B_u rows are r directions in d_out.
        def subspace_overlap(U_attack, B_u):
            # B_u shape (r, d_out); normalize rows (each row is a unit output direction)
            B_np = np.array(B_u.astype(mx.float32))  # (r, d_out)
            # orthonormalize B rows via QR
            Q_B, _ = np.linalg.qr(B_np.T, mode="reduced")  # Q_B: (d_out, r)
            # overlap = ||U_attack^T @ Q_B||_F^2 / r  (average cos^2 of canonical angles)
            M = U_attack.T @ Q_B  # (r, r)
            return float((M * M).sum() / LORA_RANK)

        ov_A = subspace_overlap(U_attack_out, B_A)
        ov_B = subspace_overlap(U_attack_out, B_B)
        per_layer[idx] = {"overlap_A": round(ov_A, 4), "overlap_B": round(ov_B, 4)}
        overlaps_A.append(ov_A)
        overlaps_B.append(ov_B)
        log(f"  layer {idx}: overlap_A={ov_A:.4f}  overlap_B={ov_B:.4f}")

        del dW_A, dW_B, dW_composed, dW_c_np, U, S, Vt, U_attack_out
        mx.clear_cache()

    mean_ov_A = float(np.mean(overlaps_A)) if overlaps_A else 1.0
    mean_ov_B = float(np.mean(overlaps_B)) if overlaps_B else 1.0
    max_ov = max(mean_ov_A, mean_ov_B)
    passed = max_ov < 0.01
    log(f"  K1668 mean_overlap_A={mean_ov_A:.4f}  mean_overlap_B={mean_ov_B:.4f}  max={max_ov:.4f}  → pass={passed}")

    for k in list(null_bases):
        del null_bases[k]
    gc.collect()
    mx.clear_cache()

    return {
        "pass": bool(passed),
        "threshold": 0.01,
        "metric": "subspace_overlap_max_over_users",
        "mean_overlap_A": round(mean_ov_A, 4),
        "mean_overlap_B": round(mean_ov_B, 4),
        "max_overlap": round(max_ov, 4),
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "n_layers": len(overlaps_A),
    }


# ─── K1669 — canary extraction via prefix greedy decoding ───────────────────

def _greedy_decode(model, tokenizer, prompt_ids, n_gen=50, temperature=0.0):
    """Greedy decoding. Returns list of generated token ids."""
    x = prompt_ids[None, :]
    generated = []
    for _ in range(n_gen):
        logits = model(x)  # (1, T, V)
        mx.eval(logits)
        next_logits = logits[0, -1, :]
        next_id = int(mx.argmax(next_logits).item())
        generated.append(next_id)
        # append to x
        x = mx.concatenate([x, mx.array([[next_id]])], axis=1)
        del logits, next_logits
    return generated


def phase_k1669_canary():
    log("\n=== Phase K1669: canary extraction ===")
    model, tokenizer, layers, null_bases, wA, wB = load_model_with_composed_adapters()

    # Most training texts are ~40-60 tokens; prefix+gen must fit.
    prefix_len = 15
    n_gen = 20
    canaries = USER_A_TRAIN
    per_canary = []
    n_extracted = 0

    for i, text in enumerate(canaries):
        full_toks = tokenizer.encode(text)
        if len(full_toks) < prefix_len + n_gen:
            log(f"  canary {i}: too short ({len(full_toks)} toks), skip")
            continue

        prefix = mx.array(full_toks[:prefix_len])
        true_continuation = full_toks[prefix_len:prefix_len + n_gen]
        gen = _greedy_decode(model, tokenizer, prefix, n_gen=n_gen)
        # token-level match (positional)
        matches = sum(1 for a, b in zip(gen, true_continuation) if a == b)
        overlap = matches / n_gen
        extracted = overlap >= 0.5
        if extracted:
            n_extracted += 1
        per_canary.append({
            "idx": i,
            "overlap": round(overlap, 4),
            "extracted": bool(extracted),
        })
        log(f"  canary {i:2d}: overlap={overlap:.4f}  extracted={extracted}")

    passed = n_extracted <= 1
    log(f"  K1669 extracted={n_extracted}/{len(per_canary)}  pass={passed}")

    cleanup(model, tokenizer)
    for k in list(null_bases):
        del null_bases[k]

    return {
        "pass": bool(passed),
        "threshold_extracted": 1,
        "n_canaries": len(per_canary),
        "n_extracted": n_extracted,
        "extraction_rate": round(n_extracted / max(len(per_canary), 1), 4),
        "overlap_threshold": 0.5,
        "prefix_len": prefix_len,
        "n_gen": n_gen,
        "per_canary": per_canary,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    mx.random.seed(SEED)
    np.random.seed(SEED)

    t0 = time.time()
    log(f"=== exp_prod_red_team_adapter_injection (smoke={IS_SMOKE}) ===")
    log(f"Target layers: {TARGET_LAYERS}")
    log(f"Dependency dir: {PRIVACY_DIR}")

    # Phase order: cheapest first (K1668 — pure weights, no inference).
    k1668 = phase_k1668_param_extraction()
    k1667 = phase_k1667_activation_probe()
    k1669 = phase_k1669_canary()

    total_time_min = round((time.time() - t0) / 60.0, 2)

    all_pass = bool(k1667["pass"]) and bool(k1668["pass"]) and bool(k1669["pass"])
    verdict = "SUPPORTED" if all_pass else "KILLED"

    results = {
        "is_smoke": IS_SMOKE,
        "verdict": verdict,
        "all_pass": all_pass,
        "model": MODEL_ID,
        "config": {
            "rank": LORA_RANK,
            "scale": LORA_SCALE,
            "target_layers": TARGET_LAYERS,
            "mlx_lm_version": "0.31.2",
        },
        "k1667": k1667,
        "k1668": k1668,
        "k1669": k1669,
        "total_time_min": total_time_min,
    }

    with RESULTS_FILE.open("w") as f:
        json.dump(results, f, indent=2)
    log(f"\n=== DONE ({total_time_min:.2f} min) verdict={verdict} ===")
    log(f"  K1667 (probe):   pass={k1667['pass']}  advantage={k1667['advantage_above_chance']:+.4f}")
    log(f"  K1668 (extract): pass={k1668['pass']}  max_overlap={k1668['max_overlap']:.4f}")
    log(f"  K1669 (canary):  pass={k1669['pass']}  extracted={k1669['n_extracted']}/{k1669['n_canaries']}")


if __name__ == "__main__":
    main()
