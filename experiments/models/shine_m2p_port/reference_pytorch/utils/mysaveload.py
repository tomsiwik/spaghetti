import os
import json
import torch
from typing import Any, Dict
import numpy as np  # add at top
import random
from omegaconf import DictConfig, OmegaConf
from utils.mylogging import get_logger
from utils.myfreeze import freeze
from torch.utils.data import DataLoader
import time
from utils.myloradict import freeze_loradict

logger = get_logger("save & load")

def save_checkpoint(metanetwork, out_dir: str, metalora: Any, ift_additional_metalora: Any = None, extra_state: Dict[str, Any] = None):
    os.makedirs(out_dir, exist_ok=True)
    if metanetwork.metamodel.model.use_mem_token:
        torch.save(metanetwork.metamodel.model.mem_tokens, os.path.join(out_dir, "mem_tokens.pt"))
    torch.save(metanetwork.metanetwork.state_dict(), os.path.join(out_dir, "metanetwork.pth"))
    torch.save(metalora, os.path.join(out_dir, "metalora.pth"))
    if ift_additional_metalora is not None:
        torch.save(ift_additional_metalora, os.path.join(out_dir, "ift_additional_metalora.pth"))
    if extra_state is not None:
        with open(os.path.join(out_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(extra_state, f, ensure_ascii=False, indent=2)

def load_checkpoint(metanetwork, in_dir, device: str, load_ift_additional_metalora: bool = False, zero_ift_additional_metalora: bool = False):
    metanetwork.to("cpu")
    if metanetwork.metamodel.model.use_mem_token:
        saved_mem_tokens = torch.load(os.path.join(in_dir, "mem_tokens.pt"), map_location="cpu", weights_only=False)
        assert saved_mem_tokens.shape == metanetwork.metamodel.model.mem_tokens.shape, f"Shape mismatch for mem_tokens: saved {saved_mem_tokens.shape}, model {metanetwork.metamodel.model.mem_tokens.shape}"
        metanetwork.metamodel.model.mem_tokens = saved_mem_tokens
    metanetwork.metanetwork.load_state_dict(torch.load(os.path.join(in_dir, "metanetwork.pth"), weights_only=False, map_location="cpu"))
    metalora = torch.load(os.path.join(in_dir, "metalora.pth"), map_location="cpu", weights_only=False)
    metanetwork.to(device)
    metalora = move_to_device_and_change_into_leaf(metalora, device)
    freeze(metanetwork.metamodel)
    ift_additional_metalora_path = os.path.join(in_dir, "ift_additional_metalora.pth")
    if os.path.isfile(ift_additional_metalora_path):
        assert load_ift_additional_metalora and not zero_ift_additional_metalora, "Found ift_additional_metalora.pth but load_ift_additional_metalora is False"
        ift_additional_metalora = torch.load(ift_additional_metalora_path, map_location="cpu", weights_only=False)
        ift_additional_metalora = move_to_device_and_change_into_leaf(ift_additional_metalora, device)
        freeze_loradict(metalora)
    else:
        assert not load_ift_additional_metalora or zero_ift_additional_metalora, "ift_additional_metalora.pth not found but load_ift_additional_metalora is True"
        if zero_ift_additional_metalora:
            freeze_loradict(metalora)
    return metanetwork, metalora, ift_additional_metalora if (load_ift_additional_metalora and not zero_ift_additional_metalora) else None

def _rng_state_dict():
    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return state

def _set_rng_state(state: Dict[str, Any]):
    if state is None:
        return
    try:
        random.setstate(state["python_random"])
        np.random.set_state(state["numpy_random"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        logger.warning(f"Could not fully restore RNG states: {e}")

def save_training_state(
    out_dir: str,
    global_step: int,
    epoch: int,
    step_in_epoch: int,
    best_eval_loss: float,
):
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "global_step": global_step,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "best_eval_loss": best_eval_loss,
        "rng_state": _rng_state_dict(),
    }
    torch.save(payload, os.path.join(out_dir, "trainer_state.pt"))

def load_training_state(
    in_dir: str,
):
    path = os.path.join(in_dir, "trainer_state.pt")
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _set_rng_state(payload.get("rng_state"))

    return {
        "global_step": payload.get("global_step", 0),
        "epoch": payload.get("epoch", 1),
        "step_in_epoch": payload.get("step_in_epoch", 0),
        "best_eval_loss": payload.get("best_eval_loss", float("inf")),
    }


def get_latest_checkpoint(root_dir: str, only_epoch=False) -> str:
    if not os.path.isdir(root_dir):
        return None
    cands = [d for d in os.listdir(root_dir) if d.startswith("checkpoint-")]
    if only_epoch:
        cands = [d for d in cands if "epoch" in d]
    if not cands:
        return None
    steps = []
    for d in cands:
        try:
            steps.append((int(d.split("-")[-1]), d))
        except Exception:
            pass
    if not steps:
        return None
    steps.sort()
    return os.path.join(root_dir, steps[-1][1])

def move_to_device_and_change_into_leaf(obj, device):
    if torch.is_tensor(obj):
        new_obj = obj.to(device).detach().requires_grad_()
        return new_obj
    elif isinstance(obj, dict):
        return {k: move_to_device_and_change_into_leaf(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device_and_change_into_leaf(x, device) for x in obj]
    else:
        return obj