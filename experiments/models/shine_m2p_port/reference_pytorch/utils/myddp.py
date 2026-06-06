import os
import random
import torch
import torch.distributed as dist

# ========= DDP helpers =========
def should_use_ddp() -> bool:
    # If launched with torchrun, WORLD_SIZE will be set (>1 for multi-proc)
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def ddp_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_world_size() -> int:
    return dist.get_world_size() if ddp_is_active() else 1

def get_rank() -> int:
    return dist.get_rank() if ddp_is_active() else 0

def get_local_rank() -> int:
    # torchrun sets LOCAL_RANK; default to 0 for single GPU/CPU
    return int(os.environ.get("LOCAL_RANK", "0"))

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if ddp_is_active():
        dist.barrier()

def ddp_init_if_needed():
    # Only initialize if we're truly in a multi-process setting
    if should_use_ddp() and dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        # Make printing on non-zero ranks quieter
        if not is_main_process():
            import builtins as __builtin__
            def _silent_print(*args, **kwargs):
                pass
            __builtin__.print = _silent_print

def ddp_cleanup_if_needed():
    if ddp_is_active():
        dist.barrier()
        dist.destroy_process_group()

@torch.no_grad()
def distributed_mean(value: float, device: torch.device) -> float:
    """Average a scalar across processes."""
    if not ddp_is_active():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return float(t.item())
# ==============================