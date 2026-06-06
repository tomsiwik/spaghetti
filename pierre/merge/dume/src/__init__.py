"""DUME: Training-Free Dynamic Upcycling of Expert Language Models (MLX)."""

from pierre.merge.dume.src.merge import moerge
from pierre.merge.dume.src.router import RidgeRouter, extract_router_weights
from pierre.merge.dume.src.model import DUMEMoEBlock, DUMEModel

__all__ = [
    "moerge",
    "RidgeRouter",
    "extract_router_weights",
    "DUMEMoEBlock",
    "DUMEModel",
]
