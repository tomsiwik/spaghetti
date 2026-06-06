"""Model registry with auto-import of all .py files in this directory."""

import importlib
import pkgutil
from pathlib import Path

MODEL_REGISTRY: dict[str, dict] = {}  # name -> {cls, parent, path}


def register(name: str, parent: str | None = None):
    """Decorator to register a model class in the arena."""
    def wrapper(cls):
        MODEL_REGISTRY[name] = {"cls": cls, "parent": parent}
        cls._registry_name = name
        return cls
    return wrapper


def get_model(name: str, **kwargs):
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]["cls"](**kwargs)


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())


# Auto-import all subpackages in this directory
_pkg_path = Path(__file__).parent
for _info in pkgutil.iter_modules([str(_pkg_path)]):
    importlib.import_module(f".{_info.name}", package=__name__)
