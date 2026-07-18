"""models/ — architecture registry.

Register models by name so configs stay declarative:
    model: resnet_cls        # in configs/base.yaml
Then `build_model(cfg)` in train.py can dispatch via MODELS[cfg["model"]].

Add a new architecture: create models/your_net.py, decorate with @register.
"""
from .registry import MODELS, register, build  # noqa: F401
from . import example_net                        # noqa: F401  (registers on import)
try:
    from . import fcos                            # noqa: F401  (needs torchvision)
except Exception:                                 # keep scaffold importable w/o tv
    pass

__all__ = ["MODELS", "register", "build"]
