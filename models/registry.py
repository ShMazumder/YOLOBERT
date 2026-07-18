"""Minimal name->constructor registry. No external deps."""

MODELS = {}


def register(name):
    """Decorator: @register('my_net') above a nn.Module subclass or factory fn."""
    def deco(obj):
        if name in MODELS:
            raise KeyError(f"model '{name}' already registered")
        MODELS[name] = obj
        return obj
    return deco


def build(cfg):
    """cfg: dict with key 'model' plus model kwargs. Returns nn.Module."""
    name = cfg.get("model", "example_net")
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'. registered: {list(MODELS)}")
    # pass through kwargs prefixed 'model_' (e.g. model_depth: 50)
    kwargs = {k[len("model_"):]: v for k, v in cfg.items() if k.startswith("model_")}
    return MODELS[name](**kwargs)
