"""example_net — reference registered model. Copy this to start a real one.

Small MLP/conv head placeholder. Replace with your backbone (ResNet/ViT/Swin)
and task head (detection/segmentation/pose/VLM). Keep the @register decorator
and a forward(x) -> logits/predictions signature.
"""
import torch.nn as nn

from .registry import register


@register("example_net")
class ExampleNet(nn.Module):
    def __init__(self, in_dim=32, hidden=128, num_classes=10, **kw):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Example of a second registered variant sharing the class:
@register("example_net_wide")
def _wide(**kw):
    kw.setdefault("hidden", 512)
    return ExampleNet(**kw)
