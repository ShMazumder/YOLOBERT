"""CPU smoke tests for the FCOS detection path. No dataset / GPU needed.

Run:  pytest tests/test_fcos.py -q     (or: python tests/test_fcos.py)

Covers forward shapes, finite+differentiable loss, and valid COCO-dict output.
Uses pretrained=False and a tiny image so it runs in seconds on CPU.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torchvision = pytest.importorskip("torchvision")  # skip whole file if absent
from models.fcos import FCOS, FPN_STRIDES  # noqa: E402

NUM_CLASSES = 5
IMG = 128            # divisible by 128 (P7 stride) -> clean feature maps


def _model():
    torch.manual_seed(0)
    return FCOS(num_classes=NUM_CLASSES, backbone="resnet18", pretrained=False)


def _fake_targets(batch=2, device="cpu"):
    targets = []
    for _ in range(batch):
        g = torch.randint(1, 4, (1,)).item()             # 1..3 boxes
        x1 = torch.rand(g) * (IMG * 0.5)
        y1 = torch.rand(g) * (IMG * 0.5)
        w = torch.rand(g) * (IMG * 0.3) + 8
        h = torch.rand(g) * (IMG * 0.3) + 8
        boxes = torch.stack([x1, y1, x1 + w, y1 + h], 1).to(device)
        labels = torch.randint(1, NUM_CLASSES + 1, (g,), device=device)  # 1..C
        targets.append({"boxes": boxes, "labels": labels})
    return targets


def test_forward_train_loss_finite_and_backprops():
    model = _model().train()
    x = torch.randn(2, 3, IMG, IMG)
    targets = _fake_targets(2)
    out = model(x, targets)
    assert set(out) == {"loss_cls", "loss_reg", "loss_ctr"}
    total = sum(out.values())
    assert torch.isfinite(total), f"non-finite loss: {out}"
    total.backward()                                     # must not raise
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)


def test_forward_eval_produces_valid_coco_dicts():
    model = _model().eval()
    x = torch.randn(2, 3, IMG, IMG)
    with torch.no_grad():
        results = model(x)
    assert isinstance(results, list)
    for r in results:
        assert set(r) >= {"image_id", "category_id", "bbox", "score"}
        assert len(r["bbox"]) == 4
        assert 1 <= r["category_id"] <= NUM_CLASSES
        assert 0.0 <= r["score"] <= 1.0


def test_empty_targets_gives_finite_loss():
    model = _model().train()
    x = torch.randn(1, 3, IMG, IMG)
    empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    out = model(x, empty)
    total = sum(out.values())
    assert torch.isfinite(total)
    total.backward()


def test_overfit_single_image_drops_loss():
    """Sanity: on one fixed image+gt the loss should decrease with training."""
    model = _model().train()
    x = torch.randn(1, 3, IMG, IMG)
    targets = _fake_targets(1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    first = last = None
    for step in range(15):
        opt.zero_grad()
        loss = sum(model(x, targets).values())
        loss.backward(); opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"


if __name__ == "__main__":
    test_forward_train_loss_finite_and_backprops()
    test_forward_eval_produces_valid_coco_dicts()
    test_empty_targets_gives_finite_loss()
    test_overfit_single_image_drops_loss()
    print("all FCOS smoke tests passed")
