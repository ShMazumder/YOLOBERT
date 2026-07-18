"""fcos.py — anchor-free single-stage detector stub (FCOS-style).

Structure is real and runnable; the LOSS and target assignment are stubbed with
clear TODOs (that is your research surface). Pipeline shape:

    backbone (ResNet) -> FPN (P3..P7) -> shared head {cls, reg(ltrb), centerness}

Predictions postprocess to COCO result dicts for tools/metrics.CocoMetric.

Registered as 'fcos'. Config:
    model: fcos
    model_num_classes: 80
    model_backbone: resnet50        # torchvision name
    model_pretrained: true

Requires torchvision. Falls back gracefully if unavailable (raises at build).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register

FPN_STRIDES = (8, 16, 32, 64, 128)      # P3..P7
# per-level regression ranges (max ltrb distance handled at each level)
FPN_RANGES = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e9))
CENTER_RADIUS = 1.5                      # center-sampling radius in strides
INF = 1e9


def _fcos_iou_loss(pred, target, eps=1e-7):
    """GIoU loss between ltrb-encoded boxes (same anchor point). pred,target (M,4)."""
    lp, tp, rp, bp = pred.unbind(1)
    lt, tt, rt, bt = target.unbind(1)
    area_p = (lp + rp) * (tp + bp)
    area_t = (lt + rt) * (tt + bt)
    w_i = torch.min(lp, lt) + torch.min(rp, rt)
    h_i = torch.min(tp, tt) + torch.min(bp, bt)
    inter = w_i.clamp(min=0) * h_i.clamp(min=0)
    union = area_p + area_t - inter + eps
    iou = inter / union
    w_c = torch.max(lp, lt) + torch.max(rp, rt)
    h_c = torch.max(tp, tt) + torch.max(bp, bt)
    area_c = w_c * h_c + eps
    giou = iou - (area_c - union) / area_c
    return 1.0 - giou                      # (M,)


# ------------------------------------------------------------------ backbone+FPN
def _build_backbone_fpn(name="resnet50", pretrained=True, out_channels=256):
    from torchvision.models import get_model
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    weights = "DEFAULT" if pretrained else None
    backbone = get_model(name, weights=weights)
    # extract C3..C5 -> FPN with P6,P7 (extra blocks) => 5 levels.
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    return _resnet_fpn_extractor(
        backbone, trainable_layers=3, returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(out_channels, out_channels),
    )


# ------------------------------------------------------------------ head
class FCOSHead(nn.Module):
    def __init__(self, in_ch=256, num_classes=80, num_convs=4):
        super().__init__()
        self.num_classes = num_classes

        def tower():
            layers = []
            for _ in range(num_convs):
                layers += [nn.Conv2d(in_ch, in_ch, 3, padding=1),
                           nn.GroupNorm(32, in_ch), nn.ReLU(inplace=True)]
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_logits = nn.Conv2d(in_ch, num_classes, 3, padding=1)
        self.bbox_reg = nn.Conv2d(in_ch, 4, 3, padding=1)          # l,t,r,b
        self.centerness = nn.Conv2d(in_ch, 1, 3, padding=1)
        # per-level learnable scale on regression (FCOS trick)
        self.scales = nn.ParameterList(nn.Parameter(torch.tensor(1.0))
                                       for _ in FPN_STRIDES)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # focal-loss friendly cls bias prior (pi=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        cls_out, reg_out, ctr_out = [], [], []
        for lvl, feat in enumerate(features):
            c = self.cls_tower(feat)
            r = self.reg_tower(feat)
            cls_out.append(self.cls_logits(c))
            ctr_out.append(self.centerness(c))
            # predict in stride units (network outputs O(10)), then * stride -> pixels.
            # relu keeps ltrb non-negative. train + infer both read pixel distances.
            reg = F.relu(self.bbox_reg(r) * self.scales[lvl]) * FPN_STRIDES[lvl]
            reg_out.append(reg)
        return cls_out, reg_out, ctr_out


# ------------------------------------------------------------------ detector
@register("fcos")
class FCOS(nn.Module):
    def __init__(self, num_classes=80, backbone="resnet50", pretrained=True,
                 score_thresh=0.05, nms_thresh=0.6, topk=1000, **kw):
        super().__init__()
        self.num_classes = num_classes
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.backbone = _build_backbone_fpn(backbone, pretrained)
        self.head = FCOSHead(self.backbone.out_channels, num_classes)

    def forward(self, images, targets=None):
        """images: (B,3,H,W) tensor. Train -> loss dict; eval -> predictions."""
        feats = list(self.backbone(images).values())      # P3..P7
        cls, reg, ctr = self.head(feats)
        if self.training:
            assert targets is not None, "targets required in train mode"
            return self.loss(cls, reg, ctr, targets, images.shape[-2:])
        # eval: pull real COCO image_ids + original sizes from targets (if given)
        image_ids = orig_sizes = None
        if targets is not None:
            image_ids = [int(t["image_id"].item()) for t in targets]
            if "orig_size" in targets[0]:
                orig_sizes = [tuple(t["orig_size"].tolist()) for t in targets]  # (W,H)
        return self.postprocess(cls, reg, ctr, images.shape[-2:], image_ids, orig_sizes)

    # ---- locations per level (pixel centers) ----
    @staticmethod
    def _locations(feat, stride, device):
        _, _, h, w = feat.shape
        # float grid: pixel-center coords feed float box math + centerness sqrt
        sx = torch.arange(w, device=device, dtype=torch.float32) * stride + stride / 2
        sy = torch.arange(h, device=device, dtype=torch.float32) * stride + stride / 2
        yy, xx = torch.meshgrid(sy, sx, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)     # (HW,2)

    # ---- point grid + per-point level metadata (cached by shape) ----
    def _points_and_meta(self, feats):
        device = feats[0].device
        pts, ranges, radius = [], [], []
        for lvl, feat in enumerate(feats):
            loc = self._locations(feat, FPN_STRIDES[lvl], device)      # (HW,2)
            n = loc.shape[0]
            pts.append(loc)
            ranges.append(loc.new_tensor(FPN_RANGES[lvl]).expand(n, 2))
            radius.append(loc.new_full((n,), CENTER_RADIUS * FPN_STRIDES[lvl]))
        return torch.cat(pts, 0), torch.cat(ranges, 0), torch.cat(radius, 0)

    @torch.no_grad()
    def _assign(self, points, ranges, radius, gt_boxes, gt_labels):
        """Assign each point to a gt (or background). Returns (cls_t, reg_t, ctr_t).

        cls_t: (P,) label in 0..C (0=background). reg_t: (P,4) ltrb. ctr_t: (P,).
        """
        P = points.shape[0]
        if gt_boxes.numel() == 0:
            z = points.new_zeros(P)
            return z.long(), points.new_zeros(P, 4), z

        xs, ys = points[:, 0], points[:, 1]                            # (P,)
        gx1, gy1, gx2, gy2 = gt_boxes.unbind(1)                        # (G,)
        l = xs[:, None] - gx1[None]; t = ys[:, None] - gy1[None]
        r = gx2[None] - xs[:, None]; b = gy2[None] - ys[:, None]
        ltrb = torch.stack([l, t, r, b], -1)                          # (P,G,4)
        inside_box = ltrb.min(-1).values > 0                          # (P,G)

        # center sampling: point must fall in a shrunk box around gt center
        cx = (gx1 + gx2) / 2; cy = (gy1 + gy2) / 2
        rad = radius[:, None]                                          # (P,1)
        xmin = torch.max(cx[None] - rad, gx1[None])
        ymin = torch.max(cy[None] - rad, gy1[None])
        xmax = torch.min(cx[None] + rad, gx2[None])
        ymax = torch.min(cy[None] + rad, gy2[None])
        in_center = ((xs[:, None] > xmin) & (xs[:, None] < xmax) &
                     (ys[:, None] > ymin) & (ys[:, None] < ymax))     # (P,G)

        # FPN level range on max regression distance
        max_ltrb = ltrb.max(-1).values                               # (P,G)
        fit = (max_ltrb >= ranges[:, 0:1]) & (max_ltrb <= ranges[:, 1:2])

        is_pos = inside_box & in_center & fit                         # (P,G)
        areas = ((gx2 - gx1) * (gy2 - gy1))[None].expand(P, -1).clone()
        areas[~is_pos] = INF
        min_area, gt_idx = areas.min(1)                               # (P,)
        pos = min_area < INF

        cls_t = points.new_zeros(P).long()
        cls_t[pos] = gt_labels[gt_idx[pos]]                           # 1..C
        reg_t = ltrb[torch.arange(P, device=points.device), gt_idx]  # (P,4)
        reg_t[~pos] = 0

        lr = reg_t[:, [0, 2]]; tb = reg_t[:, [1, 3]]
        ctr_t = torch.sqrt(
            (lr.min(1).values / lr.max(1).values.clamp(min=1e-6)) *
            (tb.min(1).values / tb.max(1).values.clamp(min=1e-6))).clamp(0, 1)
        ctr_t[~pos] = 0
        return cls_t, reg_t, ctr_t

    def loss(self, cls, reg, ctr, targets, img_hw):
        """Full FCOS loss. Returns {'loss_cls','loss_reg','loss_ctr'}.

        Reference: Tian et al., FCOS, ICCV 2019 (arXiv:1904.01355).
        """
        from torchvision.ops import sigmoid_focal_loss
        device = cls[0].device
        N, C = cls[0].shape[0], self.num_classes
        points, ranges, radius = self._points_and_meta(cls)
        P = points.shape[0]

        # flatten predictions to (N,P,·) in the same level/spatial order as points
        cls_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(N, -1, C) for c in cls], 1)
        reg_flat = torch.cat([r.permute(0, 2, 3, 1).reshape(N, -1, 4) for r in reg], 1)
        ctr_flat = torch.cat([c.permute(0, 2, 3, 1).reshape(N, -1) for c in ctr], 1)

        cls_t = points.new_zeros(N, P).long()
        reg_t = points.new_zeros(N, P, 4)
        ctr_t = points.new_zeros(N, P)
        for i, tgt in enumerate(targets):
            gb = tgt["boxes"].to(device)
            gl = tgt["labels"].to(device)
            cls_t[i], reg_t[i], ctr_t[i] = self._assign(points, ranges, radius, gb, gl)

        pos = cls_t > 0                                               # (N,P)
        num_pos = pos.sum().clamp(min=1).float()

        # classification: sigmoid focal over C channels (0=bg -> all-zero target)
        one_hot = cls_flat.new_zeros(N, P, C)
        pi, pp = torch.where(pos)
        one_hot[pi, pp, cls_t[pos] - 1] = 1.0                        # label 1..C -> 0..C-1
        loss_cls = sigmoid_focal_loss(cls_flat, one_hot, alpha=0.25,
                                      gamma=2.0, reduction="sum") / num_pos

        if pos.any():
            reg_pred = reg_flat[pos]; reg_tgt = reg_t[pos]; w = ctr_t[pos]
            iou = _fcos_iou_loss(reg_pred, reg_tgt)                   # (M,)
            loss_reg = (iou * w).sum() / w.sum().clamp(min=1e-6)
            loss_ctr = F.binary_cross_entropy_with_logits(
                ctr_flat[pos], ctr_t[pos], reduction="mean")
        else:
            loss_reg = reg_flat.sum() * 0.0
            loss_ctr = ctr_flat.sum() * 0.0

        return {"loss_cls": loss_cls, "loss_reg": loss_reg, "loss_ctr": loss_ctr}

    # ---- INFERENCE -> COCO result dicts ----
    @torch.no_grad()
    def postprocess(self, cls, reg, ctr, img_hw, image_ids=None, orig_sizes=None):
        """Decode ltrb -> xyxy boxes, gather scores. Returns per-image list of
        COCO result dicts: {image_id, category_id, bbox=[x,y,w,h], score}.
        Applies per-image class-aware NMS (torchvision.ops.batched_nms).

        orig_sizes: optional list of (W,H) per image. If given, boxes are rescaled
        from the model input resolution back to the original image size so they
        match the COCO gt json (required for correct AP).
        """
        from torchvision.ops import batched_nms
        B = cls[0].shape[0]
        C = self.num_classes
        in_h, in_w = int(img_hw[0]), int(img_hw[1])
        # gather all levels per image first, then decode + NMS once.
        per_img_boxes = [[] for _ in range(B)]
        per_img_score = [[] for _ in range(B)]
        per_img_label = [[] for _ in range(B)]
        for lvl, stride in enumerate(FPN_STRIDES):
            loc = self._locations(cls[lvl], stride, cls[lvl].device)    # (HW,2)
            scores = (cls[lvl].sigmoid() * ctr[lvl].sigmoid())          # (B,C,H,W)
            scores = scores.permute(0, 2, 3, 1).reshape(B, -1, C)       # (B,HW,C)
            ltrb = reg[lvl].permute(0, 2, 3, 1).reshape(B, -1, 4)       # (B,HW,4)
            x, y = loc[:, 0], loc[:, 1]
            for b in range(B):
                l, t, r, d = ltrb[b].unbind(1)
                boxes = torch.stack([x - l, y - t, x + r, y + d], 1)    # xyxy
                sc, cl = scores[b].max(1)
                keep = sc > self.score_thresh
                per_img_boxes[b].append(boxes[keep])
                per_img_score[b].append(sc[keep])
                per_img_label[b].append(cl[keep])

        results = []
        for b in range(B):
            boxes = torch.cat(per_img_boxes[b], 0)
            sc = torch.cat(per_img_score[b], 0)
            cl = torch.cat(per_img_label[b], 0)
            iid = int(image_ids[b]) if image_ids is not None else b
            if boxes.numel():
                keep = batched_nms(boxes, sc, cl, self.nms_thresh)[:self.topk]
                boxes, sc, cl = boxes[keep], sc[keep], cl[keep]
                if orig_sizes is not None:                 # rescale to original size
                    ow, oh = orig_sizes[b]
                    scale = boxes.new_tensor([ow / in_w, oh / in_h, ow / in_w, oh / in_h])
                    boxes = boxes * scale
            for (x1, y1, x2, y2), s, c_ in zip(boxes.tolist(), sc.tolist(), cl.tolist()):
                results.append({
                    "image_id": iid,
                    "category_id": int(c_) + 1,            # 0..C-1 -> 1..C
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(s),
                })
        return results        # flat list for CocoMetric.update
