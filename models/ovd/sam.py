"""SAM proposal adapter — class-agnostic localization source for the L axis.

The oracle-label-free L is contaminated by the OVD model's own misses (it can only
count boxes the detector itself produced). A frozen SAM gives the true "can ANYTHING
box these objects" signal, so L reflects localizability, not the detector's recall.

Uses Ultralytics SAM (no extra dep; weights auto-download):
    weights: 'mobile_sam.pt' (fast) | 'sam_b.pt' | 'sam2_b.pt'

This adapter ignores class_names (it is class-agnostic) and returns every proposal
with class_index 0. Feed its output as diagnose(..., results_agnostic=<these>).
"""
from .base import OVDAdapter, register_adapter


@register_adapter("sam")
class SamProposalAdapter(OVDAdapter):
    def __init__(self, weights="mobile_sam.pt", device=None, imgsz=1024, **kw):
        import torch
        from ultralytics import SAM
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = SAM(weights)
        self.model.to(device)
        self.device = device
        self.imgsz = imgsz

    def predict(self, image_path, class_names=None, score_thresh=0.0):
        # "everything" mode: no prompts -> automatic class-agnostic masks.
        res = self.model.predict(image_path, imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        confs = res.boxes.conf if res.boxes.conf is not None else None
        for i, b in enumerate(res.boxes):
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            score = float(confs[i]) if confs is not None else 1.0
            out.append(([x1, y1, x2 - x1, y2 - y1], score, 0))
        return out
