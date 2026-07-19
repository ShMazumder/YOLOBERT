"""proposer.py — generic class-agnostic region proposer from any detector weight.

Wraps any Ultralytics detection checkpoint (a YOLO trained on the target domain,
e.g. a VinDr chest-X-ray YOLO) and returns ALL its boxes regardless of class. Used
as a DOMAIN-APPROPRIATE localization reference alongside natural-image SAM and the
detector-intrinsic L_det, so the localization axis does not depend on a single
natural-image proposer.

    # produce domain-proposer agnostic results:
    python tools/run_agnostic.py --model proposer --weights /path/vindr_yolo.pt \
        --ann ... --imgs ... --out runs/diag/medical --limit 200
    # then diagnose(..., results_agnostic=<those>) -> L from the domain proposer

Register name: 'proposer'. Weight can be any .pt Ultralytics can load.
"""
from .base import OVDAdapter, register_adapter


@register_adapter("proposer")
class GenericProposerAdapter(OVDAdapter):
    def __init__(self, weights, device=None, imgsz=800, conf=0.001, **kw):
        import torch
        from ultralytics import YOLO
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(weights)
        self.model.to(device)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf

    def predict(self, image_path, class_names=None, score_thresh=None):
        thr = self.conf if score_thresh is None else score_thresh
        res = self.model.predict(image_path, imgsz=self.imgsz, conf=thr,
                                 device=self.device, verbose=False)[0]
        out = []
        if res.boxes is None:
            return out
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(([x1, y1, x2 - x1, y2 - y1], float(b.conf[0]), 0))  # class-agnostic
        return out
