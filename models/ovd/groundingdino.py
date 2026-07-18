"""Grounding DINO adapter (HuggingFace transformers) — grounding-based OVD.

pip install transformers
weights: 'IDEA-Research/grounding-dino-base' | 'IDEA-Research/grounding-dino-tiny'

Takes a single period-separated text prompt and returns boxes with matched text
PHRASES (or label indices, version-dependent). We map phrases back to a class index
by substring match; unmatchable detections are dropped. Post-process kwargs are
resolved at runtime (transformers renamed box_threshold -> threshold).
"""
import inspect

from .base import OVDAdapter, register_adapter


def _phrase_to_index(phrase, class_names):
    p = str(phrase).lower().strip().strip(".")
    for i, c in enumerate(class_names):
        cl = c.lower()
        if cl == p or cl in p or p in cl:
            return i
    return None


@register_adapter("groundingdino")
class GroundingDinoAdapter(OVDAdapter):
    def __init__(self, weights="IDEA-Research/grounding-dino-base", device=None,
                 threshold=0.05, text_threshold=0.05, **kw):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.threshold = threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(weights)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(weights).to(device).eval()

    def predict(self, image_path, class_names, score_thresh=0.001):
        import torch
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        text = ". ".join(c.lower() for c in class_names) + "."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        pp = self.processor.post_process_grounded_object_detection
        params = inspect.signature(pp).parameters
        kw = {"target_sizes": [image.size[::-1]]}
        if "input_ids" in params:
            kw["input_ids"] = inputs.input_ids
        if "threshold" in params:                       # new name
            kw["threshold"] = max(self.threshold, score_thresh)
        elif "box_threshold" in params:                 # old name
            kw["box_threshold"] = max(self.threshold, score_thresh)
        if "text_threshold" in params:
            kw["text_threshold"] = self.text_threshold
        res = pp(outputs, **kw)[0]

        labels = res.get("text_labels", res.get("labels"))
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], labels):
            idx = _phrase_to_index(label, class_names) if isinstance(label, str) else int(label)
            if idx is None or idx >= len(class_names):
                continue
            x1, y1, x2, y2 = box.tolist()
            out.append(([x1, y1, x2 - x1, y2 - y1], float(score), idx))
        return out
