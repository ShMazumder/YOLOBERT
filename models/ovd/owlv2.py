"""OWLv2 adapter (HuggingFace transformers) — dense image-text matching OVD.

pip install transformers
weights: 'google/owlv2-base-patch16-ensemble' | 'google/owlv2-large-patch14-ensemble'

Prompts with the full vocabulary as text queries; post_process returns boxes in
original-image pixels with labels indexing into the query list.
"""
from .base import OVDAdapter, register_adapter


@register_adapter("owlv2")
class Owlv2Adapter(OVDAdapter):
    def __init__(self, weights="google/owlv2-base-patch16-ensemble", device=None, **kw):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(weights)
        self.model = Owlv2ForObjectDetection.from_pretrained(weights).to(device).eval()

    def predict(self, image_path, class_names, score_thresh=0.001):
        import torch
        from PIL import Image
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=[list(class_names)], images=image,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)  # (h,w)
        res = self.processor.post_process_object_detection(
            outputs, threshold=score_thresh, target_sizes=target_sizes)[0]
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
            x1, y1, x2, y2 = box.tolist()
            out.append(([x1, y1, x2 - x1, y2 - y1], float(score), int(label)))
        return out
