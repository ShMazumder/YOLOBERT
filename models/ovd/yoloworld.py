"""YOLO-World adapter (Ultralytics) — cheapest OVD model, good first target.

pip install ultralytics
weights: 'yolov8s-world.pt' / 'yolov8x-worldv2.pt' (auto-downloaded).

Ultralytics YOLO-World: set_classes(names) defines the open vocabulary, then
predict() returns boxes in xyxy pixel coords with class indices into `names`.
"""
from .base import OVDAdapter, register_adapter


@register_adapter("yoloworld")
class YoloWorldAdapter(OVDAdapter):
    def __init__(self, weights="yolov8s-world.pt", device=None, imgsz=800):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(weights)
        self.device = device
        self.imgsz = imgsz
        self._vocab = None

    def predict(self, image_path, class_names, score_thresh=0.001):
        # set_classes re-encodes text prompts; only redo when the vocabulary changes.
        if class_names != self._vocab:
            self.model.set_classes(list(class_names))
            self._vocab = list(class_names)
        res = self.model.predict(image_path, imgsz=self.imgsz, conf=score_thresh,
                                 device=self.device, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(([x1, y1, x2 - x1, y2 - y1], float(b.conf[0]), int(b.cls[0])))
        return out
