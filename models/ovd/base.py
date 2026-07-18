"""Uniform interface for frozen OVD models.

An adapter must implement predict(image_path, class_names) and return detections
as (bbox_xywh, score, class_index) where class_index indexes into class_names.
Register with @register_adapter('name') so run_ovd.py can build it by string.
"""
from abc import ABC, abstractmethod

ADAPTERS = {}


def register_adapter(name):
    def deco(cls):
        ADAPTERS[name] = cls
        return cls
    return deco


def build_adapter(name, **kwargs):
    if name not in ADAPTERS:
        raise KeyError(f"unknown OVD adapter '{name}'. registered: {list(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)


class OVDAdapter(ABC):
    """Wrap a frozen open-vocabulary detector.

    predict() is called once per (image, prompt-vocabulary). The runner varies the
    vocabulary to realize the Global / Oracle / Agnostic modes.
    """

    @abstractmethod
    def predict(self, image_path, class_names, score_thresh=0.001):
        """Return list of (bbox_xywh, score, class_index).

        bbox_xywh : [x, y, w, h] in ORIGINAL image pixel coords.
        score     : float confidence.
        class_index : int index into `class_names`.
        Use a very low score_thresh so calibration/ECE sees the full score range;
        filtering happens later in the diagnostic.
        """
        raise NotImplementedError
