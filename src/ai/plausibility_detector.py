"""A decorator detector that keeps any other detector's output but discards the
physically impossible boxes.

The wrapped detector returns plain ``BBox3D`` with no class label, so only the
class-agnostic checks apply (in front of the camera, projects roughly into the
image, sensible absolute size). It is the cheapest possible defence against the
always-answering model and composes on top of any base detector -- e.g.
``PlausibilityFilterDetector(PerClassDetector(classes))``.
"""

import common.pointcloud as pc
from ai.detector import Detector
from common.geometry_filter import is_plausible
from common.pointcloud import BBox3D


class PlausibilityFilterDetector(Detector):
    def __init__(self, base: Detector, min_confidence: float = 0.0):
        self.base = base
        self.min_confidence = min_confidence

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        boxes = self.base.detect(img_base64, intrinsics)
        return [
            b for b in boxes
            if b.confidence >= self.min_confidence and is_plausible(b, intrinsics)
        ]
