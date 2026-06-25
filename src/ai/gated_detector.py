"""Two-stage detector: a presence gate first prunes the class list, then any
base detector runs only on the surviving classes.

This is the cost-saving structure the project needs -- the slow per-class VLM is
only invoked for classes the gate believes are present, instead of all 18 on
every frame. The base detector is fully pluggable: pass a factory that builds the
detector for a given class subset, so you can drop in ``PerClassDetector`` (one
call per class, cheapest) or any of the ensemble detectors (more robust) without
changing the gating.
"""

from typing import Callable

import common.pointcloud as pc
from ai.detector import Detector
from ai.gate import Gate
from ai.per_class_detector import PerClassDetector
from common.pointcloud import BBox3D


def _default_base_factory(classes: list[str]) -> Detector:
    return PerClassDetector(classes)


class GatedDetector(Detector):
    def __init__(
            self,
            classes: list[str],
            gate: Gate,
            base_factory: Callable[[list[str]], Detector] = _default_base_factory,
    ):
        self.classes = classes
        self.gate = gate
        self.base_factory = base_factory

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        selected = self.gate.select(img_base64, intrinsics, self.classes)
        print(f"[gate] {len(selected)}/{len(self.classes)} classes kept: {selected}")
        if not selected:
            return []
        return self.base_factory(selected).detect(img_base64, intrinsics)
