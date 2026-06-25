import numpy as np
import cupy as cp

import common.pointcloud as pc
from ai.detector import Detector
from common.pointcloud import BBox3D


class BaselineDetector(Detector):
    """Random baseline: 4 boxes per image, uniformly sampled in camera space.

    Useful as a lower-bound sanity check — any real detector should beat this.
    """

    def __init__(self, n_boxes: int = 4, seed: int = None):
        self.n_boxes = n_boxes
        self.rng = np.random.default_rng(seed)

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        boxes = []
        for _ in range(self.n_boxes):
            # Position: random point in front of camera, 1–6 m away.
            z = self.rng.uniform(1.0, 3.0)
            x = self.rng.uniform(-z * 0.5, z * 0.5)
            y = self.rng.uniform(0, 0)
            position = cp.array([x, y, z])

            # Size: random box between 0.3 and 1.5 m per side.
            size = cp.array(self.rng.uniform(0.3, 1.5, size=3))

            boxes.append(BBox3D(position=position, size=size, rotation=cp.eye(3)))
        return boxes
