from abc import ABC, abstractmethod

import common.pointcloud as pc
from common.pointcloud import BBox3D


class Detector(ABC):
    @abstractmethod
    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        pass
