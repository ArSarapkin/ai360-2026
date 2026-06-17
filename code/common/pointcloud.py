from dataclasses import dataclass
from abc import abstractmethod, ABC

import numpy as np
import random


@dataclass
class Distortion:
    k: list[float]  # len(k) = 3
    p1: float
    p2: float


@dataclass
class Extrinsics:
    rotation: np.ndarray  # 3 * 3
    translation: np.ndarray  # 3


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    K: np.ndarray  # 3 * 3


@dataclass
class BBox3D:
    position: np.ndarray  # 3
    size: np.ndarray  # 3
    rotation: np.ndarray  # 3 * 3


def relative(base: Extrinsics, to: Extrinsics) -> Extrinsics:
    rotation = to.rotation @ base.rotation.transpose()
    translation_np = to.translation - rotation @ base.translation
    return Extrinsics(
        rotation=rotation,
        translation=translation_np,
    )


def to_intrinsics(fx, fy, cx, cy) -> Intrinsics:
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    return Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, K=K)


DISTORTION_ITERATIONS = 50


@dataclass
class Camera:
    intrinsics: Intrinsics
    extrinsics: Extrinsics
    distortion: Distortion

    def to_world_pos(self, camera_pos: np.ndarray) -> np.ndarray:
        return self.extrinsics.rotation.transpose() @ (camera_pos - self.extrinsics.translation)

    def to_camera_pos(self, world_pos: np.ndarray) -> np.ndarray:
        return (self.extrinsics.rotation @ world_pos) + self.extrinsics.translation

    def calculate_distortion(self, x, y) -> tuple[float, float, float]:
        r2 = x ** 2 + y ** 2
        rad = 1
        for k in range(len(self.distortion.k)):
            R = r2 ** (k + 1)
            rad += self.distortion.k[k] * R
        ndx = 2 * self.distortion.p1 * x * y + self.distortion.p2 * (r2 + 2 * (x ** 2))
        ndy = 2 * self.distortion.p2 * x * y + self.distortion.p1 * (r2 + 2 * (y ** 2))
        return rad, ndx, ndy

    def to_Z1(self, i, j) -> tuple[float, float]:
        x = (j - self.intrinsics.cx) / self.intrinsics.fx
        y = (i - self.intrinsics.cy) / self.intrinsics.fy
        x0, y0 = x, y
        for _ in range(DISTORTION_ITERATIONS):
            rad, ndx, ndy = self.calculate_distortion(x, y)
            x = (x0 - ndx) / rad
            y = (y0 - ndy) / rad
        return x, y

    def from_Z1(self, x, y) -> tuple[int, int]:
        rad, ndx, ndy = self.calculate_distortion(x, y)
        x, y = x * rad + ndx, y * rad + ndy
        i = self.intrinsics.fy * y + self.intrinsics.cy
        j = self.intrinsics.fx * x + self.intrinsics.cx
        return round(i), round(j)

    def depth_to_world_pos(self, i, j, depth) -> np.ndarray:
        x, y = self.to_Z1(i, j)
        camera_pos = np.array([x * depth, y * depth, depth])
        return self.to_world_pos(camera_pos)

    def world_to_image_pos(
            self,
            world_pos: np.ndarray,
    ) -> tuple[int, int]:
        camera_pos = self.to_camera_pos(world_pos)
        x, y = camera_pos[0] / camera_pos[2], camera_pos[1] / camera_pos[2]
        return self.from_Z1(x, y)

    def bbox_to_world_pos(
            self,
            bbox: BBox3D,
    ):
        point_1 = self.to_world_pos(bbox.point_1)
        point_2 = self.to_world_pos(bbox.point_2)
        return BBox3D(point_1, point_2)


@dataclass
class WorldPoint:
    position: np.ndarray  # 3
    color: np.ndarray  # 3


def is_valid_image_pos(image: np.ndarray, i: int, j: int) -> bool:
    H, W = image.shape[:2]
    return 0 <= i < H and 0 <= j < W


class DepthMap(ABC):
    image: np.ndarray

    def __init__(self, image: np.ndarray):
        self.image = image

    @abstractmethod
    def get(self, i, j) -> float | None:
        pass

    def shape(self):
        return self.image.shape[:2]


class DepthMapShuffling(DepthMap):
    depth_map: DepthMap
    factor: float

    def __init__(self, depth_map: DepthMap, factor: float):
        super().__init__(depth_map.image)
        self.depth_map = depth_map
        self.factor = factor

    def get(self, i, j) -> float | None:
        depth = self.depth_map.get(i, j)
        if depth is None:
            return None
        depth = depth + random.uniform(-self.factor, self.factor)
        return depth


def shuffling(depth_map: DepthMap, factor: float) -> DepthMap:
    return DepthMapShuffling(depth_map, factor)


@dataclass
class Frame:
    rgb_image: np.ndarray
    depth_map: DepthMap


class Scene:
    depth_camera: Camera
    rgb_camera: Camera

    def __init__(self, depth_camera: Camera, rgb_camera: Camera):
        self.rgb_camera = rgb_camera
        self.depth_camera = depth_camera

    def process_frame(
            self,
            frame: Frame,
    ) -> list[WorldPoint]:
        H, W = frame.depth_map.shape()
        point_cloud: list[WorldPoint] = []
        for i in range(H):
            for j in range(W):
                point = self.pixel_to_world_point(frame, i, j)
                if point is not None:
                    point_cloud.append(point)
        return point_cloud

    def pixel_to_world_point(
            self,
            frame: Frame,
            i: int,
            j: int,
    ) -> WorldPoint | None:
        world_pos = self.depth_to_world_pos(frame.depth_map, i, j)
        if world_pos is None:
            return None
        rgb_i, rgb_j = self.rgb_camera.world_to_image_pos(world_pos)
        if not is_valid_image_pos(frame.rgb_image, rgb_i, rgb_j):
            return None
        color: np.ndarray = frame.rgb_image[rgb_i, rgb_j]
        return WorldPoint(position=world_pos, color=color)

    def depth_to_world_pos(
            self,
            depth_map: DepthMap,
            i: int,
            j: int,
    ) -> np.ndarray | None:
        depth = depth_map.get(i, j)
        if depth is None:
            return None
        return self.depth_camera.depth_to_world_pos(i, j, depth)
