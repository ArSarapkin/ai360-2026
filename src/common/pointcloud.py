import time
from dataclasses import dataclass

import numpy as np


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


@dataclass
class Camera:
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    def to_world_pos(self, camera_pos: np.ndarray) -> np.ndarray:
        return self.extrinsics.rotation.transpose() @ (camera_pos - self.extrinsics.translation)

    def to_camera_pos(self, world_pos: np.ndarray) -> np.ndarray:
        return (self.extrinsics.rotation @ world_pos) + self.extrinsics.translation

    def to_Z1(self, i, j) -> tuple[float, float]:
        x = (j - self.intrinsics.cx) / self.intrinsics.fx
        y = (i - self.intrinsics.cy) / self.intrinsics.fy
        return x, y

    def from_Z1(self, x, y) -> tuple[int, int]:
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

    def depths_to_world_positions(self, ij: np.ndarray, depths: np.ndarray) -> np.ndarray:
        x = (ij[:, 1] - self.intrinsics.cx) / self.intrinsics.fx
        y = (ij[:, 0] - self.intrinsics.cy) / self.intrinsics.fy
        camera_pos = np.stack([x * depths, y * depths, depths], axis=1)  # (N, 3)
        return (camera_pos - self.extrinsics.translation) @ self.extrinsics.rotation

    def world_to_image_positions(self, world_positions: np.ndarray) -> np.ndarray:
        camera_pos = world_positions @ self.extrinsics.rotation.T + self.extrinsics.translation  # (N, 3)
        x = camera_pos[:, 0] / camera_pos[:, 2]
        y = camera_pos[:, 1] / camera_pos[:, 2]
        i = np.round(self.intrinsics.fy * y + self.intrinsics.cy).astype(np.int32)
        j = np.round(self.intrinsics.fx * x + self.intrinsics.cx).astype(np.int32)
        return np.stack([i, j], axis=1)  # (N, 2)


@dataclass
class WorldPoint:
    position: np.ndarray  # 3
    color: np.ndarray  # 3


def is_valid_image_pos(image: np.ndarray, i: int, j: int) -> bool:
    H, W = image.shape[:2]
    return 0 <= i < H and 0 <= j < W


def shuffling(depth: np.ndarray, factor: float) -> np.ndarray:
    noise = np.random.uniform(-factor, factor, size=depth.shape).astype(depth.dtype)
    return depth + noise


@dataclass
class Frame:
    rgb_image: np.ndarray
    depth: np.ndarray


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
        H, W = frame.depth.shape[:2]

        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        ij_all = np.stack([ii.ravel(), jj.ravel()], axis=1)  # (H*W, 2)
        depths_all = frame.depth.ravel()  # (H*W,)

        valid_mask = np.isfinite(depths_all)
        ij = ij_all[valid_mask]
        depths = depths_all[valid_mask]

        world_positions = self.depth_camera.depths_to_world_positions(ij, depths)  # (N, 3)

        rgb_ij = self.rgb_camera.world_to_image_positions(world_positions)  # (N, 2)

        H_rgb, W_rgb = frame.rgb_image.shape[:2]
        in_bounds = (
                (rgb_ij[:, 0] >= 0) & (rgb_ij[:, 0] < H_rgb) &
                (rgb_ij[:, 1] >= 0) & (rgb_ij[:, 1] < W_rgb)
        )
        world_positions = world_positions[in_bounds]
        rgb_ij = rgb_ij[in_bounds]

        colors = frame.rgb_image[rgb_ij[:, 0], rgb_ij[:, 1]]  # (N, 3)

        point_cloud = [
            WorldPoint(position=world_positions[k], color=colors[k])
            for k in range(len(world_positions))
        ]

        return point_cloud

    def pixel_to_world_point(
            self,
            frame: Frame,
            i: int,
            j: int,
    ) -> WorldPoint | None:
        world_pos = self.depth_to_world_pos(frame.depth, i, j)
        if world_pos is None:
            return None
        rgb_i, rgb_j = self.rgb_camera.world_to_image_pos(world_pos)
        if not is_valid_image_pos(frame.rgb_image, rgb_i, rgb_j):
            return None
        color: np.ndarray = frame.rgb_image[rgb_i, rgb_j]
        return WorldPoint(position=world_pos, color=color)

    def depth_to_world_pos(
            self,
            depth: np.ndarray,
            i: int,
            j: int,
    ) -> np.ndarray | None:
        d = depth[i, j]
        if not np.isfinite(d):
            return None
        return self.depth_camera.depth_to_world_pos(i, j, d)

    def bbox_camera_to_world(self, R: np.ndarray, bbox: BBox3D) -> BBox3D:
        center = self.rgb_camera.to_world_pos(bbox.position)
        rotation = self.rgb_camera.extrinsics.rotation.transpose() @ bbox.rotation
        center = R @ center
        rotation = R @ rotation
        return BBox3D(position=center, size=bbox.size, rotation=rotation)
