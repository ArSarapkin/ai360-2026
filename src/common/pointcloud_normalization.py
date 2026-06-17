import numpy as np
import open3d as o3d
from common.pointcloud import *
from common.open3d_utils import *
import scipy.spatial.transform as transform


def floor_normal(point_cloud: list[WorldPoint]):
    o3d_point_cloud = pointcloud_open3d(point_cloud)
    plane_model, _ = o3d_point_cloud.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    return normal


def floor_rotation(point_cloud: list[WorldPoint]):
    normal = floor_normal(point_cloud)
    R, _ = transform.Rotation.align_vectors([[0, 0, 1]], [normal])
    return R.as_matrix()


def normalize_point(point: WorldPoint, rotation: np.ndarray) -> WorldPoint:
    new_position = (rotation @ point.position.transpose()).transpose()
    return WorldPoint(new_position, point.color)


def normalize(point_cloud: list[WorldPoint]) -> tuple[list[WorldPoint], np.ndarray]:  # normalized, rotation
    R = floor_rotation(point_cloud)
    normalized = [normalize_point(p, R) for p in point_cloud]
    return normalized, R


def normalize_transpose(point_cloud: list[WorldPoint]) -> tuple[list[WorldPoint], np.ndarray]:
    transpose = np.zeros(3)
    for point in point_cloud:
        transpose -= point.position
    transpose /= len(point_cloud)
    normalized = []
    for point in point_cloud:
        normalized.append(
            WorldPoint(
                position=point.position + transpose,
                color=point.color
            )
        )
    return normalized, transpose
