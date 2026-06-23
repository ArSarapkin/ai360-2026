import cupy as np
import numpy
import open3d as o3d
from common.pointcloud import *
from common.open3d_utils import *
import scipy.spatial.transform as transform


def floor_normal(point_cloud: np.ndarray):
    o3d_point_cloud = pointcloud_open3d(point_cloud)
    plane_model, _ = o3d_point_cloud.segment_plane(
        distance_threshold=0.02,
        ransac_n=3,
        num_iterations=1000
    )
    normal = np.array(plane_model[:3])
    normal /= np.linalg.norm(normal)
    return normal


def floor_rotation(point_cloud: np.ndarray):
    normal = floor_normal(point_cloud)
    R, _ = transform.Rotation.align_vectors([[0, 0, 1]], [numpy.asarray(normal)])
    return np.asarray(R.as_matrix())


def normalize(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = floor_rotation(point_cloud)
    normalized = point_cloud.copy()
    normalized[:, :3] = (R @ point_cloud[:, :3].T).T
    return normalized, R


def normalize_transpose(point_cloud: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = -point_cloud[:, :3].mean(axis=0)
    normalized = point_cloud.copy()
    normalized[:, :3] = point_cloud[:, :3] + t
    return normalized, t
