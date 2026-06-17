import numpy as np

import nyu.nyu_frame as nyu_frame
from nyu.nyu_camera_params import init_scene
from common.pointcloud import BBox3D, Scene
from common.visualize import visualize
import common.pointcloud_normalization as norm

# Результат, который мне выдал ChatGPT по просьбе найди 3d BBox.
def build_bbox(scene: Scene, R: np.ndarray) -> BBox3D:
    position = np.array([-0.53, -0.39, 3.8])
    size = np.array([1.1, 1.8, 0.55])
    rotation = np.array([
        [0.9888, 0.0, 0.1494],
        [0.0, 1.0, 0.0],
        [-0.1494, 0.0, 0.9888],
    ])
    bbox = BBox3D(position=position, size=size, rotation=rotation)
    return transform_bbox(scene, R, bbox)


def transform_bbox(scene: Scene, R: np.ndarray, bbox: BBox3D) -> BBox3D:
    center = scene.rgb_camera.to_world_pos(bbox.position)
    rotation = scene.rgb_camera.extrinsics.rotation.transpose() @ bbox.rotation
    center = R @ center
    rotation = R @ rotation
    return BBox3D(position=center, size=bbox.size, rotation=rotation)


scene = init_scene()

A = nyu_frame.frame_a(0.02)
B = nyu_frame.frame_b(0.02)
C = nyu_frame.frame_c(0.02)

point_cloud = scene.process_frame(A)

point_cloud, R = norm.normalize(point_cloud)

bbox = build_bbox(scene, R)

visualize(point_cloud, [bbox])

