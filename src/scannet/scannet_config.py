import cv2
import cupy as np
import numpy

from common.pointcloud import *
from PIL import Image


def _raw_to_depth_scannet(raw: np.ndarray) -> np.ndarray:
    return raw.astype(np.float32) / 1000


def get_intrinsics(scene_path) -> Intrinsics:
    intrinsics_matrix = numpy.loadtxt(scene_path + '/intrinsic.txt')
    fx, fy, cx, cy = (
        intrinsics_matrix[0, 0],
        intrinsics_matrix[1, 1],
        intrinsics_matrix[0, 2],
        intrinsics_matrix[1, 2],
    )
    return to_intrinsics(fx, fy, cx, cy)


class ScannetScene:
    path: str
    intrinsics: Intrinsics

    def __init__(self, path: str):
        self.path = path
        self.intrinsics = get_intrinsics(path)

    def get_intrinsics(self, scale_x, scale_y):
        return to_intrinsics(
            fx=self.intrinsics.fx * scale_x,
            cx=self.intrinsics.cx * scale_x,
            fy=self.intrinsics.fy * scale_y,
            cy=self.intrinsics.cy * scale_y,
        )

    def get_extrinsics(self, frame_id) -> Extrinsics:
        extrinsics_matrix = numpy.loadtxt(self.path + f'/{frame_id}.txt')
        rotation = np.asarray(extrinsics_matrix[:3, :3].transpose())
        translation = -rotation @ np.asarray(extrinsics_matrix[:3, 3])
        return Extrinsics(rotation, translation)

    def build_scene(self, frame_id) -> Scene:
        rgb_img = Image.open(self.path + f'/{frame_id}.jpg')
        depth_img = cv2.imread(self.path + f'/{frame_id}.png', cv2.IMREAD_UNCHANGED)
        scale_x = depth_img.shape[1] / rgb_img.width
        scale_y = depth_img.shape[0] / rgb_img.height
        intrinsics = self.get_intrinsics(scale_x, scale_y)
        extrinsics = self.get_extrinsics(frame_id)
        rgb_camera = Camera(intrinsics, extrinsics)
        depth_camera = Camera(intrinsics, extrinsics)
        return Scene(depth_camera, rgb_camera)

    def load_frame(self, frame_id) -> Frame:
        rgb_img = Image.open(self.path + f'/{frame_id}.jpg')
        depth_img = cv2.imread(self.path + f'/{frame_id}.png', cv2.IMREAD_UNCHANGED)
        rgb_img = rgb_img.resize((depth_img.shape[1], depth_img.shape[0]))
        rgb = np.asarray(numpy.array(rgb_img))
        depth = _raw_to_depth_scannet(np.asarray(depth_img.astype(numpy.float32)))
        depth = shuffling(depth, 0.01)
        return Frame(rgb, depth)
