import cv2
import numpy as np

from common.pointcloud import *
from PIL import Image


class DepthMapScannet(DepthMap):
    def get(self, i, j) -> float | None:
        raw = float(self.image[i, j])
        return raw / 1000


def get_intrinsics(scene_path) -> Intrinsics:
    intrinsics_matrix = np.loadtxt(scene_path + '/intrinsic.txt')
    fx, fy, cx, cy = (
        intrinsics_matrix[0, 0],
        intrinsics_matrix[1, 1],
        intrinsics_matrix[0, 2],
        intrinsics_matrix[1, 2]
    )
    return to_intrinsics(fx, fy, cx, cy)


class ScannetScene:
    path: str
    intrinsics: Intrinsics

    def __init__(self, path: str):
        self.path = path
        self.intrinsics = get_intrinsics(path)

    def get_extrinsics(self, frame_id) -> Extrinsics:
        extrinsics_matrix = np.loadtxt(self.path + f'/{frame_id}.txt')
        rotation = extrinsics_matrix[:3, :3].transpose()
        translation = -extrinsics_matrix[:3, 3]
        return Extrinsics(rotation, translation)

    def build_scene(self, frame_id) -> Scene:
        extrinsics = self.get_extrinsics(frame_id)
        distortion = Distortion([0, 0, 0], 0, 0)
        rgb_camera = Camera(self.intrinsics, extrinsics, distortion)
        depth_camera = Camera(self.intrinsics, extrinsics, distortion)
        return Scene(depth_camera, rgb_camera)

    def load_frame(self, frame_id) -> Frame:
        rgb_img = Image.open(self.path + f'/{frame_id}.jpg')
        depth_img = cv2.imread(self.path + f'/{frame_id}.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
        rgb_img = rgb_img.resize((depth_img.shape[1], depth_img.shape[0]))
        rgb = np.array(rgb_img)
        depth = np.array(depth_img)
        depth_map = shuffling(DepthMapScannet(depth), 0.01)
        return Frame(rgb, depth_map)

