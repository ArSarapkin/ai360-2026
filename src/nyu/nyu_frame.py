from common.pointcloud import *
from nyu.nyu_config import *
from PIL import Image
import cv2

rgb_a = '../../data/basements/a/r.ppm'
depth_a = '../../data/basements/a/d.pgm'
rgb_b = '../../data/basements/b/r.ppm'
depth_b = '../../data/basements/b/d.pgm'
rgb_c = '../../data/basements/c/r.ppm'
depth_c = '../../data/basements/c/d.pgm'

class DepthMapNYU(DepthMap):
    depth_param_1: float
    depth_param_2: float

    def __init__(self, image, depth_param_1, depth_param_2):
        super().__init__(image)
        self.depth_param_1 = depth_param_1
        self.depth_param_2 = depth_param_2

    def get(self, i, j) -> float | None:
        raw = float(self.image[i, j])
        depth = self.depth_param_1 / (self.depth_param_2 - raw)
        if depth > 10:
            return None
        return depth

def get_frame(rgb_path, depth_path, factor: float = 0):
    rgb = np.array(Image.open(rgb_path))
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth.byteswap()
    depth_map = DepthMapNYU(image=depth, depth_param_1=depthParam1, depth_param_2=depthParam2)
    return Frame(rgb, shuffling(depth_map, factor))

def frame_a(factor: float = 0):
    return get_frame(rgb_a, depth_a, factor)

def frame_b(factor: float = 0):
    return get_frame(rgb_b, depth_b, factor)

def frame_c(factor: float = 0):
    return get_frame(rgb_c, depth_c, factor)