from common.pointcloud import *
from nyu.nyu_config import *
from PIL import Image
import cv2
import numpy

rgb_a = '../../data/basements/a/r.ppm'
depth_a = '../../data/basements/a/d.pgm'
rgb_b = '../../data/basements/b/r.ppm'
depth_b = '../../data/basements/b/d.pgm'
rgb_c = '../../data/basements/c/r.ppm'
depth_c = '../../data/basements/c/d.pgm'


def _raw_to_depth_nyu(raw: np.ndarray, depth_param_1: float, depth_param_2: float) -> np.ndarray:
    depth = depth_param_1 / (depth_param_2 - raw.astype(np.float32))
    depth[depth > 10] = np.nan
    return depth


def get_frame(rgb_path, depth_path, factor: float = 0):
    rgb = np.asarray(numpy.array(Image.open(rgb_path)))
    raw = np.asarray(cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).byteswap())
    depth = _raw_to_depth_nyu(raw, depthParam1, depthParam2)
    depth = shuffling(depth, factor)
    return Frame(rgb, depth)

def frame_a(factor: float = 0):
    return get_frame(rgb_a, depth_a, factor)

def frame_b(factor: float = 0):
    return get_frame(rgb_b, depth_b, factor)

def frame_c(factor: float = 0):
    return get_frame(rgb_c, depth_c, factor)
