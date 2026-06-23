import cupy as np
import numpy
from common.visualize import visualize


def load_scan(pcd_path) -> np.ndarray:
    return np.asarray(numpy.fromfile(pcd_path, dtype=numpy.float32).reshape(-1, 6))
