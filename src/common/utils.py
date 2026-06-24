from common.pointcloud import BBox3D


def intersection_volume(b1: BBox3D, b2: BBox3D) -> float:
    overlap = 1.0
    for i in range(3):
        lo = max(b1.position[i] - b1.size[i] / 2, b2.position[i] - b2.size[i] / 2)
        hi = min(b1.position[i] + b1.size[i] / 2, b2.position[i] + b2.size[i] / 2)
        if hi <= lo:
            return 0.0
        overlap *= hi - lo
    return overlap

def IoU(b1: BBox3D, b2: BBox3D) -> float:
    I = intersection_volume(b1, b2)
    U = b1.volume() + b2.volume() - I
    return I / U
