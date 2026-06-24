from common.pointcloud import BBox3D
from common.utils import IoU


def NMS(bboxes: list[BBox3D], threshold: float) -> list[BBox3D]:
    bboxes = sorted(bboxes, key=lambda b: b.confidence, reverse=True)
    result = []
    while bboxes:
        best = bboxes.pop(0)
        result.append(best)
        bboxes = [b for b in bboxes if IoU(best, b) < threshold]
    return result
