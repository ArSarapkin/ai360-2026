import cupy as cp

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


def merge(bboxes: list[BBox3D], threshold: float) -> list[BBox3D]:
    remaining = list(bboxes)
    result = []
    while remaining:
        cluster = [remaining.pop(0)]
        i = 0
        while i < len(remaining):
            if any(IoU(remaining[i], c) >= threshold for c in cluster):
                cluster.append(remaining.pop(i))
            else:
                i += 1
        result.append(_merge_cluster(cluster))
    return result


def _merge_cluster(cluster: list[BBox3D]) -> BBox3D:
    position = cp.mean(cp.stack([b.position for b in cluster]), axis=0)
    size = cp.mean(cp.stack([b.size for b in cluster]), axis=0)
    yaws = cp.array([cp.arctan2(b.rotation[1, 0], b.rotation[0, 0]) for b in cluster])
    yaw = cp.arctan2(cp.mean(cp.sin(yaws)), cp.mean(cp.cos(yaws)))
    cos_y, sin_y = cp.cos(yaw), cp.sin(yaw)
    rotation = cp.eye(3)
    rotation[0, 0] = cos_y;  rotation[0, 1] = -sin_y
    rotation[1, 0] = sin_y;  rotation[1, 1] = cos_y
    confidence = float(cp.mean(cp.array([b.confidence for b in cluster])))
    return BBox3D(position=position, size=size, rotation=rotation, confidence=confidence)
