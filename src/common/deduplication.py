import cupy as cp

from common.pointcloud import BBox3D
from common.utils import IoU
from common.voting import _average_rotations


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
    rotation = _average_rotations([b.rotation for b in cluster])
    confidence = float(cp.mean(cp.array([b.confidence for b in cluster])))
    return BBox3D(position=position, size=size, rotation=rotation, confidence=confidence)
