"""Cluster boxes that came from repeated/perturbed queries and turn the size of
each cluster into a real confidence.

The model never returns a usable confidence (it is locked to one output format),
so downstream NMS / merge always saw ``confidence == 1.0``. By sampling the same
class several times and counting how many samples agree on a box, we recover a
meaningful confidence: a true object is detected consistently, a hallucination
lands in a different place every time.
"""

import cupy as cp

from common.pointcloud import BBox3D
from common.utils import IoU


def _average_rotations(rotations: list) -> cp.ndarray:
    """Geodesic mean of rotation matrices via SVD projection.

    Average the matrices element-wise, then project the result back onto SO(3)
    with U @ Vt. Correct for any combination of roll/pitch/yaw, unlike
    arctan2-based yaw extraction which breaks when pitch leaves ±90°.
    """
    R_mean = cp.mean(cp.stack(rotations), axis=0)
    U, _, Vt = cp.linalg.svd(R_mean)
    R = U @ Vt
    # Flip last column if det = -1 (reflection, not rotation).
    if cp.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def cluster_boxes(boxes: list[BBox3D], iou_threshold: float) -> list[list[BBox3D]]:
    """Greedy single-linkage clustering by 3D IoU (same scheme as
    ``deduplication.merge``)."""
    remaining = list(boxes)
    clusters = []
    while remaining:
        cluster = [remaining.pop(0)]
        i = 0
        while i < len(remaining):
            if any(IoU(remaining[i], c) >= iou_threshold for c in cluster):
                cluster.append(remaining.pop(i))
            else:
                i += 1
        clusters.append(cluster)
    return clusters


def _merge_cluster(cluster: list[BBox3D], confidence: float) -> BBox3D:
    position = cp.mean(cp.stack([b.position for b in cluster]), axis=0)
    size = cp.mean(cp.stack([b.size for b in cluster]), axis=0)
    rotation = _average_rotations([b.rotation for b in cluster])
    return BBox3D(position=position, size=size, rotation=rotation, confidence=confidence)


def vote(
        boxes: list[BBox3D],
        total_samples: int,
        iou_threshold: float = 0.25,
        min_votes: int = 2,
) -> list[BBox3D]:
    """Keep only boxes that ``min_votes`` independent samples agreed on.

    Each surviving cluster is merged into one representative box whose
    confidence is ``votes / total_samples`` (clamped to 1.0). This is the core
    anti-hallucination step: scattered, single-vote boxes are dropped.
    """
    result = []
    for cluster in cluster_boxes(boxes, iou_threshold):
        votes = len(cluster)
        if votes >= min_votes:
            confidence = min(votes / max(total_samples, 1), 1.0)
            result.append(_merge_cluster(cluster, confidence))
    return result
