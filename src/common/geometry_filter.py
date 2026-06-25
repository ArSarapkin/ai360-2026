"""Geometric plausibility checks for a single camera-frame 3D box.

These run inside one ``detect()`` call, where we have only the image and its
intrinsics (no depth, no other frames). They cannot tell a real object from a
hallucination, but they cheaply reject boxes that are physically impossible --
behind the camera, projecting outside the image, or the wrong size for their
class -- which the always-answering model produces in abundance.
"""

from common.pointcloud import Intrinsics, BBox3D

# Loose upper bound on the largest dimension of each ScanNet class, in metres.
# Used to discard obviously-wrong boxes (e.g. a 6 m "chair"). Anything not
# listed falls back to GLOBAL_MAX_SIZE.
SIZE_PRIORS = {
    "cabinet": 2.5, "bed": 2.5, "chair": 1.4, "sofa": 3.0, "table": 3.0,
    "door": 2.6, "window": 3.0, "bookshelf": 3.0, "picture": 2.0, "counter": 4.0,
    "desk": 3.0, "curtain": 3.5, "refrigerator": 2.2, "showercurtrain": 2.6,
    "toilet": 1.0, "sink": 1.2, "bathtub": 2.2, "garbagebin": 1.0,
}
GLOBAL_MAX_SIZE = 4.0
GLOBAL_MIN_SIZE = 0.05


def max_size_for(target: str | None) -> float:
    if target is None:
        return GLOBAL_MAX_SIZE
    return SIZE_PRIORS.get(target, GLOBAL_MAX_SIZE)


def _size_ok(bbox: BBox3D, min_size: float, max_size: float) -> bool:
    dims = [float(bbox.size[i]) for i in range(3)]
    if any((d != d) for d in dims):  # NaN guard
        return False
    if any(d <= 0 for d in dims):
        return False
    return min_size <= max(dims) <= max_size


def _in_front_and_visible(bbox: BBox3D, intrinsics: Intrinsics) -> bool:
    x, y, z = (float(bbox.position[i]) for i in range(3))
    if not (z > 0):  # behind camera or degenerate
        return False
    u = intrinsics.fx * x / z + intrinsics.cx
    v = intrinsics.fy * y / z + intrinsics.cy
    # Approximate image extent from the principal point (cx ~ W/2, cy ~ H/2),
    # widened by 50% so we only reject boxes that are clearly off-frame.
    w, h = 2 * intrinsics.cx, 2 * intrinsics.cy
    return -0.5 * w <= u <= 1.5 * w and -0.5 * h <= v <= 1.5 * h


def is_plausible(
        bbox: BBox3D,
        intrinsics: Intrinsics,
        target: str | None = None,
        min_size: float = GLOBAL_MIN_SIZE,
) -> bool:
    return (
        _size_ok(bbox, min_size, max_size_for(target))
        and _in_front_and_visible(bbox, intrinsics)
    )
