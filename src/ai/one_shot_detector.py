import json

import common.pointcloud as pc
from ai import ai
from ai.detector import Detector
from ai.vlm_query import OUTPUT_FORMAT, coords_to_bbox
from common.pointcloud import BBox3D

_PROMPT = (
    "Detect the most prominent objects in this image (at most 4, largest and most visible only). "
    "Output a JSON array of 3D bounding boxes: "
    '`[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]}, ...]`\n'
)


def _parse(text: str) -> list[BBox3D]:
    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        # model returned a single object instead of array
        start = text.find('{')
        end = text.rfind('}')
        if start == -1:
            return []
        raw = json.loads(text[start:end + 1])
        return [coords_to_bbox(raw["bbox_3d"])]
    items = json.loads(text[start:end + 1])
    if not isinstance(items, list):
        return []
    boxes = []
    for item in items:
        try:
            boxes.append(coords_to_bbox(item["bbox_3d"]))
        except (KeyError, ValueError, TypeError):
            continue
    return boxes


class OneShotDetector(Detector):
    """Ask the model to detect everything in one prompt.

    Known risk: small models tend to loop and emit the same box repeatedly on
    long structured output. Included as a comparison point.
    """

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        try:
            response = ai.ask(_PROMPT, img_base64)
            return _parse(response.strip())
        except Exception as e:
            print(f"[one_shot] error: {e}")
            return []
