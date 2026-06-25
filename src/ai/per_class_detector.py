import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cupy as cp

from ai import ai
from ai.detector import Detector
from common.pointcloud import BBox3D
import common.pointcloud as pc

_output_format = "`{\"bbox_3d\":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]}`"


class PerClassDetector(Detector):
    def __init__(self, classes: list[str], max_parallel: int = 18):
        self.classes = classes
        self.max_parallel = max_parallel

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        def detect_one(target):
            return self._detect_bbox(target, img_base64, intrinsics)

        t_start = time.time()
        print(f"[detect] starting {len(self.classes)} parallel requests")
        result = []
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = [executor.submit(detect_one, t) for t in self.classes]
            for future in as_completed(futures):
                try:
                    result.extend(future.result())
                except Exception as e:
                    print(f"Detection error: {e}")
        print(f"[detect] all done: {len(result)} bboxes, time: {time.time() - t_start:.2f}s")
        return result

    def _detect_bbox(self, target: str, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        t_start = time.time()
        prompt = self._build_prompt(target, intrinsics)
        response = ai.ask(prompt, img_base64)
        try:
            detections = self._parse_detections(response.strip())
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[detect:{target}] JSON error: {e}\nRaw response: {response}")
            return []
        bboxes = []
        for coords, confidence in detections:
            position = cp.array(coords[0:3])
            size = cp.array(coords[3:6])
            roll, pitch, yaw = coords[6], coords[7], coords[8]
            rotation = cp.asarray(ai.angles_to_rotation(roll, pitch, yaw))
            bboxes.append(BBox3D(position=position, size=size, rotation=rotation, confidence=confidence))
        print(f"[{target}] detected {len(bboxes)}, time: {time.time() - t_start:.2f}s")
        return bboxes

    def _build_prompt(self, target: str, intrinsics: pc.Intrinsics) -> str:
        return (
            f"Find the {target} in this image and output ONE 3D bounding box as JSON: "
            f"{_output_format}\n"
        )

    def _parse_detections(self, text: str) -> list[tuple[list, float]]:
        first_bracket = text.find('[')
        first_brace = text.find('{')
        if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
            start, end = first_bracket, text.rfind(']')
            raw = json.loads(text[start:end + 1])
            return [(item["bbox_3d"], item.get("confidence", 1.0)) for item in raw]
        else:
            start, end = first_brace, text.rfind('}')
            raw = json.loads(text[start:end + 1])
            return [(raw["bbox_3d"], raw.get("confidence", 1.0))]
