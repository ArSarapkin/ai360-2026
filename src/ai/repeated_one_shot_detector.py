from concurrent.futures import ThreadPoolExecutor, as_completed

import common.pointcloud as pc
from ai.detector import Detector
from ai.one_shot_detector import OneShotDetector, _parse, _PROMPT
from ai import ai
from common.pointcloud import BBox3D
from common.voting import vote


class RepeatedOneShotDetector(Detector):
    """Run the one-shot prompt N times and keep only boxes that appear in at
    least ``min_votes`` responses (IoU >= ``iou_threshold``).

    Each run may produce a different set of boxes due to sampling noise; real
    objects tend to reappear across runs while hallucinations and loop-artifacts
    do not. This reuses the voting logic from the ensemble detectors but applied
    to all-class responses rather than per-class ones.
    """

    def __init__(
            self,
            n_samples: int = 4,
            temperature: float = 0.7,
            iou_threshold: float = 0.25,
            min_votes: int = 2,
            max_parallel: int = 4,
    ):
        self.n_samples = n_samples
        self.temperature = temperature
        self.iou_threshold = iou_threshold
        self.min_votes = min_votes
        self.max_parallel = max_parallel

    def _single_run(self, img_base64: str) -> list[BBox3D]:
        try:
            response = ai.ask(_PROMPT, img_base64, temperature=self.temperature)
            return _parse(response.strip())
        except Exception as e:
            print(f"[repeated_one_shot] error: {e}")
            return []

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        all_boxes: list[BBox3D] = []
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = [executor.submit(self._single_run, img_base64)
                       for _ in range(self.n_samples)]
            for future in as_completed(futures):
                all_boxes.extend(future.result())

        return vote(
            all_boxes,
            total_samples=self.n_samples,
            iou_threshold=self.iou_threshold,
            min_votes=self.min_votes,
        )
