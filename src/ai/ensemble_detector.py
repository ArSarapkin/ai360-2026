"""Base class for single-image detectors that fight the model's
always-answers behaviour by *agreement*.

The shape is always the same: for every class, issue several queries that should
all land on a real object but disagree on a hallucination, run them in parallel,
then keep only the boxes enough queries agreed on (``voting.vote``) and drop the
physically impossible ones (``geometry_filter.is_plausible``). Subclasses only
decide *what* the several queries are -- repeated sampling, paraphrases, or
photometric variants -- by overriding :meth:`_query_specs`.

The interface is unchanged: ``detect(img_base64, intrinsics) -> list[BBox3D]``,
one image in, world-agnostic camera-frame boxes out. Cross-frame aggregation
stays where it already is, in the scene-level pipeline.
"""

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import common.pointcloud as pc
from ai.detector import Detector
from ai.vlm_query import query_class_boxes, PROMPT_TEMPLATES
from common.geometry_filter import is_plausible
from common.pointcloud import BBox3D
from common.voting import vote


@dataclass
class QuerySpec:
    """One model call: which image (variant), which phrasing, how random."""
    img_base64: str
    template: str = PROMPT_TEMPLATES[0]
    temperature: float = 0.0


class EnsembleDetector(Detector):
    def __init__(
            self,
            classes: list[str],
            iou_threshold: float = 0.25,
            min_votes: int = 2,
            max_parallel: int = 18,
    ):
        self.classes = classes
        self.iou_threshold = iou_threshold
        self.min_votes = min_votes
        self.max_parallel = max_parallel

    @abstractmethod
    def _query_specs(self, target: str, img_base64: str) -> list[QuerySpec]:
        """Return the set of queries to issue for one class on one image."""

    def detect(self, img_base64: str, intrinsics: pc.Intrinsics) -> list[BBox3D]:
        # (target, spec) jobs flattened so the whole ensemble runs in one pool.
        jobs = []
        n_specs: dict[str, int] = {}
        for target in self.classes:
            specs = self._query_specs(target, img_base64)
            n_specs[target] = len(specs)
            for spec in specs:
                jobs.append((target, spec))

        boxes_by_target: dict[str, list[BBox3D]] = {t: [] for t in self.classes}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {
                executor.submit(
                    query_class_boxes, target, spec.img_base64,
                    spec.template, spec.temperature,
                ): target
                for target, spec in jobs
            }
            for future in as_completed(futures):
                target = futures[future]
                try:
                    boxes_by_target[target].extend(future.result())
                except Exception as e:
                    print(f"[{type(self).__name__}:{target}] job failed: {e}")

        result = []
        for target, boxes in boxes_by_target.items():
            voted = vote(
                boxes,
                total_samples=n_specs[target],
                iou_threshold=self.iou_threshold,
                min_votes=self.min_votes,
            )
            kept = [b for b in voted if is_plausible(b, intrinsics, target)]
            result.extend(kept)
        return result
