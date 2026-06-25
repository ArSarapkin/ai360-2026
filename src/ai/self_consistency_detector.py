"""Detector that queries each class several times at non-zero temperature and
keeps only the boxes that repeat.

Rationale: the model never abstains, so a single query for an absent class still
returns a confident-looking box -- but a *different* one each time, because the
"answer" is noise. A genuine object, by contrast, is detected in nearly the same
place on every sample. Counting agreeing samples (``vote``) both removes the
hallucinations and yields the real confidence the model itself refuses to give.
"""

from ai.ensemble_detector import EnsembleDetector, QuerySpec
from ai.vlm_query import PROMPT_TEMPLATES


class SelfConsistencyDetector(EnsembleDetector):
    def __init__(
            self,
            classes: list[str],
            n_samples: int = 4,
            temperature: float = 0.7,
            iou_threshold: float = 0.25,
            min_votes: int = 2,
            max_parallel: int = 18,
    ):
        super().__init__(classes, iou_threshold, min_votes, max_parallel)
        self.n_samples = n_samples
        self.temperature = temperature

    def _query_specs(self, target: str, img_base64: str) -> list[QuerySpec]:
        return [
            QuerySpec(img_base64, PROMPT_TEMPLATES[0], self.temperature)
            for _ in range(self.n_samples)
        ]
