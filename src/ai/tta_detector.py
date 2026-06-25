"""Detector that queries each class on several photometric variants of the image
and keeps only the boxes stable across them (test-time augmentation).

Rationale: brightness/contrast changes leave the scene geometry untouched, so a
box found on a variant is directly comparable to one found on the original -- no
unreliable monocular un-transform needed. A real object survives a lighting
change; a hallucination, being tied to incidental pixel patterns, often does
not. This probes a different failure axis than self-consistency (input
perturbation rather than sampling noise) and composes well with it.
"""

from ai.ensemble_detector import EnsembleDetector, QuerySpec
from ai.vlm_query import PROMPT_TEMPLATES, photometric_variants


class TTADetector(EnsembleDetector):
    def __init__(
            self,
            classes: list[str],
            factors: list[float] = (0.7, 1.3),
            iou_threshold: float = 0.25,
            min_votes: int = 2,
            max_parallel: int = 18,
    ):
        super().__init__(classes, iou_threshold, min_votes, max_parallel)
        self.factors = list(factors)

    def _query_specs(self, target: str, img_base64: str) -> list[QuerySpec]:
        variants = photometric_variants(img_base64, self.factors)
        return [QuerySpec(v, PROMPT_TEMPLATES[0], 0.0) for v in variants]
