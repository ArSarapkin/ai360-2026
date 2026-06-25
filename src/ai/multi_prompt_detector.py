"""Detector that asks for each class with several different phrasings and keeps
only the boxes that survive across phrasings.

Rationale: the model is locked to one output *format*, but it is still sensitive
to how the instruction is worded. A real object is reported in the same place
whether you say "find", "locate" or "where is"; a hallucination is unstable
under rewording. Unlike self-consistency this needs no temperature support from
the server -- it works even with fully greedy decoding.
"""

from ai.ensemble_detector import EnsembleDetector, QuerySpec
from ai.vlm_query import PROMPT_TEMPLATES


class MultiPromptEnsembleDetector(EnsembleDetector):
    def __init__(
            self,
            classes: list[str],
            templates: list[str] = None,
            iou_threshold: float = 0.25,
            min_votes: int = 2,
            max_parallel: int = 18,
    ):
        super().__init__(classes, iou_threshold, min_votes, max_parallel)
        self.templates = templates if templates is not None else PROMPT_TEMPLATES

    def _query_specs(self, target: str, img_base64: str) -> list[QuerySpec]:
        return [QuerySpec(img_base64, template, 0.0) for template in self.templates]
