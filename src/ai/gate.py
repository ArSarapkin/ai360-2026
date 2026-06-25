"""Presence gates that decide *which* classes to detect, using ONLY the same
Qwen3-VL model -- no auxiliary classifier, no training.

Why this is limited by design
-----------------------------
The model's output format carries no class label: a single call returns boxes,
not names. So one call can never tell us *which* of the 18 classes are present --
the only way to ask about class X is to name X in the prompt, which is inherently
one call per class. A sub-`C` gate is therefore impossible to get "for free" from
this model; the literature answer (a cheap external tagger like RAM++/CLIP/
YOLO-World) is ruled out by the "Qwen3 only" constraint.

What *is* achievable
--------------------
* **Group gating** (`SuperCategoryGate`): one call asks for a whole super-category
  ("seating furniture such as a chair or sofa"). If the returned box is
  implausible we skip *all* members of that group at once, so a handful of group
  calls can prune many classes. This is the only construction that can push the
  total below one-call-per-class.
* **Cheap per-class probing** (`VlmProbeGate`): still one (downscaled, cheap) call
  per class, but used purely as a plausibility test so the *expensive* detector
  (e.g. an ensemble) runs only on survivors. Reduces cost, not call count.

Both lean toward recall: a gate that wrongly drops a present class loses that
object forever, so when unsure we keep the class.
"""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import common.pointcloud as pc
from ai.vlm_query import query_class_boxes, downscale
from common.geometry_filter import is_plausible

# Super-categories for the 18 ScanNet classes. The key is the phrase shown to the
# model; the value is the member classes pruned/kept together. Singletons are
# included without a separate gate call (gating them would cost two calls for one
# class) unless ``gate_singletons=True``.
SCANNET_SUPER_CATEGORIES = {
    "seating furniture such as a chair or sofa": ["chair", "sofa"],
    "a table, desk or counter": ["table", "desk", "counter"],
    "a cabinet, bookshelf or other storage furniture": ["cabinet", "bookshelf"],
    "a door or window": ["door", "window"],
    "a bathroom fixture such as a toilet, sink, bathtub or shower curtain":
        ["toilet", "sink", "bathtub", "showercurtrain"],
    "a picture or curtain on the wall": ["picture", "curtain"],
    "a bed": ["bed"],
    "a refrigerator": ["refrigerator"],
    "a garbage bin": ["garbagebin"],
}


class Gate(ABC):
    @abstractmethod
    def select(
            self,
            img_base64: str,
            intrinsics: pc.Intrinsics,
            classes: list[str],
    ) -> list[str]:
        """Return the subset of ``classes`` worth running detection on."""


class SuperCategoryGate(Gate):
    """One gate call per multi-member super-category; keep a category's classes
    only if its box is geometrically plausible."""

    def __init__(
            self,
            categories: dict[str, list[str]] = None,
            probe_max_side: int = 0,
            gate_singletons: bool = False,
            max_parallel: int = 9,
    ):
        self.categories = categories if categories is not None else SCANNET_SUPER_CATEGORIES
        self.probe_max_side = probe_max_side
        self.gate_singletons = gate_singletons
        self.max_parallel = max_parallel

    def select(self, img_base64, intrinsics, classes) -> list[str]:
        wanted = set(classes)
        img = downscale(img_base64, self.probe_max_side) if self.probe_max_side else img_base64

        selected: set[str] = set()
        to_probe: list[tuple[str, list[str]]] = []
        for phrase, members in self.categories.items():
            members_in = [c for c in members if c in wanted]
            if not members_in:
                continue
            # Only skip gating for categories *defined* as a single class (gating
            # one class would cost two calls for one detection). A multi-member
            # category that merely narrows to one in-scope class is still gated,
            # so it can be pruned.
            if len(members) == 1 and not self.gate_singletons:
                selected.update(members_in)
            else:
                to_probe.append((phrase, members_in))

        def probe(item):
            phrase, members_in = item
            boxes = query_class_boxes(phrase, img)
            present = any(is_plausible(b, intrinsics, target=None) for b in boxes)
            return members_in if present else []

        if to_probe:
            with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
                for kept in executor.map(probe, to_probe):
                    selected.update(kept)
        return sorted(selected)


class VlmProbeGate(Gate):
    """One cheap (optionally downscaled) call per class, kept if its box is a
    plausible instance of that class. Does not reduce the number of calls, but
    lets an expensive base detector run only on plausible classes."""

    def __init__(self, probe_max_side: int = 256, max_parallel: int = 18):
        self.probe_max_side = probe_max_side
        self.max_parallel = max_parallel

    def select(self, img_base64, intrinsics, classes) -> list[str]:
        img = downscale(img_base64, self.probe_max_side) if self.probe_max_side else img_base64

        def probe(target):
            boxes = query_class_boxes(target, img)
            return target if any(is_plausible(b, intrinsics, target) for b in boxes) else None

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            kept = [t for t in executor.map(probe, classes) if t is not None]
        return kept
