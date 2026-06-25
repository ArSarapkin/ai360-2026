"""Shared helpers for querying the (black-box) VLM for single-class 3D boxes.

The underlying model has hard constraints that shape every detector built on
top of it (see RESEARCH.md):

* it ALWAYS answers, even when the class is absent (-> false positives);
* it only reliably handles ONE class per request;
* it is locked to the output format ``{"bbox_3d":[x,y,z, dx,dy,dz, r,p,y]}``
  and ignores requests to add confidence or to return an empty answer;
* it cannot do verification / classification ("is there an X?" -> garbage).

So the only levers a single-image detector has are *how it queries the model*
(repeat / perturb the request and look for agreement) and *how it filters the
returned boxes*. This module centralises the query + parse so every detector
shares one, tested implementation.
"""

import base64
import io
import json

import cupy as cp
from PIL import Image, ImageEnhance

from ai import ai
from common.pointcloud import BBox3D

OUTPUT_FORMAT = (
    '`{"bbox_3d":[x_center, y_center, z_center, '
    'x_size, y_size, z_size, roll, pitch, yaw]}`'
)

# A few paraphrases of the same request. The model is locked to one output
# format, but the *phrasing* of the instruction still nudges it differently;
# a genuine object stays put across paraphrases while a hallucination jumps
# around -- which is exactly what the ensemble detectors exploit.
PROMPT_TEMPLATES = [
    "Find the {target} in this image and output ONE 3D bounding box as JSON: {fmt}\n",
    "Locate the {target} and return its 3D bounding box as JSON: {fmt}\n",
    "Detect the {target}. Output the 3D bounding box in JSON: {fmt}\n",
    "Where is the {target}? Give one 3D bounding box as JSON: {fmt}\n",
]


def build_prompt(target: str, template: str = PROMPT_TEMPLATES[0]) -> str:
    return template.format(target=target, fmt=OUTPUT_FORMAT)


def parse_detections(text: str) -> list[list]:
    """Extract the raw ``bbox_3d`` coordinate lists from a model response.

    Tolerates both a single ``{...}`` object and a ``[...]`` array, and ignores
    any prose the model wraps around the JSON. The model never emits a usable
    confidence, so we deliberately do not read one here -- confidence is assigned
    later from cross-sample agreement.
    """
    first_bracket = text.find('[')
    first_brace = text.find('{')
    if first_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        start, end = first_bracket, text.rfind(']')
        raw = json.loads(text[start:end + 1])
        return [item["bbox_3d"] for item in raw]
    else:
        start, end = first_brace, text.rfind('}')
        raw = json.loads(text[start:end + 1])
        return [raw["bbox_3d"]]


def coords_to_bbox(coords: list, confidence: float = 1.0) -> BBox3D:
    position = cp.array(coords[0:3])
    size = cp.array(coords[3:6])
    roll, pitch, yaw = coords[6], coords[7], coords[8]
    rotation = cp.asarray(ai.angles_to_rotation(roll, pitch, yaw))
    return BBox3D(position=position, size=size, rotation=rotation, confidence=confidence)


def query_class_boxes(
        target: str,
        img_base64: str,
        template: str = PROMPT_TEMPLATES[0],
        temperature: float = 0.0,
        model: str = "qwen-vl-max",
) -> list[BBox3D]:
    """One model call for one class; returns the parsed boxes (may be empty on
    a malformed response). All errors are swallowed so a single bad sample never
    kills an ensemble."""
    prompt = build_prompt(target, template)
    try:
        response = ai.ask(prompt, img_base64, model=model, temperature=temperature)
        coords_list = parse_detections(response.strip())
    except (json.JSONDecodeError, KeyError, ValueError, IndexError, TypeError) as e:
        print(f"[query:{target}] parse/response error: {e}")
        return []
    boxes = []
    for coords in coords_list:
        try:
            boxes.append(coords_to_bbox(coords))
        except (ValueError, IndexError, TypeError) as e:
            print(f"[query:{target}] bad coords {coords}: {e}")
    return boxes


def _decode(img_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(img_base64))).convert("RGB")


def _encode(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def downscale(img_base64: str, max_side: int) -> str:
    """Re-encode the image so its longest side is at most ``max_side`` px.

    Gate calls only need a coarse "is this kind of thing here" signal, so feeding
    the model a smaller image cuts the (image-token-dominated) cost of each gate
    call without changing what we ask. Returns the input unchanged if it is
    already small enough.
    """
    img = _decode(img_base64)
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img_base64
    return _encode(img.resize((max(1, int(w * scale)), max(1, int(h * scale)))))


def photometric_variants(img_base64: str, factors: list[float]) -> list[str]:
    """Build brightness/contrast-perturbed copies of the image.

    Photometric only -- the geometry is left untouched, so 3D boxes detected on
    a variant are directly comparable to those on the original without any
    coordinate un-transform (which would be unreliable for monocular 3D). The
    original image is always included as the first variant.
    """
    base = _decode(img_base64)
    variants = [img_base64]
    for f in factors:
        v = ImageEnhance.Brightness(base).enhance(f)
        v = ImageEnhance.Contrast(v).enhance(2.0 - f)
        variants.append(_encode(v))
    return variants
