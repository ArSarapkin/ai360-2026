import base64
import io
import os
import sys
import time

from PIL import Image

from ai.baseline_detector import BaselineDetector
from ai.one_shot_detector import OneShotDetector
from ai.repeated_one_shot_detector import RepeatedOneShotDetector
from ai.per_class_detector import PerClassDetector
from ai.plausibility_detector import PlausibilityFilterDetector
from ai.self_consistency_detector import SelfConsistencyDetector
from ai.multi_prompt_detector import MultiPromptEnsembleDetector
from ai.tta_detector import TTADetector
from ai.gate import SuperCategoryGate
from ai.gated_detector import GatedDetector
from common.deduplication import merge, NMS
from common.pointcloud import apply_axis_align
from common.utils import IoU
from common.visualize import visualize
from scannet.labels import LabelsLoader
from scannet.scannet_config import ScannetScene
from scannet.utils import scene_pointcloud

data_path = '../../data/scannet'

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin']

DETECTORS = {
    # 0) Random baseline: 4 random boxes per image, no model calls.
    "baseline": lambda: BaselineDetector(n_boxes=10),

    # 5) One prompt for all objects at once.
    "one_shot": lambda: OneShotDetector(),

    # 6) Repeat the one-shot prompt N times, keep boxes that reappear.
    "repeated_one_shot": lambda: RepeatedOneShotDetector(n_samples=6, min_votes=2),

    # 1) Prompt-space ensemble: same image, several paraphrases, keep boxes that
    #    agree. Works with greedy decoding (no temperature needed).
    "multi_prompt": lambda: MultiPromptEnsembleDetector(
        classes_scannet, min_votes=2, max_parallel=18),

    # 2) Input-space ensemble: photometric variants of the image, keep boxes that
    #    survive a lighting change.
    "tta": lambda: TTADetector(
        classes_scannet, factors=[0.7, 1.3], min_votes=2, max_parallel=18),

    # 3) Sampling-space ensemble + geometry filter: repeated temperature samples
    #    voted, then physically-impossible boxes dropped.
    "self_consistency": lambda: PlausibilityFilterDetector(
        SelfConsistencyDetector(
            classes_scannet, n_samples=4, temperature=0.7, min_votes=2, max_parallel=18)),

    # 4) Cost-reducing gate: one cheap call per super-category prunes whole groups
    #    of classes, then a filtered single-shot detector runs only on survivors.
    "gated": lambda: GatedDetector(
        classes_scannet,
        SuperCategoryGate(probe_max_side=512),
        base_factory=lambda cls: PlausibilityFilterDetector(PerClassDetector(cls)),
    ),

    "default": lambda: PerClassDetector(classes_scannet)
}

# Pick the detector for this run: edit DETECTOR, or pass a name as a CLI argument
# (e.g. `python -m show.metric tta`).
DETECTOR = "self_consistency"
if len(sys.argv) > 1:
    DETECTOR = sys.argv[1]
if DETECTOR not in DETECTORS:
    raise SystemExit(f"unknown detector '{DETECTOR}'; choose one of {list(DETECTORS)}")


def encode_image(scene_name: str, img_id: str) -> str:
    img = Image.open(f'{data_path}/posed_images/{scene_name}/{img_id}.jpg')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_frames(scene_name: str, scannet_scene: ScannetScene, images):
    frames = []
    for img_id in images:
        img_id = str(img_id).zfill(5)
        frames.append((img_id, scannet_scene.build_scene(img_id), encode_image(scene_name, img_id)))
    return frames


def detect_scene(detector, frames, axis_R, axis_t):
    t_start = time.time()
    world_bboxes = []
    for img_id, scene, img_base64 in frames:
        cam_bboxes = detector.detect(img_base64, scene.rgb_camera.intrinsics)
        world_bboxes.extend(
            scene.bbox_camera_to_world(bbox, axis_R, axis_t) for bbox in cam_bboxes
        )
    merged = NMS(world_bboxes, 0.25)
    print(f"  detected {len(world_bboxes)} -> {len(merged)} after merge, "
          f"{time.time() - t_start:.1f}s")
    return merged


def calc_metrics(bboxes, expected, threshold):
    TP, FP, FN = 0, 0, len(expected)
    used = [False] * len(expected)
    for bbox in bboxes:
        best_id = -1
        best_iou = threshold
        for i in range(len(expected)):
            if used[i]:
                continue
            iou = IoU(bbox, expected[i])
            if iou >= best_iou:
                best_iou = iou
                best_id = i
        if best_id != -1:
            TP += 1
            used[best_id] = True
            FN -= 1
        else:
            FP += 1

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


# All scenes available in posed_images/, sorted.
all_scenes = sorted(
    d for d in os.listdir(f'{data_path}/posed_images')
    if os.path.isdir(f'{data_path}/posed_images/{d}')
)
if not all_scenes:
    raise SystemExit("No scenes found in data/scannet/posed_images/")

labels = LabelsLoader()
images = range(0, 300, 10)

print(f"detector: {DETECTOR}")
print(f"scenes:   {len(all_scenes)}")
print()

accumulated_p = []
accumulated_r = []

vis = True

for scene_idx, scene_name in enumerate(all_scenes, 1):
    print(f"=== scene {scene_idx}/{len(all_scenes)}: {scene_name} ===")

    scannet_scene = ScannetScene(f'{data_path}/posed_images/{scene_name}')
    point_cloud = scene_pointcloud(scene_name)
    axis_R, axis_t = labels.load_axis_align(scene_name)
    point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

    frames = build_frames(scene_name, scannet_scene, images)
    detector = DETECTORS[DETECTOR]()

    boxes = detect_scene(detector, frames, axis_R, axis_t)
    expected = labels.load_bboxes(scene_name)
    p, r, f = calc_metrics(boxes, expected, 0.25)

    print(f"  [{scene_name}] P={p:.2f}  R={r:.2f}  F1={f:.2f}  "
          f"boxes={len(boxes)}  gt={len(expected)}")

    accumulated_p.append(p)
    accumulated_r.append(r)

    mean_p = sum(accumulated_p) / len(accumulated_p)
    mean_r = sum(accumulated_r) / len(accumulated_r)
    mean_f = (2 * mean_p * mean_r / (mean_p + mean_r)) if (mean_p + mean_r) else 0.0

    print(f"  [cumulative {scene_idx} scenes]  "
          f"P={mean_p:.2f}  R={mean_r:.2f}  F1={mean_f:.2f}")
    print()

    if(vis):
        visualize(point_cloud, boxes)
