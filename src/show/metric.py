import base64
import io
import time

from PIL import Image

from ai.per_class_detector import PerClassDetector
from common.deduplication import NMS, merge
from common.pointcloud import apply_axis_align
from common.utils import IoU
from common.visualize import visualize
from scannet.labels import LabelsLoader
from scannet.scannet_config import ScannetScene
from scannet.utils import scene_pointcloud

scene_name = 'scene0000_00'
data_path = '../../data/scannet'

scannet_scene = ScannetScene(f'{data_path}/posed_images/{scene_name}')
point_cloud = scene_pointcloud(scene_name)

images = range(0, 300, 9)

bboxes = []

labels = LabelsLoader()
axis_R, axis_t = labels.load_axis_align(scene_name)
point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin']

detector = PerClassDetector(classes_scannet, max_parallel=18)


def detect_all(img_id, scene, axis_R, axis_t):
    t_start = time.time()

    img_id = str(img_id).zfill(5)
    t = time.time()
    img = Image.open(f'{data_path}/posed_images/{scene_name}/{img_id}.jpg')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    print(f"[{img_id}] encoded image: {time.time() - t:.2f}s")

    bboxes = [scene.bbox_camera_to_world(bbox, axis_R, axis_t)
              for bbox in detector.detect(img_base64, scene.rgb_camera.intrinsics)]

    print(f"[{img_id}] detect_all: {len(bboxes)} bboxes, time: {time.time() - t_start:.2f}s")
    return bboxes


t0 = time.time()

for img_id in images:
    img_id = str(img_id).zfill(5)
    scene = scannet_scene.build_scene(img_id)
    img_bboxes = detect_all(img_id, scene, axis_R, axis_t)
    bboxes.extend(img_bboxes)

print(f"total detection time: {time.time() - t0:.2f}s, total bboxes before NMS: {len(bboxes)}")

t = time.time()
bboxes = merge(bboxes, 0.25)
print(f"NMS: {len(bboxes)} bboxes remaining, time: {time.time() - t:.2f}s")

t = time.time()
expected = labels.load_bboxes(scene_name)
print(f"loaded {len(expected)} GT bboxes: {time.time() - t:.2f}s")

def calc_metrics(bboxes, expected, threshold):
    TP, FP, FN = 0, 0, len(expected)
    used = [False] * len(expected)
    for bbox in bboxes:
        id = -1
        best_iou = threshold
        for i in range(len(expected)):
            if used[i]:
                continue
            iou = IoU(bbox, expected[i])
            if iou >= best_iou:
                best_iou = iou
                id = i
        if id != -1:
            TP += 1
            used[id] = True
            FN -= 1
        else:
            FP += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

p, r, f = calc_metrics(bboxes, expected, 0.25)

print(f"precision: {p:.2f}")
print(f"recall: {r:.2f}")
print(f"f1-score: {f:.2f}")

visualize(point_cloud, bboxes)
