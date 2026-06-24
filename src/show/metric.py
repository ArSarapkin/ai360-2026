import base64
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai import ai
from common.nms import NMS
from common.pointcloud import apply_axis_align
from common.visualize import visualize
from scannet.labels import LabelsLoader
from scannet.scannet_config import ScannetScene
from scannet.utils import scene_pointcloud
from PIL import Image

scene_name = 'scene0000_00'
data_path = '../../data/scannet'

scannet_scene = ScannetScene(f'{data_path}/posed_images/{scene_name}')
point_cloud = scene_pointcloud(scene_name)

images = range(0, 10, 2)

bboxes = []

labels = LabelsLoader()
axis_R, axis_t = labels.load_axis_align(scene_name)
point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin']

MAX_PARALLEL = 18

def detect_all(img_id, scene, axis_R, axis_t):
    t_start = time.time()

    img_id = str(img_id).zfill(5)
    img = Image.open(f'{data_path}/posed_images/{scene_name}/{img_id}.jpg')
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    def detect_one(target):
        bboxes = ai.detect_bbox(target, img_base64, scene.rgb_camera.intrinsics)
        return [scene.bbox_camera_to_world(bbox, axis_R, axis_t) for bbox in bboxes]

    bboxes = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = [executor.submit(detect_one, t) for t in classes_scannet]
        for future in as_completed(futures):
            try:
                bboxes.extend(future.result())
            except Exception as e:
                print(f"Detection error: {e}")

    t_stop = time.time()
    print(f"detected {len(bboxes)}, time: {t_stop - t_start:.2f}s")

    return bboxes


t0 = time.time()

for img_id in images:
    img_id = str(img_id).zfill(5)
    scene = scannet_scene.build_scene(img_id)
    img_bboxes = detect_all(img_id, scene, axis_R, axis_t)
    bboxes.extend(img_bboxes)

total_time = time.time() - t0
print(f"total time: {total_time:.2f}s")

bboxes = NMS(bboxes, 0.25)

visualize(point_cloud, bboxes)
