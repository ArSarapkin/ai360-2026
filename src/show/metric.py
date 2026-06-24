import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

from ai import ai
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

images = range(0, 100   , 10)

bboxes = []

labels = LabelsLoader()
axis_R, axis_t = labels.load_axis_align(scene_name)
point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door',
                   'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                   'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
                   'garbagebin']

def detect_all(img, scene, axis_R, axis_t):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    def detect_one(target):
        bboxes = ai.detect_bbox(target, img_base64, scene.rgb_camera.intrinsics)
        return [scene.bbox_camera_to_world(bbox, axis_R, axis_t) for bbox in bboxes]

    bboxes = []
    with ThreadPoolExecutor(max_workers=len(classes_scannet)) as executor:
        futures = [executor.submit(detect_one, t) for t in classes_scannet]
        for future in as_completed(futures):
            try:
                bboxes.extend(future.result())
            except Exception as e:
                print(f"Detection error: {e}")

    return bboxes


for img_id in images:
    img_id = str(img_id).zfill(5)
    img = Image.open(f'{data_path}/posed_images/{scene_name}/{img_id}.jpg')
    scene = scannet_scene.build_scene(img_id)
    img_bboxes = detect_all(img, scene, axis_R, axis_t)
    bboxes.extend(img_bboxes)

visualize(point_cloud, bboxes)
