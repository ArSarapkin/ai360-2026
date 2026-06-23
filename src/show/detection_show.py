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

targets = [
    (5, "cabinet"),
    (17, "table"),
]

bboxes = []

labels = LabelsLoader()
axis_R, axis_t = labels.load_axis_align(scene_name)
point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

for target in targets:
    img_id, target_name = target
    img_id = str(img_id).zfill(5)
    img = Image.open(f'{data_path}/posed_images/{scene_name}/{img_id}.jpg')
    scene = scannet_scene.build_scene(img_id)
    bbox = ai.detect_bbox(target_name, img, scene.rgb_camera.intrinsics)
    bbox = scene.bbox_camera_to_world(bbox, axis_R, axis_t)
    bboxes.append(bbox)

visualize(point_cloud, bboxes)
