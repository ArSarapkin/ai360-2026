from ai import ai
from common.visualize import visualize
from scannet.scannet_config import ScannetScene
from scannet.utils import scene_pointcloud
from PIL import Image

scene_name = 'scene0000_00'

scannet_scene = ScannetScene(f'../../data/scannet/posed_images/{scene_name}')
point_cloud = scene_pointcloud(f'../../data/scannet/posed_images/{scene_name}')

targets = [
    (5, "cabinet"),
    (17, "table"),
]

bboxes = []

for target in targets:
    img_id, target_name = target
    img_id = str(img_id).zfill(5)
    img = Image.open(f'../../data/scannet/posed_images/{scene_name}/{img_id}.jpg')
    scene = scannet_scene.build_scene(img_id)
    bbox = ai.detect_bbox(target_name, img, scene.rgb_camera.intrinsics)
    bbox = scene.bbox_camera_to_world(bbox)
    bboxes.append(bbox)

visualize(point_cloud, bboxes)
