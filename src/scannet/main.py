import numpy as np

from scannet.scannet_config import *
from common.pointcloud_normalization import *
from common.visualize import *
import ai.ai as ai

scannet_scene = ScannetScene('../../data/scannet/posed_images/scene0000_00')


def frame_pointcloud(frame_id):
    start_time = time.time()
    print(f"Starting processing frame {frame_id}")
    scene = scannet_scene.build_scene(frame_id)
    frame = scannet_scene.load_frame(frame_id)
    finish_time = time.time()
    print(f"Finished processing frame in {finish_time - start_time} seconds")
    return scene.process_frame(frame)


point_cloud = []
frame_id = 0
frame_id_limit = 20
while True:
    if frame_id > frame_id_limit:
        break
    try:
        frame = str(frame_id).zfill(5)
        point_cloud += frame_pointcloud(frame)
        frame_id += 5
    except:
        break
point_cloud, R = normalize(point_cloud)
point_cloud, t = normalize_transpose(point_cloud)

targets = [
    (5, "cabinet"),
    (17, "table"),
]

bboxes = []

for target in targets:
    img_id, target_name = target
    img_id = str(img_id).zfill(5)
    img = Image.open(f'../../data/scannet/posed_images/scene0000_00/{img_id}.jpg')
    scene = scannet_scene.build_scene(img_id)
    bbox = ai.detect_bbox(target_name, img, scene.rgb_camera.intrinsics)
    bbox = scene.bbox_camera_to_world(R, t, bbox)
    bboxes.append(bbox)

visualize(point_cloud, bboxes)
