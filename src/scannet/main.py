import cupy as np

from scannet.labels import load_scan
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


chunks = []
frame_id = 0
frame_id_limit = 2000
while True:
    if frame_id > frame_id_limit:
        break
    try:
        frame = str(frame_id).zfill(5)
        chunks.append(frame_pointcloud(frame))
        frame_id += 10
    except:
        break
point_cloud = np.concatenate(chunks, axis=0)
R = np.eye(3, 3)# point_cloud, R = normalize(point_cloud)
point_cloud, t = normalize_transpose(point_cloud)

targets = [
    #(5, "cabinet"),
    #(17, "table"),
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

expected = load_scan('../../data/scannet/points/scene0000_00.bin')
expected, _ = normalize_transpose(expected)

RED = np.array([255, 0, 0])
BLUE = np.array([0, 0, 255])

visualize(np.concatenate([colored(point_cloud, BLUE), colored(expected, RED)], axis=0), bboxes)
