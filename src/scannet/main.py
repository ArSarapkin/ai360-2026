from scannet.scannet_config import *
from common.pointcloud_normalization import *
from common.visualize import *

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
while True:
    try:
        frame = str(frame_id).zfill(5)
        point_cloud += frame_pointcloud(frame)
        frame_id += 5
    except:
        break
point_cloud, R = normalize(point_cloud)
point_cloud, t = normalize_transpose(point_cloud)
visualize(point_cloud, [])
