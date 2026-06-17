from scannet.scannet_config import *
from common.pointcloud_normalization import *
from common.visualize import *

scannet_scene = ScannetScene('../../data/scannet/posed_images/scene0000_00')

def frame_pointcloud(frame_id):
    scene = scannet_scene.build_scene(frame_id)
    frame = scannet_scene.load_frame(frame_id)
    return scene.process_frame(frame)

frames = ['00000', '00006', '00007', '00012', '00018', '00030', '00034', '00044', '00047']
point_cloud = []
for frame in frames:
    point_cloud += frame_pointcloud(frame)
point_cloud, R = normalize(point_cloud)
point_cloud, t = normalize_transpose(point_cloud)
visualize(point_cloud, [])
