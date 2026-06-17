from scannet.scannet_config import *
from common.pointcloud_normalization import *
from common.visualize import *

scannet_scene = ScannetScene('../../data/scannet/posed_images/scene0000_00')

def visualize_frame(frame_id):
    scene = scannet_scene.build_scene(frame_id)
    frame = scannet_scene.load_frame(frame_id)
    point_cloud = scene.process_frame(frame)
    point_cloud, R = normalize(point_cloud)
    point_cloud, t = normalize_transpose(point_cloud)
    visualize(point_cloud, [])

visualize_frame('00000')