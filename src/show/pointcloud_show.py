from scannet.labels import load_scan
from common.visualize import *
from scannet.utils import scene_pointcloud

scene = 'scene0000_00'

actual_pointcloud = scene_pointcloud(scene)
expected_pointcloud = load_scan(scene)

RED = np.array([255, 0, 0])
expected_pointcloud = colored(expected_pointcloud, RED)

point_cloud = np.concatenate([actual_pointcloud, expected_pointcloud])

visualize(point_cloud, [])