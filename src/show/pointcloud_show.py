from scannet.labels import load_scan
from common.visualize import *
from scannet.utils import scene_pointcloud

data_path = '../../data/scannet'
scene = 'scene0448_01'

actual_pointcloud = scene_pointcloud(f'{data_path}/posed_images/{scene}')
expected_pointcloud = load_scan(f'{data_path}/points/{scene}.bin')

RED = np.array([255, 0, 0])
expected_pointcloud = colored(expected_pointcloud, RED)

point_cloud = np.concatenate([actual_pointcloud, expected_pointcloud])

visualize(point_cloud, [])