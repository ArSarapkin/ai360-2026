import pickle

import cupy as cp
import numpy

from common.pointcloud import BBox3D
from common.visualize import visualize

data_path = '../../data/scannet'

def load_scan(pcd_path) -> cp.ndarray:
    pcd_path = f'{data_path}/points/{pcd_path}.bin'
    return cp.asarray(numpy.fromfile(pcd_path, dtype=numpy.float32).reshape(-1, 6))

class LabelsLoader:
    instances: dict
    axis_align_matrix: dict

    def __init__(self):
        with open('../../data/scannet/scannet_infos_train.pkl', 'rb') as f:
            data = pickle.load(f)

        self.instances = {
            s['lidar_points']['lidar_path'].removesuffix('.bin'): s['instances']
            for s in data['data_list']
        }

        self.axis_align_matrix = {
            s['lidar_points']['lidar_path'].removesuffix('.bin'): cp.array(s['axis_align_matrix'])
            for s in data['data_list']
        }

    def load_bboxes(self, scene):
        bboxes = []
        for json in self.instances[scene]:
            position = json['bbox_3d'][:3]
            size = json['bbox_3d'][3:6]
            rotation = cp.eye(3)
            bbox = BBox3D(cp.array(position), cp.array(size), rotation)
            bboxes.append(bbox)
        return bboxes

    def load_axis_align(self, scene):
        axis_align_matrix = self.axis_align_matrix[scene]
        rotation = axis_align_matrix[:3, :3]
        translation = axis_align_matrix[:3, 3]
        return rotation, translation