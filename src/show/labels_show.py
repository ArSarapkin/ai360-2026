import pickle
import cupy as cp

from common.pointcloud import BBox3D, apply_axis_align
from common.visualize import visualize
from scannet.labels import load_scan, LabelsLoader
from scannet.utils import scene_pointcloud

scene = 'scene0000_00'

loader = LabelsLoader()
bboxes = loader.load_bboxes(scene)
point_cloud = scene_pointcloud(scene)
axis_R, axis_t = loader.load_axis_align(scene)
point_cloud = apply_axis_align(point_cloud, axis_R, axis_t)

visualize(point_cloud, bboxes)