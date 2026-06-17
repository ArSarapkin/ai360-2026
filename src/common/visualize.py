from common.open3d_utils import *

def visualize(point_cloud: list[WorldPoint], bboxes: list[BBox3D]):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    objects = [axis, pointcloud_open3d(point_cloud)]
    objects += [bbox_open3d(bbox) for bbox in bboxes]
    o3d.visualization.draw_geometries(objects)
