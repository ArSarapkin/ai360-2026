from common.open3d_utils import *

def visualize(point_cloud: np.ndarray, bboxes: list[BBox3D]):
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(axis)
    vis.add_geometry(pointcloud_open3d(point_cloud))
    for bbox in bboxes:
        vis.add_geometry(bbox_open3d(bbox))
    options = vis.get_render_option()
    options.point_size = 1.0
    options.line_width = 8.0
    vis.run()
    vis.destroy_window()
