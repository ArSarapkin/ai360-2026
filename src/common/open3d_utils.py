from common.pointcloud import *
import open3d as o3d
import numpy

def pointcloud_open3d(point_cloud: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3].get())
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6].get() / 255.0)
    return pcd

def bbox_open3d(bbox: BBox3D) -> o3d.geometry.OrientedBoundingBox:
    result = o3d.geometry.OrientedBoundingBox(
        center=bbox.position.get().astype(numpy.float64),
        R=bbox.rotation.get().astype(numpy.float64),
        extent=bbox.size.get().astype(numpy.float64),
    )
    result.color = (1, 0, 0)
    return result
