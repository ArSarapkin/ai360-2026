from common.pointcloud import *
import open3d as o3d

def pointcloud_open3d(point_cloud: list[WorldPoint]) -> o3d.geometry.PointCloud:
    points = np.array([p.position for p in point_cloud])
    colors = np.array([p.color for p in point_cloud]) / 255.0

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def bbox_open3d(bbox: BBox3D) -> o3d.geometry.OrientedBoundingBox:
    bbox = o3d.geometry.OrientedBoundingBox(
        center=bbox.position,
        R=bbox.rotation,
        extent=bbox.size,
    )
    bbox.color = (1, 0, 0)
    return bbox
