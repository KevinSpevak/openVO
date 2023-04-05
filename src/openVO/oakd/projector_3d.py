import open3d as o3d


# Modified from: https://github.com/luxonis/depthai-experiments/blob/master/gen2-camera-demo/projector_3d.py
class PointCloudVisualizer:
    def __init__(self, intrinsic_matrix, width, height):
        self._depth_map = None
        self._rgb = None
        self._pcl = None

        self._pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsic_matrix[0][0],
            intrinsic_matrix[1][1],
            intrinsic_matrix[0][2],
            intrinsic_matrix[1][2],
        )
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window()
        self._isstarted = False

    def rgbd_to_projection(self, depth_map, rgb, is_rgb):
        self._depth_map = depth_map
        self._rgb = rgb
        rgb_o3d = o3d.geometry.Image(self._rgb)
        depth_o3d = o3d.geometry.Image(self._depth_map)
        # TODO: query frame shape to get this, and remove the param 'is_rgb'
        if is_rgb:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
            )
        else:
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d
            )
        if self._pcl is None:
            self._pcl = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self._pinhole_camera_intrinsic
            )
        else:
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image, self._pinhole_camera_intrinsic
            )
            self._pcl.points = pcd.points
            self._pcl.colors = pcd.colors
        return self._pcl

    def visualize_pcd(self):
        if not self.isstarted:
            self._vis.add_geometry(self._pcl)
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.3, origin=[0, 0, 0]
            )
            self._vis.add_geometry(origin)
            self._isstarted = True
        else:
            self._vis.update_geometry(self._pcl)
            self._vis.poll_events()
            self._vis.update_renderer()

    def close_window(self):
        self._vis.destroy_window()
