from threading import Thread, Condition
from typing import List, Tuple, Optional
import atexit

import depthai as dai
import numpy as np
import cv2
import open3d as o3d


# TODO: Implement all from link
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_post_processing/#depth-post-processing

# TODO: Implement all from link
# https://docs.luxonis.com/projects/api/en/latest/tutorials/image_quality/#improving-image-quality

# TODO: Implement all from link
# https://docs.luxonis.com/projects/api/en/latest/components/nodes/stereo_depth/#currently-configurable-blocks


# KNOWN BUGS:
# - Enabling the speckle filter crashes the camera
class OAK_Camera:
    """
    Class for interfacing with the OAK-D camera.
    Params:
        rgb_size: Size of the RGB image. Options are 1080p, 4K
        enable_rgb: Whether to enable the RGB camera
        mono_size: Size of the monochrome image. Options are 720p, 480p, 400p
        enable_mono: Whether to enable the monochrome camera
        primary_mono_left: Whether the primary monochrome image is the left image or the right image
        use_cv2_Q_matrix: Whether to use the cv2.Q matrix for disparity to depth conversion
        compute_im3d_on_demand: Whether to compute the IM3D on update
        compute_point_cloud_on_demand: Whether to compute the point cloud on update
        display_size: Size of the display window
        display_rgb: Whether to display the RGB image
        display_mono: Whether to display the monochrome image
        display_depth: Whether to display the depth image
        display_disparity: Whether to display the disparity image
        display_rectified: Whether to display the rectified image
        display_point_cloud: Whether to display the point cloud
        extended_disparity: Whether to use extended disparity
        subpixel: Whether to use subpixel
        lr_check: Whether to use left-right check
        median_filter: Whether to use median filter. If so, what size
        stereo_confidence_threshold: Confidence threshold for stereo matching
        stereo_speckle_filter_enable: Whether to use speckle filter
        stereo_speckle_filter_range: Speckle filter range
        stereo_temporal_filter_enable: Whether to use temporal filter
        stereo_spatial_filter_enable: Whether to use spatial filter
        stereo_spatial_filter_radius: Spatial filter radius
        stereo_spatial_filter_num_iterations: Spatial filter number of iterations
        stereo_threshold_filter_min_range: Threshold filter minimum range
        stereo_threshold_filter_max_range: Threshold filter maximum range
        stereo_decimation_filter_factor: Decimation filter factor. Options are 1, 2
        enable_imu: Whether to enable the IMU
        imu_batch_report_threshold: IMU batch report threshold
        imu_max_batch_reports: IMU maximum report batches
        imu_accelerometer_refresh_rate: IMU accelerometer refresh rate
        imu_gyroscope_refresh_rate: IMU gyroscope refresh rate
    """

    def __init__(
        self,
        rgb_size: str = "1080p",
        enable_rgb: bool = True,
        mono_size: str = "400p",
        enable_mono: bool = True,
        rgb_fps: int = 30,
        mono_fps: int = 30,
        primary_mono_left: bool = True,
        use_cv2_Q_matrix: bool = True,
        compute_im3d_on_demand: bool = True,
        compute_point_cloud_on_demand: bool = True,
        display_size: Tuple[int, int] = (640, 400),
        display_rgb: bool = False,
        display_mono: bool = False,
        display_depth: bool = False,
        display_disparity: bool = True,
        display_rectified: bool = False,
        display_point_cloud: bool = False,
        extended_disparity: bool = True,
        subpixel: bool = False,
        lr_check: bool = True,
        median_filter: Optional[int] = 7,
        stereo_confidence_threshold: int = 200,
        stereo_speckle_filter_enable: bool = False,
        stereo_speckle_filter_range: int = 60,
        stereo_temporal_filter_enable: bool = True,
        stereo_spatial_filter_enable: bool = True,
        stereo_spatial_filter_radius: int = 2,
        stereo_spatial_filter_num_iterations: int = 1,
        stereo_threshold_filter_min_range: int = 200,
        stereo_threshold_filter_max_range: int = 20000,
        stereo_decimation_filter_factor: int = 1,
        enable_imu: bool = True,
        imu_batch_report_threshold: int = 20,
        imu_max_batch_reports: int = 20,
        imu_accelerometer_refresh_rate: int = 400,
        imu_gyroscope_refresh_rate: int = 400,
    ):
        self._enable_rgb = enable_rgb
        self._enable_mono = enable_mono
        self._enable_imu = enable_imu

        self._rgb_fps = rgb_fps
        self._mono_fps = mono_fps

        self._primary_mono_left = primary_mono_left
        self._use_cv2_Q_matrix = use_cv2_Q_matrix

        self._display_size = display_size
        self._display_rgb = display_rgb
        self._display_mono = display_mono
        self._display_depth = display_depth
        self._display_disparity = display_disparity
        self._display_rectified = display_rectified
        self._display_point_cloud = display_point_cloud

        self._extended_disparity = extended_disparity
        self._subpixel = subpixel
        self._lr_check = lr_check

        self._stereo_confidence_threshold = stereo_confidence_threshold
        self._stereo_speckle_filter_enable = stereo_speckle_filter_enable
        self._stereo_speckle_filter_range = stereo_speckle_filter_range
        self._stereo_temporal_filter_enable = stereo_temporal_filter_enable
        self._stereo_spatial_filter_enable = stereo_spatial_filter_enable
        self._stereo_spatial_filter_radius = stereo_spatial_filter_radius
        self._stereo_spatial_filter_num_iterations = (
            stereo_spatial_filter_num_iterations
        )
        self._stereo_threshold_filter_min_range = stereo_threshold_filter_min_range
        self._stereo_threshold_filter_max_range = stereo_threshold_filter_max_range
        self._stereo_decimation_filter_factor = stereo_decimation_filter_factor

        if rgb_size not in ["4k", "1080p"]:
            raise ValueError('rgb_size must be one of "1080p" or "4k"')
        else:
            if rgb_size == "4k":
                self._rgb_size = (
                    3840,
                    2160,
                    dai.ColorCameraProperties.SensorResolution.THE_4_K,
                )
            elif rgb_size == "1080p":
                self._rgb_size = (
                    1920,
                    1080,
                    dai.ColorCameraProperties.SensorResolution.THE_1080_P,
                )

        if mono_size not in ["720p", "480p", "400p"]:
            raise ValueError('mono_size must be one of "720p", "480p", or "400p"')
        else:
            if mono_size == "720p":
                self._mono_size = (
                    1280,
                    720,
                    dai.MonoCameraProperties.SensorResolution.THE_720_P,
                )
            elif mono_size == "480p":
                self._mono_size = (
                    640,
                    480,
                    dai.MonoCameraProperties.SensorResolution.THE_480_P,
                )
            elif mono_size == "400p":
                self._mono_size = (
                    640,
                    400,
                    dai.MonoCameraProperties.SensorResolution.THE_400_P,
                )

        if self._stereo_decimation_filter_factor == 2:
            # need to divide the mono height by 2
            self._mono_size = (
                self._mono_size[0],
                self._mono_size[1] // 2,
                self._mono_size[2],
            )

        if median_filter not in [0, 3, 5, 7] and median_filter is not None:
            raise ValueError("Unsupported median filter size, use 0, 3, 5, 7, or None")
        else:
            self._median_filter = median_filter
            if self._median_filter == 3:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_3x3
            elif self._median_filter == 5:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_5x5
            elif self._median_filter == 7:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7
            else:
                self._median_filter = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

        with dai.Device() as device:
            calibData = device.readCalibration2()

            self._K_rgb = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB, self._rgb_size[0], self._rgb_size[1]
                )
            )
            self._D_rgb = np.array(
                calibData.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
            )
            self._fx_rgb = self._K_rgb[0][0]
            self._fy_rgb = self._K_rgb[1][1]
            self._cx_rgb = self._K_rgb[0][2]
            self._cy_rgb = self._K_rgb[1][2]

            self._K_left = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.LEFT, self._mono_size[0], self._mono_size[1]
                )
            )
            self._fx_left = self._K_left[0][0]
            self._fy_left = self._K_left[1][1]
            self._cx_left = self._K_left[0][2]
            self._cy_left = self._K_left[1][2]
            self._K_right = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RIGHT, self._mono_size[0], self._mono_size[1]
                )
            )

            self._fx_right = self._K_right[0][0]
            self._fy_right = self._K_right[1][1]
            self._cx_right = self._K_right[0][2]
            self._cy_right = self._K_right[1][2]
            self._D_left = np.array(
                calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)
            )
            self._D_right = np.array(
                calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)
            )

            self._rgb_fov = calibData.getFov(dai.CameraBoardSocket.RGB)
            self._mono_fov = calibData.getFov(dai.CameraBoardSocket.LEFT)

            self._K_primary = self._K_left if self._primary_mono_left else self._K_right
            self._fx_primary = (
                self._fx_left if self._primary_mono_left else self._fx_right
            )
            self._fy_primary = (
                self._fy_left if self._primary_mono_left else self._fy_right
            )
            self._cx_primary = (
                self._cx_left if self._primary_mono_left else self._cx_right
            )
            self._cy_primary = (
                self._cy_left if self._primary_mono_left else self._cy_right
            )

            self._R1 = np.array(calibData.getStereoLeftRectificationRotation())
            self._R2 = np.array(calibData.getStereoRightRectificationRotation())
            self._R_primary = self._R1 if self._primary_mono_left else self._R2

            self._T1 = np.array(
                calibData.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                )
            ) / 100  # convert to meters
            self._T2 = np.array(
                calibData.getCameraTranslationVector(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                )
            ) / 100  # convert to meters
            self._T_primary = self._T1 if self._primary_mono_left else self._T2

            self._H_left = np.matmul(
                np.matmul(self._K_right, self._R1), np.linalg.inv(self._K_left)
            )
            self._H_right = np.matmul(
                np.matmul(self._K_right, self._R1), np.linalg.inv(self._K_right)
            )

            self._l2r_extrinsic = np.array(
                calibData.getCameraExtrinsics(
                    srcCamera=dai.CameraBoardSocket.LEFT,
                    dstCamera=dai.CameraBoardSocket.RIGHT,
                )
            )
            self._r2l_extrinsic = np.array(
                calibData.getCameraExtrinsics(
                    srcCamera=dai.CameraBoardSocket.RIGHT,
                    dstCamera=dai.CameraBoardSocket.LEFT,
                )
            )
            self._primary_extrinsic = (
                self._l2r_extrinsic if self._primary_mono_left else self._r2l_extrinsic
            )

            self._baseline = calibData.getBaselineDistance() / 100  # in meters

        def _create_Q_matrix(fx, fy, cx, cy, baseline):
            return np.array(
                [
                    1,
                    0,
                    0,
                    -cx,
                    0,
                    1,
                    0,
                    -cy,
                    0,
                    0,
                    0,
                    (fx + fy) // 2,
                    0,
                    0,
                    -1 / baseline,
                    (cx - cy) / baseline,
                ]
            ).reshape(4, 4)

        self._Q_left = _create_Q_matrix(
            self._fx_left, self._fy_left, self._cx_left, self._cy_left, self._baseline
        )
        self._Q_right = _create_Q_matrix(
            self._fx_right,
            self._fy_right,
            self._cx_right,
            self._cy_right,
            self._baseline,
        )
        self._Q_primary = self._Q_left if self._primary_mono_left else self._Q_right

        (
            R1,
            R2,
            P1,
            P2,
            Q,
            self._valid_region_left,
            self._valid_region_right,
        ) = cv2.stereoRectify(
            self._K_left,
            self._D_left,
            self._K_right,
            self._D_right,
            (self._mono_size[0], self._mono_size[1]),
            self._R_primary,
            self._T_primary,
        )
        self._map_left_1, self._map_left_2 = cv2.initUndistortRectifyMap(
            self._K_left,
            self._D_left,
            R1,
            P1,
            (self._mono_size[0], self._mono_size[1]),
            cv2.CV_16SC2,
        )
        self._map_right_1, self._map_right_2 = cv2.initUndistortRectifyMap(
            self._K_right,
            self._D_right,
            R2,
            P2,
            (self._mono_size[0], self._mono_size[1]),
            cv2.CV_16SC2,
        )

        self._primary_valid_region = (
            self._valid_region_left
            if self._primary_mono_left
            else self._valid_region_right
        )
        self._Q_primary = Q if self._use_cv2_Q_matrix else self._Q_primary

        # run cv2.getOptimalNewCameraMatrix for RGB cam
        self._P_rgb, self._valid_region_rgb = cv2.getOptimalNewCameraMatrix(
            self._K_rgb,
            self._D_rgb,
            (self._rgb_size[1], self._rgb_size[0]),
            1,
            (self._rgb_size[1], self._rgb_size[0]),
        )
        self._map_rgb_1, self._map_rgb_2 = cv2.initUndistortRectifyMap(
            self._K_rgb,
            self._D_rgb,
            None,
            self._P_rgb,
            (self._rgb_size[0], self._rgb_size[1]),
            cv2.CV_16SC2,
        )

        self._o3d_pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self._mono_size[0],
            self._mono_size[1],
            self._K_primary[0][0],
            self._K_primary[1][1],
            self._K_primary[0][2],
            self._K_primary[1][2],
        )

        # pipeline
        self._pipeline: dai.Pipeline = dai.Pipeline()
        # storage for the nodes
        self._nodes: List[str] = []
        # stop condition
        self._stopped: bool = False
        # thread for the camera
        self._cam_thread = Thread(target=self._target)

        self._rgb_frame: Optional[np.ndarray] = None
        self._rectified_rgb_frame: Optional[np.ndarray] = None
        self._disparity: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None
        self._left_rect_frame: Optional[np.ndarray] = None
        self._right_rect_frame: Optional[np.ndarray] = None
        self._primary_rect_frame: Optional[np.ndarray] = None

        self._im3d: Optional[np.ndarray] = None
        self._compute_im3d_on_demand = compute_im3d_on_demand
        self._im3d_current = False

        self._point_cloud: Optional[o3d.geometry.PointCloud] = None
        self._compute_point_cloud_on_demand = compute_point_cloud_on_demand
        self._point_cloud_current = False
        self._point_cloud_vis = o3d.visualization.Visualizer()
        self._started_point_cloud_vis = False

        # imu information
        self._imu_packet: Optional[np.ndarray] = None
        self._imu_batch_report_threshold: int = imu_batch_report_threshold
        self._imu_max_batch_reports: int = imu_max_batch_reports
        self._imu_accelerometer_refresh_rate: float = imu_accelerometer_refresh_rate
        self._imu_gyroscope_refresh_rate: float = imu_gyroscope_refresh_rate
        self._imu_pose: List[float] = [0, 0, 0]
        self._imu_rotation: List[float] = [0, 0, 0]

        # packet for compute_3d
        self._3d_packet: Tuple[
            Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]
        ] = (None, None, None)

        # display information
        self._display_thread = Thread(target=self._display)
        self._display_stopped = False

        # Condition for data
        self._data_condition = Condition()

        # set atexit methods
        atexit.register(self.stop)

    @property
    def rgb(self) -> Optional[np.ndarray]:
        """
        Get the rgb color frame
        """
        return self._rgb_frame

    @property
    def rectified_rgb(self) -> Optional[np.ndarray]:
        """
        Get the rectified rgb color frame
        """
        return self._rectified_rgb_frame

    @property
    def disparity(self) -> Optional[np.ndarray]:
        """
        Gets the disparity frame
        """
        return self._disparity

    @property
    def depth(self) -> Optional[np.ndarray]:
        """
        Gets the depth frame
        """
        return self._depth

    @property
    def left(self) -> Optional[np.ndarray]:
        """
        Gets the left frame
        """
        return self._left_frame

    @property
    def right(self) -> Optional[np.ndarray]:
        """
        Gets the right frame
        """
        return self._right_frame

    @property
    def rectified_left_frame(self) -> Optional[np.ndarray]:
        """
        Gets the rectified left frame
        """
        return self._left_rect_frame

    @property
    def rectified_right_frame(self) -> Optional[np.ndarray]:
        """
        Gets the rectified right frame
        """
        return self._right_rect_frame

    @property
    def im3d(self) -> Optional[np.ndarray]:
        """
        Gets the 3d image
        """
        return self._im3d

    @property
    def point_cloud(self) -> Optional[o3d.geometry.PointCloud]:
        """
        Gets the point cloud
        """
        return self._point_cloud

    @property
    def imu_pose(self) -> List[float]:
        """
        Gets the imu pose (in meters)
        """
        return self._imu_pose

    @property
    def imu_rotation(self) -> List[float]:
        """
        Gets the imu rotation (in radians)
        """
        return self._imu_rotation

    @property
    def started(self) -> bool:
        """
        Returns true if the camera is started
        """
        return self._cam_thread.is_alive()

    def start(self, block=True) -> None:
        """
        Starts the camera
        """
        self._cam_thread.start()
        if block:
            self.wait_for_data()

    def stop(self) -> None:
        """
        Stops the camera
        """
        self._stopped = True
        try:
            self._cam_thread.join()
        except RuntimeError:
            pass

        # stop the displays
        self._display_stopped = True
        try:
            self._display_thread.join()
        except RuntimeError:
            pass

        # close displays
        cv2.destroyAllWindows()

    def wait_for_data(self) -> None:
        """
        Blocks until a full set of data has arrived from the camera
        """
        with self._data_condition:
            self._data_condition.wait()

    def _display(self) -> None:
        with self._data_condition:
            self._data_condition.wait()
        while not self._display_stopped:
            if self._rgb_frame is not None and self._display_rgb:
                cv2.imshow("rgb", cv2.resize(self._rgb_frame, self._display_size))
            if self._disparity is not None and self._display_disparity:
                frame = (
                    self._disparity * (255 / self._stereo_confidence_threshold)
                ).astype(np.uint8)
                frame = cv2.resize(frame, self._display_size)
                cv2.imshow("disparity-gray", frame)
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
                cv2.imshow("disparity-color", frame)
            if self._depth is not None and self._display_depth:
                cv2.imshow("depth", cv2.resize(self._depth, self._display_size))
            if self._left_frame is not None and self._display_mono:
                cv2.imshow("left", cv2.resize(self._left_frame, self._display_size))
            if self._right_frame is not None and self._display_mono:
                cv2.imshow("right", cv2.resize(self._right_frame, self._display_size))
            if self._left_rect_frame is not None and self._display_rectified:
                cv2.imshow(
                    "rectified left",
                    cv2.resize(self._left_rect_frame, self._display_size),
                )
            if self._right_rect_frame is not None and self._display_rectified:
                cv2.imshow(
                    "rectified right",
                    cv2.resize(self._right_rect_frame, self._display_size),
                )
            if self._point_cloud is not None and self._display_point_cloud:
                if not self._started_point_cloud_vis:
                    self._point_cloud_vis.create_window()
                    self._point_cloud_vis.add_geometry(self._point_cloud)
                    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.3, origin=[0, 0, 0]
                    )
                    self._point_cloud_vis.add_geometry(origin)
                    self._started_point_cloud_vis = True
                else:
                    self._point_cloud_vis.update_geometry(self._point_cloud)
                    self._point_cloud_vis.poll_events()
                    self._point_cloud_vis.update_renderer()
            cv2.waitKey(50)
        if self._started_point_cloud_vis:
            self._point_cloud_vis.destroy_window()

    def start_display(self) -> None:
        """
        Starts the display thread
        """
        self._display_thread.start()

    def stop_display(self) -> None:
        """
        Stops the display thread
        """
        self._display_stopped = True
        self._display_thread.join()

    def _create_cam_rgb(self) -> None:
        cam = self._pipeline.create(dai.node.ColorCamera)
        xout_video = self._pipeline.create(dai.node.XLinkOut)

        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setResolution(self._rgb_size[2])
        cam.setInterleaved(False)
        cam.setFps(self._rgb_fps)

        xout_video.setStreamName("rgb")
        cam.video.link(xout_video.input)

        self._nodes.extend(["rgb"])

    def _create_stereo(self) -> None:
        left = self._pipeline.create(dai.node.MonoCamera)
        right = self._pipeline.create(dai.node.MonoCamera)
        stereo = self._pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        xout_left = self._pipeline.create(dai.node.XLinkOut)
        xout_right = self._pipeline.create(dai.node.XLinkOut)
        xout_depth = self._pipeline.create(dai.node.XLinkOut)
        xout_disparity = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_left = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_right = self._pipeline.create(dai.node.XLinkOut)

        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [left, right]:
            cam.setFps(self._mono_fps)
            cam.setResolution(self._mono_size[2])

        stereo.initialConfig.setConfidenceThreshold(self._stereo_confidence_threshold)
        stereo.setRectifyEdgeFillColor(0)
        stereo.initialConfig.setMedianFilter(self._median_filter)
        stereo.setLeftRightCheck(self._lr_check)
        stereo.setExtendedDisparity(self._extended_disparity)
        stereo.setSubpixel(self._subpixel)

        config = stereo.initialConfig.get()
        config.postProcessing.speckleFilter.enable = self._stereo_speckle_filter_enable
        config.postProcessing.speckleFilter.speckleRange = (
            self._stereo_speckle_filter_range
        )
        config.postProcessing.temporalFilter.enable = (
            self._stereo_temporal_filter_enable
        )
        config.postProcessing.spatialFilter.enable = self._stereo_spatial_filter_enable
        config.postProcessing.spatialFilter.holeFillingRadius = (
            self._stereo_spatial_filter_radius
        )
        config.postProcessing.spatialFilter.numIterations = (
            self._stereo_spatial_filter_num_iterations
        )
        config.postProcessing.thresholdFilter.minRange = (
            self._stereo_threshold_filter_min_range
        )
        config.postProcessing.thresholdFilter.maxRange = (
            self._stereo_threshold_filter_max_range
        )
        config.postProcessing.decimationFilter.decimationFactor = (
            self._stereo_decimation_filter_factor
        )
        stereo.initialConfig.set(config)

        xout_left.setStreamName("left")
        xout_right.setStreamName("right")
        xout_depth.setStreamName("depth")
        xout_disparity.setStreamName("disparity")
        xout_rect_left.setStreamName("rectified_left")
        xout_rect_right.setStreamName("rectified_right")

        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.syncedLeft.link(xout_left.input)
        stereo.syncedRight.link(xout_right.input)
        stereo.depth.link(xout_depth.input)
        stereo.disparity.link(xout_disparity.input)
        stereo.rectifiedLeft.link(xout_rect_left.input)
        stereo.rectifiedRight.link(xout_rect_right.input)

        self._nodes.extend(
            ["left", "right", "depth", "disparity", "rectified_left", "rectified_right"]
        )

    def _create_imu(self) -> None:
        imu = self._pipeline.create(dai.node.IMU)
        xout_imu = self._pipeline.create(dai.node.XLinkOut)

        imu.enableIMUSensor(dai.IMUSensor.GRAVITY, self._imu_accelerometer_refresh_rate)
        imu.enableIMUSensor(
            dai.IMUSensor.GYROSCOPE_CALIBRATED, self._imu_gyroscope_refresh_rate
        )
        imu.setBatchReportThreshold(self._imu_batch_report_threshold)
        imu.setMaxBatchReports(self._imu_max_batch_reports)

        xout_imu.setStreamName("imu")

        imu.out.link(xout_imu.input)

        self._nodes.extend(["imu"])

    def _update_point_cloud(self) -> None:
        rgb_frame = cv2.resize(
            self._rgb_frame, (self._depth.shape[1], self._depth.shape[0])
        )
        rgb_o3d = o3d.geometry.Image(rgb_frame)
        depth_o3d = o3d.geometry.Image(self._depth)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            self._o3d_pinhole_camera_intrinsic,
        )
        # pcd = pcd.voxel_down_sample(voxel_size=0.01)
        pcd = pcd.remove_statistical_outlier(30, 0.1)[0]

        if self._point_cloud is None:
            self._point_cloud = pcd
        else:
            self._point_cloud.points = pcd.points
            self._point_cloud.colors = pcd.colors

    def _update_im3d(self) -> None:
        self._im3d = cv2.reprojectImageTo3D(self._disparity, self._Q_primary)

    def _target(self) -> None:
        if self._enable_rgb:
            self._create_cam_rgb()
        if self._enable_mono:
            self._create_stereo()
        if self._enable_imu:
            self._create_imu()
        with dai.Device(self._pipeline) as device:
            queues = {}
            for stream in self._nodes:
                queues[stream] = device.getOutputQueue(
                    name=stream, maxSize=1, blocking=False
                )

            base_accel_timestamp = None
            base_gyro_timestamp = None
            while not self._stopped:
                for name, queue in queues.items():
                    if queue is not None:
                        data = queue.get()
                        if name == "rgb":
                            self._rgb_frame = data.getCvFrame()
                            self._rectified_rgb_frame = cv2.remap(
                                self._rgb_frame,
                                self._map_rgb_1,
                                self._map_rgb_2,
                                cv2.INTER_LINEAR,
                            )
                        elif name == "left":
                            self._left_frame = data.getCvFrame()
                        elif name == "right":
                            self._right_frame = data.getCvFrame()
                        elif name == "depth":
                            self._depth = data.getCvFrame()
                        elif name == "disparity":
                            self._disparity = data.getCvFrame()
                        elif name == "rectified_left":
                            self._left_rect_frame = data.getCvFrame()
                        elif name == "rectified_right":
                            self._right_rect_frame = data.getCvFrame()
                        elif name == "imu":
                            packets = data.packets
                            for imuPacket in packets:
                                acceleroValues = imuPacket.acceleroMeter
                                gyroValues = imuPacket.gyroscope

                                acclero_ts_device = acceleroValues.getTimestampDevice()
                                gyro_ts_device = gyroValues.getTimestampDevice()

                                if base_accel_timestamp is None:
                                    base_accel_timestamp = acclero_ts_device
                                if base_gyro_timestamp is None:
                                    base_gyro_timestamp = gyro_ts_device

                                accelero_ts = (
                                    acclero_ts_device - base_accel_timestamp
                                ).total_seconds()
                                gyro_ts = (
                                    gyro_ts_device - base_gyro_timestamp
                                ).total_seconds()

                                ax, ay, az = (
                                    acceleroValues.x,
                                    acceleroValues.y,
                                    acceleroValues.z,
                                )
                                gx, gy, gz = gyroValues.x, gyroValues.y, gyroValues.z

                                # double integrate each ax, ay, az
                                self._imu_pose[0] += ax * (accelero_ts ** 2)
                                self._imu_pose[1] += ay * (accelero_ts ** 2)
                                self._imu_pose[2] += az * (accelero_ts ** 2)

                                # integrate each gx, gy, gz
                                self._imu_rotation[0] += gx * gyro_ts
                                self._imu_rotation[1] += gy * gyro_ts
                                self._imu_rotation[2] += gz * gyro_ts

                                base_accel_timestamp = acclero_ts_device
                                base_gyro_timestamp = gyro_ts_device

                # handle primary mono camera
                self._primary_rect_frame = (
                    self._left_rect_frame
                    if self._primary_mono_left
                    else self._right_rect_frame
                )
                # handle 3d images and odometry packets
                if not self._compute_im3d_on_demand:
                    self._update_im3d()
                self._3d_packet = (
                    self._im3d,
                    self._disparity,
                    self._primary_rect_frame,
                )
                # handle the point cloud
                if not self._compute_point_cloud_on_demand:
                    self._update_point_cloud()

                with self._data_condition:
                    self._data_condition.notify_all()

    def _crop_to_valid_primary_region(self, img: np.ndarray) -> np.ndarray:
        return img[
            self._primary_valid_region[1] : self._primary_valid_region[3],
            self._primary_valid_region[0] : self._primary_valid_region[2],
        ]

    def compute_point_cloud(self, block=True) -> Optional[o3d.geometry.PointCloud]:
        """
        Compute point cloud from depth map.
        Returns:
            Optional[o3d.geometry.PointCloud]: point cloud
        """
        if block:
            with self._data_condition:
                self._data_condition.wait()
        if self._rgb_frame is None and self._depth is None:
            return None
        if self._compute_point_cloud_on_demand:
            self._update_point_cloud()
        return self._point_cloud

    def compute_im3d(
        self, block=True
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute 3D points from disparity map.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: depth map, disparity map, left frame
        """
        if block:
            with self._data_condition:
                self._data_condition.wait()
        im3d, disparity, rect = self._3d_packet
        if im3d is None and disparity is None and rect is None:
            return None, None, None
        if self._compute_im3d_on_demand:
            self._update_im3d()
            im3d = self._im3d
        return (
            self._crop_to_valid_primary_region(im3d),
            self._crop_to_valid_primary_region(disparity),
            self._crop_to_valid_primary_region(rect),
        )
