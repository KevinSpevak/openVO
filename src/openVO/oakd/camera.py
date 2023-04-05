from threading import Thread, Lock
from typing import List, Tuple, Optional

import depthai as dai
import numpy as np
import cv2
import open3d as o3d


class OAK_Camera:
    def __init__(
        self,
        rgb_size: Tuple[int, int] = (1920, 1080),
        mono_size: Tuple[int, int] = (1280, 720),
        display_size: Tuple[int, int] = (640, 480),
        extended_disparity: bool = True,
        subpixel: bool = False,
        lr_check: bool = True,
        median_filter: Optional[int] = None,
    ):
        self._rgb_width = rgb_size[0]
        self._rgb_height = rgb_size[1]
        self._left_width = mono_size[0]
        self._right_width = self._left_width
        self._left_height = mono_size[1]
        self._right_height = self._left_height

        self._rgb_size = rgb_size
        self._mono_size = mono_size
        self._display_size = display_size

        self._extended_disparity = extended_disparity
        self._subpixel = subpixel
        self._lr_check = lr_check

        if self._extended_disparity:
            self._subpixel = False

        if median_filter not in [3, 5, 7] and median_filter is not None:
            raise ValueError("Unsupported median filter size, use 3, 5, 7, or None")
        elif self._extended_disparity or self._subpixel or self._lr_check:
            self._median_filter = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF
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
            calibData = device.readCalibration()

            self._K_rgb = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RGB, self._right_width, self._rgb_height
                )
            )
            self._K_left = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.LEFT, self._left_width, self._left_height
                )
            )
            self._K_right = np.array(
                calibData.getCameraIntrinsics(
                    dai.CameraBoardSocket.RIGHT, self._right_width, self._right_height
                )
            )
            self._D_left = np.array(
                calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)
            )
            self._D_right = np.array(
                calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)
            )
            self._rgb_fov = calibData.getFov(dai.CameraBoardSocket.RGB)
            self._mono_fov = calibData.getFov(dai.CameraBoardSocket.LEFT)

            self._R1 = np.array(calibData.getStereoLeftRectificationRotation())
            self._R2 = np.array(calibData.getStereoRightRectificationRotation())

            self._T1 = np.array(calibData.getCameraTranslationVector(srcCamera=dai.CameraBoardSocket.LEFT, dstCamera=dai.CameraBoardSocket.RIGHT))
            self._T2 = np.array(calibData.getCameraTranslationVector(srcCamera=dai.CameraBoardSocket.RIGHT, dstCamera=dai.CameraBoardSocket.LEFT))

            self._H_left = np.matmul(
                np.matmul(self._K_right, self._R1), np.linalg.inv(self._K_left)
            )
            self._H_right = np.matmul(
                np.matmul(self._K_right, self._R1), np.linalg.inv(self._K_right)
            )

            self._baseline = calibData.getBaselineDistance()  # in centimeters

        # pipeline
        self._pipeline: dai.Pipeline = dai.Pipeline()
        # storage for the nodes
        self._nodes: List[str] = []
        # stop condition
        self._stopped: bool = False
        # thread for the camera
        self._cam_thread = Thread(target=self._target)

        self._rgb_frame: Optional[np.ndarray] = None
        self._disparity: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._left_frame: Optional[np.ndarray] = None
        self._right_frame: Optional[np.ndarray] = None
        self._left_rect_frame: Optional[np.ndarray] = None
        self._right_rect_frame: Optional[np.ndarray] = None

        # packet for compute_3d
        self._3d_packet: Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]] = (None, None, None)

        # display information
        self._display_thread = Thread(target=self._display)
        self._display_stopped = False

    @property
    def rgb(self) -> Optional[np.ndarray]:
        """
        Get the rgb color frame
        """
        return self._rgb_frame

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
    def started(self) -> bool:
        """
        Returns true if the camera is started
        """
        return self._cam_thread.is_alive()

    def start(self) -> None:
        """
        Starts the camera
        """
        self._cam_thread.start()

    def stop(self) -> None:
        """
        Stops the camera
        """
        self._stopped = True
        self._cam_thread.join()

        # stop the displays
        self._display_stopped = True
        try:
            self._display_thread.join()
        except RuntimeError:
            pass
        # close displays
        cv2.destroyAllWindows()

    def _display(self) -> None:
        while not self._display_stopped:
            if self._rgb_frame is not None:
                cv2.imshow("rgb", cv2.resize(self._rgb_frame, self._display_size))
            if self._disparity is not None:
                # disparity = cv2.normalize(
                #     self._disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                # )
                cv2.imshow("disparity", cv2.resize(self._disparity, self._display_size))
            if self._depth is not None:
                cv2.imshow("depth", cv2.resize(self._depth, self._display_size))
            if self._left_frame is not None:
                cv2.imshow("left", cv2.resize(self._left_frame, self._display_size))
            if self._right_frame is not None:
                cv2.imshow("right", cv2.resize(self._right_frame, self._display_size))
            if self._left_rect_frame is not None:
                cv2.imshow(
                    "rectified left",
                    cv2.resize(self._left_rect_frame, self._display_size),
                )
            if self._right_rect_frame is not None:
                cv2.imshow(
                    "rectified right",
                    cv2.resize(self._right_rect_frame, self._display_size),
                )
            cv2.waitKey(50)

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
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)

        xout_video.setStreamName("rgb")
        cam.video.link(xout_video.input)

        self._nodes.append("rgb")

    def _create_stereo(self) -> None:
        left = self._pipeline.create(dai.node.MonoCamera)
        right = self._pipeline.create(dai.node.MonoCamera)
        stereo = self._pipeline.create(dai.node.StereoDepth)
        xout_left = self._pipeline.create(dai.node.XLinkOut)
        xout_right = self._pipeline.create(dai.node.XLinkOut)
        xout_depth = self._pipeline.create(dai.node.XLinkOut)
        xout_disparity = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_left = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_right = self._pipeline.create(dai.node.XLinkOut)

        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        for cam in [left, right]:
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo.initialConfig.setConfidenceThreshold(200)
        stereo.setRectifyEdgeFillColor(0) 
        stereo.initialConfig.setMedianFilter(self._median_filter)
        stereo.setLeftRightCheck(self._lr_check)
        stereo.setExtendedDisparity(self._extended_disparity)
        stereo.setSubpixel(self._subpixel)

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

        self._nodes.extend(["left", "right", "depth", "disparity", "rectified_left", "rectified_right"])

    def _target(self) -> None:
        self._create_cam_rgb()
        self._create_stereo()
        with dai.Device(self._pipeline) as device:
            queues = {}
            for stream in self._nodes:
                queues[stream] = device.getOutputQueue(
                    name=stream, maxSize=1, blocking=False
                )

            # TODO: handle these concurrently
            # TODO: make sure we handle synced frames correctly
            while not self._stopped:
                for name, queue in queues.items():
                    if queue is not None:
                        data = queue.get()
                        if name == "rgb":
                            self._rgb_frame = data.getCvFrame()
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
                self._3d_packet = (self._depth, self._disparity, self._left_rect_frame)

    def compute_3d(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Compute 3D point cloud from disparity map.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 3D point cloud, disparity map, left frame
        """
        return self._3d_packet
