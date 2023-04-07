from threading import Thread, Lock
from typing import Dict, Tuple, List, Optional

import depthai as dai
import numpy as np


class OAK_Camera:
    def __init__(self, rgb_size=(1920, 1080), mono_size=(1280, 720), extended_disparity=True, subpixel=True, lr_check=True):
        self._rgb_width = rgb_size[0]
        self._rgb_height = rgb_size[1]
        self._left_width = mono_size[0]
        self._right_width = self._left_width
        self._left_height = mono_size[1]
        self._right_height = self._left_height

        self._extended_disparity = extended_disparity
        self._subpixel = subpixel
        self._lr_check = lr_check

        with dai.Device() as device:
            calibData = device.readCalibration()

            self._M_rgb = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RGB, self._right_width, self._rgb_height))
            self._M_left = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, self._left_width, self._left_height))
            self._M_right = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, self._right_width, self._right_height))
            self._D_left = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
            self._D_right = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
            self._rgb_fov = calibData.getFov(dai.CameraBoardSocket.RGB)
            self._mono_fov = calibData.getFov(dai.CameraBoardSocket.LEFT)

            self._R1 = np.array(calibData.getStereoLeftRectificationRotation())
            self._R2 = np.array(calibData.getStereoRightRectificationRotation())

            self._H_left = np.matmul(np.matmul(self._M_right, self._R1), np.linalg.inv(self._M_left))
            self._H_right = np.matmul(np.matmul(self._M_right, self._R1), np.linalg.inv(self._M_right))

        # pipeline
        self._pipeline: dai.Pipeline = dai.Pipeline()
        # storage for the nodes
        self._nodes: Dict[str, Tuple[dai.Node, dai.XLinkOut]] = {}
        # stop condition
        self._stopped: bool = False
        # thread for the camera
        self._cam_thread = Thread(target=self._target)

        self._rgb_frame: Optional[np.ndarray] = None
        self._rgb_frame_lock = Lock()
        self._depth_frame: Optional[np.ndarray] = None
        self._depth_frame_lock = Lock()
        self._left_frame: Optional[np.ndarray] = None
        self._left_frame_lock = Lock()
        self._right_frame: Optional[np.ndarray] = None
        self._right_frame_lock = Lock()
        self._rectified_left_frame: Optional[np.ndarray] = None
        self._rectified_left_frame_lock = Lock()
        self._rectified_right_frame: Optional[np.ndarray] = None
        self._rectified_right_frame_lock = Lock()

    @property
    def rgb_frame(self) -> Optional[np.ndarray]:
        """
        Get the rgb color frame
        """
        self._rgb_frame_lock.acquire()
        ret_val = self._rgb_frame
        self._rgb_frame = None
        self._rgb_frame_lock.release()
        return ret_val
    
    @property
    def disparity(self) -> Optional[np.ndarray]:
        """
        Gets the disparity frame
        """
        self._depth_frame_lock.acquire()
        ret_val = self._depth_frame
        self._depth_frame = None
        self._depth_frame_lock.release()
        return ret_val
    
    @property
    def left_frame(self) -> Optional[np.ndarray]:
        """
        Gets the left frame
        """
        self._left_frame_lock.acquire()
        ret_val = self._left_frame
        self._left_frame = None
        self._left_frame_lock.release()
        return ret_val

    @property
    def right_frame(self) -> Optional[np.ndarray]:
        """
        Gets the right frame
        """
        self._right_frame_lock.acquire()
        ret_val = self._right_frame
        self._right_frame = None
        self._right_frame_lock.release()
        return ret_val

    @property
    def rectified_left_frame(self) -> Optional[np.ndarray]:
        """
        Gets the rectified left frame
        """
        self._rectified_left_frame_lock.acquire()
        ret_val = self._rectified_left_frame
        self._rectified_left_frame = None
        self._rectified_left_frame_lock.release()
        return ret_val

    @property
    def rectified_right_frame(self) -> Optional[np.ndarray]:
        """
        Gets the rectified right frame
        """
        self._rectified_right_frame_lock.acquire()
        ret_val = self._rectified_right_frame
        self._rectified_right_frame = None
        self._rectified_right_frame_lock.release()
        return ret_val
    
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

    def _create_cam_rgb(self) -> None:
        cam_rgb = self._pipeline.create(dai.node.ColorCamera)
        xout_video = self._pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("color_camera")
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        if self._rgb_height == 1080:
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        elif self._rgb_height == 2160:
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
        else:
            raise NotImplementedError(f"Resolution not implemented: {self._rgb_width}, {self._rgb_height}")
        
        cam_rgb.setVideoSize(self._rgb_width, self._rgb_height)
        xout_video.input.setBlocking(False)
        xout_video.input.setQueueSize(1)
        cam_rgb.video.link(xout_video.input)

        self._nodes["color_camera"] = (cam_rgb, xout_video)

    def _create_stereo(self) -> None:
        # Define sources and outputs
        mono_left = self._pipeline.create(dai.node.MonoCamera)
        mono_right = self._pipeline.create(dai.node.MonoCamera)
        depth = self._pipeline.create(dai.node.StereoDepth)

        xout_depth = self._pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("disparity")
        xout_rect_left = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_left.setStreamName("rectified_left")
        xout_rect_right = self._pipeline.create(dai.node.XLinkOut)
        xout_rect_right.setStreamName("rectified_right")

        # Properties
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        # Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
        depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
        depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        depth.setLeftRightCheck(self._lr_check)
        depth.setExtendedDisparity(self._extended_disparity)
        depth.setSubpixel(self._subpixel)

        config = depth.initialConfig.get()
        config.postProcessing.speckleFilter.enable = False
        config.postProcessing.speckleFilter.speckleRange = 50
        config.postProcessing.temporalFilter.enable = True
        config.postProcessing.spatialFilter.enable = True
        config.postProcessing.spatialFilter.holeFillingRadius = 2
        config.postProcessing.spatialFilter.numIterations = 1
        config.postProcessing.thresholdFilter.minRange = 400
        config.postProcessing.thresholdFilter.maxRange = 15000
        config.postProcessing.decimationFilter.decimationFactor = 1
        depth.initialConfig.set(config)

        # Linking
        mono_left.out.link(depth.left)
        mono_right.out.link(depth.right)
        depth.disparity.link(xout_depth.input)
        depth.rectifiedLeft.link(xout_rect_left.input)
        depth.rectifiedRight.link(xout_rect_right.input)

        self._nodes["stereo"] = (depth, xout_depth)
        self._nodes["mono_left"] = (mono_left, None)
        self._nodes["mono_right"] = (mono_right, None)
        self._nodes["rectified_left"] = (depth, xout_rect_left)
        self._nodes["rectified_right"] = (depth, xout_rect_right)

    def _target(self) -> None:
        self._create_cam_rgb()
        self._create_stereo()
        with dai.Device(self._pipeline) as device:
            video_queue = None
            if self._nodes["color_camera"] is not None:
                video_queue = device.getOutputQueue(
                    name="color_camera", maxSize=1, blocking=False
                )

            depth_queue = None
            if self._nodes["stereo"] is not None:
                depth_queue = device.getOutputQueue(
                    name="disparity", maxSize=1, blocking=False
                )

            left_queue = None
            if self._nodes["mono_left"] is not None:
                left_queue = device.getOutputQueue(
                    name="mono_left", maxSize=1, blocking=False
                )
            
            right_queue = None
            if self._nodes["mono_right"] is not None:
                right_queue = device.getOutputQueue(
                    name="mono_right", maxSize=1, blocking=False
                )
            
            left_rect_queue = None
            if self._nodes["rectified_left"] is not None:
                left_rect_queue = device.getOutputQueue(
                    name="rectified_left", maxSize=1, blocking=False
                )
            
            right_rect_queue = None
            if self._nodes["rectified_right"] is not None:
                right_rect_queue = device.getOutputQueue(
                    name="rectified_right", maxSize=1, blocking=False
                )

            while not self._stopped:
                if video_queue is not None:
                    video_frame = video_queue.get()
                    rgb_frame = video_frame.getCvFrame()

                    self._rgb_frame_lock.acquire()
                    self._rgb_frame = rgb_frame
                    self._rgb_frame_lock.release()

                if depth_queue is not None:
                    depth_frame = depth_queue.get()
                    depth_frame = depth_frame.getCvFrame()

                    self._depth_frame_lock.acquire()
                    self._depth_frame = depth_frame
                    self._depth_frame_lock.release()

                if left_queue is not None:
                    left_frame = left_queue.get()
                    left_frame = left_frame.getCvFrame()

                    self._left_frame_lock.acquire()
                    self._left_frame = left_frame
                    self._left_frame_lock.release()
                
                if right_queue is not None:
                    right_frame = right_queue.get()
                    right_frame = right_frame.getCvFrame()

                    self._right_frame_lock.acquire()
                    self._right_frame = right_frame
                    self._right_frame_lock.release()
                
                if left_rect_queue is not None:
                    left_rect_frame = left_rect_queue.get()
                    left_rect_frame = left_rect_frame.getCvFrame()

                    self._rectified_left_frame_lock.acquire()
                    self._left_rect_frame = left_rect_frame
                    self._rectified_left_frame_lock.release()
                
                if right_rect_queue is not None:
                    right_rect_frame = right_rect_queue.get()
                    right_rect_frame = right_rect_frame.getCvFrame()

                    self._rectified_right_frame_lock.acquire()
                    self._right_rect_frame = right_rect_frame
                    self._rectified_right_frame_lock.release()

    def compute_3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return img3d, self.disparity, self.left_frame

cam = OAK_Camera()
