import cv2
import numpy as np
import sys
import pickle

class StereoCamera:
    @classmethod
    def from_pfiles(cls, left_cam_file, right_cam_file, rect_file, sgbm_file, img_size):
        cam_left = pickle.load(open(left_cam_file, "rb"))
        cam_right = pickle.load(open(right_cam_file, "rb"))
        rect_params = pickle.load(open(rect_file, "rb"))
        sgbm_params = pickle.load(open(sgbm_file, "rb"))
        return cls(cam_left['K'], cam_left['dist'], cam_right['K'],
                   cam_right['dist'], rect_params, sgbm_params, img_size)

    def __init__(self, K_left, dist_left, K_right, dist_right, rect_params, sgbm_params, img_size):
        R1, R2, P1, P2, self.Q, self.valid_region_left, self.valid_region_right = cv2.stereoRectify(
            K_left, dist_left, K_right, dist_right, img_size, rect_params['R'], rect_params['T'])
        self.map_left_1, self.map_left_2 = cv2.initUndistortRectifyMap(
            K_left, dist_left, R1, P1, img_size, cv2.CV_16SC2)
        self.map_right_1, self.map_right_2 = cv2.initUndistortRectifyMap(
            K_right, dist_right, R2, P2, img_size, cv2.CV_16SC2)
        self.stereoSGBM = cv2.StereoSGBM_create(
            sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
            sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
            # TODO: mode=1 still needed?
            0 and sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"], mode=1)
        print("sgbm",
              sgbm_params["minDisparity"], sgbm_params["numDisparities"], sgbm_params["blockSize"],
              sgbm_params["P1"], sgbm_params["P2"], sgbm_params["disp12MaxDiff"], sgbm_params["preFilterCap"],
              0 and sgbm_params["uniquenessRatio"], sgbm_params["speckleWindowSize"], sgbm_params["speckleRange"])

    def undistort_rectify_left(self, img):
        return cv2.remap(img, self.map_left_1, self.map_left_2, cv2.INTER_LINEAR)

    def undistort_rectify_right(self, img):
        return cv2.remap(img, self.map_right_1, self.map_right_2, cv2.INTER_LINEAR)

    def crop_to_valid_region_left(self, img):
        return img[self.valid_region_left[1]: self.valid_region_left[3],
                   self.valid_region_left[0]: self.valid_region_left[2]]

    def crop_to_valid_region_right(self, img):
        return img[self.valid_region_right[1]: self.valid_region_right[3],
                   self.valid_region_right[0]: self.valid_region_right[2]]

    def compute_3d(self, img_left, img_right):
        img_left = self.undistort_rectify_left(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY))
        img_right = self.undistort_rectify_right(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))
        disparity = self.stereoSGBM.compute(img_left, img_right).astype(np.float32)/16
        disp_img = (disparity - disparity.min())/(disparity.max()-disparity.min())
        img_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        return (self.crop_to_valid_region_left(img_3d),
                self.crop_to_valid_region_left(disparity),
                self.crop_to_valid_region_left(img_left))
