import cv2
import os
import pickle
import numpy as np
from openVO import StereoCamera

config_path = "../examples/data/calibration/"
assert(os.path.exists(config_path))
intr_left = pickle.load(open(config_path + "intrinsics_left.p", "rb"))
K_left, dist_left = intr_left["K"], intr_left["dist"]
intr_right = pickle.load(open(config_path + "intrinsics_right.p", "rb"))
K_right, dist_right = intr_right["K"], intr_right["dist"]
rect_params = pickle.load(open(config_path + "rectParams.p", "rb"))
print("rect params", rect_params)

# Generated with script
# {'minDisparity': 0, 'numDisparities': 80, 'blockSize': 3, 'P1': 0, 'P2': 675, 'disp12MaxDiff': 2, 'preFilterCap': 0, 'uniquenessRatio': 6, 'speckleWindowSize': 50, 'speckleRange': 2}
# Existing params
# "minDisparity": 0,
# "numDisparities": 48,
# "blockSize": 3,
# "P1": 27,
# "P2": 675,
# "disp12MaxDiff": 50,
# "preFilterCap": 0,
# "uniquenessRatio": 10,
# "speckleWindowSize": 0,
# "speckleRange": 4}

sgbm_params = {"minDisparity": 0,
               "numDisparities": 80,
               "blockSize": 3,
               "P1": 27,
               "P2": 675,
               "disp12MaxDiff": 2,
               "preFilterCap": 0,
               "uniquenessRatio": 6,
               "speckleWindowSize": 50,
               "speckleRange": 2}

cam_left = cv2.VideoCapture(2)
cam_right = cv2.VideoCapture(4)

# Adjust to overlay images (depends on distance to object)
shift = 100 # pixels

while True:
    got_left, left = cam_left.read()
    got_right, right = cam_right.read()
    if not (got_left and got_right):
        break
    combined = np.concatenate((left, right), axis=1)
    cv2.imshow("original", combined)
    undist_left = cv2.undistort(left, K_left, dist_left)
    undist_right = cv2.undistort(right, K_right, dist_right)
    undist = np.concatenate((undist_left, undist_right), axis=1)
    cv2.imshow("Undistorted", undist)

    stereo = StereoCamera(K_left, dist_left, K_right, dist_right, rect_params, sgbm_params, (left.shape[1], left.shape[0]))
    rect_left = stereo.undistort_rectify_left(left)
    rect_right = stereo.undistort_rectify_right(right)
    rect = np.concatenate((rect_left, rect_right), axis=1)
    cv2.imshow("Rectified", rect)
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    gray_undist_left = cv2.cvtColor(undist_left, cv2.COLOR_BGR2GRAY)
    gray_undist_right = cv2.cvtColor(undist_right, cv2.COLOR_BGR2GRAY)
    gray_rect_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_rect_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    h, w = gray_left.shape
    shift_block = np.zeros((h, shift))
    red_channel = np.zeros((h, w+shift))
    gray_left = np.concatenate((gray_left, shift_block), axis = 1)
    gray_undist_left = np.concatenate((gray_undist_left, shift_block), axis = 1)
    gray_rect_left = np.concatenate((gray_rect_left, shift_block), axis = 1)
    gray_right = np.concatenate((shift_block, gray_right), axis = 1)
    gray_undist_right = np.concatenate((shift_block, gray_undist_right), axis = 1)
    gray_rect_right = np.concatenate((shift_block, gray_rect_right), axis = 1)
    stacked = np.stack((gray_left, gray_right, red_channel), axis=2).astype(np.uint8)
    stacked_undist = np.stack((gray_undist_left, gray_undist_right, red_channel), axis=2).astype(np.uint8)
    stacked_rect = np.stack((gray_rect_left, gray_rect_right, red_channel), axis=2).astype(np.uint8)

    for x in range(h//10 - 1):
        cv2.line(stacked, (0, 10 * x), (w+shift - 1, 10 * x), (0,0,255))
        cv2.line(stacked_undist, (0, 10 * x), (w+shift - 1, 10 * x), (0,0,255))
        cv2.line(stacked_rect, (0, 10 * x), (w+shift - 1, 10 * x), (0,0,255))
    cv2.imshow("Stacked Original", stacked)
    cv2.imshow("Stacked Undistoreted", stacked_undist)
    cv2.imshow("Stacked Rectified", stacked_rect)

    _, disparity, _ = stereo.compute_3d(left, right)
    max, min = disparity.max(), disparity.min()
    normed_disp = (disparity - min) / (max - min)
    cv2.imshow('disparity', normed_disp)
    if cv2.waitKey(40) == 27:
        break
