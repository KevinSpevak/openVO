# Utility for finding a good set of options for OpenCV's StereoSGBM
import cv2
import pickle
import sys
import numpy as np


VIDEO_LEFT_PATH="../examples/data/videos/rocks/stereo_left.avi"
VIDEO_RIGHT_PATH="../examples/data/videos/rocks/stereo_right.avi"
CALIB_FILE_LEFT="../examples/data/calibration/intrinsics_left.p"
CALIB_FILE_RIGHT="../examples/data/calibration/intrinsics_right.p"
RECT_FILE="../examples/data/calibration/rectParams.p"

def find_sgbm_settings(calib_file_left, calib_file_right, rect_file, video_left, video_right, img_size):
    cam_left = pickle.load(open(calib_file_left, "rb"))
    cam_right = pickle.load(open(calib_file_right, "rb"))
    rect_params = pickle.load(open(rect_file, "rb"))
    img_height, img_width = img_size

    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cam_left['K'], cam_left['dist'], cam_right['K'],
        cam_right['dist'], (img_width, img_height), rect_params['R'],
        rect_params['T'])
    map_left_1, map_left_2 = cv2.initUndistortRectifyMap(cam_left['K'], cam_left['dist'], R1, P1, (img_width, img_height), cv2.CV_16SC2)
    map_right_1, map_right_2 = cv2.initUndistortRectifyMap(cam_right['K'], cam_right['dist'], R2, P2, (img_width, img_height), cv2.CV_16SC2)

    # min disparity, num disparities, block size, p1, p2,
    # disp12MaxDiff, prefilter cap, uniqueness ratio,
    # speckle window size, speckle range
    sgbm_vars = [0, 80, 1, 0, 675, 2, 0, 6, 50, 2]
    print("min disparity, num disparities, block size, p1, p2, disp12MaxDiff, prefilter cap, uniqueness ratio, speckle window size, speckle range", sgbm_vars)
    stereo = cv2.StereoSGBM_create(*sgbm_vars)
    cv2.namedWindow("disparity")
    dummy_img = np.zeros((1,1), np.float32)
    cv2.imshow("Trackbars", dummy_img)

    def display():
        disparity = stereo.compute(rectified_left, rectified_right).astype(np.float32)
        max, min = disparity.max(), disparity.min()
        normalized = 255 * (disparity - min)/(max-min)
        bad = 255 * (disparity < 0)
        disp = np.stack((normalized, normalized, bad), axis=2).astype(np.uint8)
        cv2.imshow("disparity", disp)

    def update_min_disp(val):
        stereo.setMinDisparity(val)
        sgbm_vars[0] = val
        display()
    cv2.createTrackbar("Min Disparity", "Trackbars", 0, 15, update_min_disp)

    def update_num_disp(val):
        stereo.setNumDisparities(16 * val)
        sgbm_vars[1] = 16 * val
        display()
    cv2.createTrackbar("Num Disparities/16", "Trackbars", 1, 10, update_num_disp)

    def update_block_size(val):
        print("value:", val)
        stereo.setBlockSize(val * 2 + 1)
        sgbm_vars[2] = val * 2 + 1
        print("update:", sgbm_vars[2])
        display()
    cv2.createTrackbar("(block_size-1)/2", "Trackbars", 0, 10, update_block_size)

    # example value: 8 * blockSize^2
    def update_p1(val):
        stereo.setP1(val * sgbm_vars[2]**2)
        sgbm_vars[3] = val * sgbm_vars[2]**2
        display()
    cv2.createTrackbar("P1/block_size^2", "Trackbars", 0, 15, update_p1)

    # example value: 32 * blockSize^2
    def update_p2(val):
        stereo.setP2(3 * val * sgbm_vars[2]**2)
        sgbm_vars[4] = 3 * val * sgbm_vars[2]**2
        display()
    cv2.createTrackbar("P2/(3*block_size%2)", "Trackbars", 20, 50, update_p2)


    def update_max_disp(val):
        stereo.setDisp12MaxDiff(val)
        sgbm_vars[5] = val
        display()
    cv2.createTrackbar("Max Disparity", "Trackbars", 0, 50, update_max_disp)

    def update_prefilter_cap(val):
        stereo.setPreFilterCap(val * 10)
        sgbm_vars[6] = val * 10
        display()
    cv2.createTrackbar("Prefilter Cap", "Trackbars", 0, 10, update_prefilter_cap)

    # Usually 5-15
    def update_uniq_ratio(val):
        stereo.setUniquenessRatio(val)
        sgbm_vars[7] = val
        display()
    cv2.createTrackbar("Uniqueness Ratio", "Trackbars", 1, 20, update_uniq_ratio)

    # usually 50-200
    def update_speckle_size(val):
        stereo.setSpeckleWindowSize(val * 10)
        sgbm_vars[8] = val * 10
        display()
    cv2.createTrackbar("speckle_size/10", "Trackbars", 0, 25, update_speckle_size)

    # usually 1-2
    def update_speckle_range(val):
        stereo.setSpeckleRange(val)
        sgbm_vars[9] = val
        display()
    cv2.createTrackbar("Speckle Range", "Trackbars", 0, 4, update_speckle_range)

    def update_mode(val):
        stereo.setMode(val)
        display()
#    cv2.createTrackbar("Mode", "Trackbars", 0, 3, update_mode)

    while True:
        got_image_left, img_left = video_left.read()  # Make sure we can read video from the left camera
        got_image_right, img_right = video_right.read()  # Make sure we can read video from the right camera

        if not got_image_left or not got_image_right:
            print("Cannot read video source")
            sys.exit()

        rectified_left = cv2.remap(img_left, map_left_1, map_left_2, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, map_right_1, map_right_2, cv2.INTER_LINEAR)
        display()

        if cv2.waitKey(0) == 27:
            break

    params = {"minDisparity": sgbm_vars[0], "numDisparities": sgbm_vars[1], "blockSize": sgbm_vars[2],
              "P1": sgbm_vars[3], "P2": sgbm_vars[4], "disp12MaxDiff": sgbm_vars[5],
              "preFilterCap": sgbm_vars[6], "uniquenessRatio": sgbm_vars[7],
              "speckleWindowSize": sgbm_vars[8], "speckleRange": sgbm_vars[9]}
    print(params)

if __name__ == "__main__":
    #video_left = cv2.VideoCapture(4)
    #video_right = cv2.VideoCapture(2)
    video_left = cv2.VideoCapture(VIDEO_LEFT_PATH)
    video_right = cv2.VideoCapture(VIDEO_RIGHT_PATH)
    got_left, img_left = video_left.read()  # Make sure we can read video from the left camera
    got_right, img_right = video_right.read()  # Make sure we can read video from the right camera

    if not got_left or not got_right:
        print("Cannot read video source")
        sys.exit()

    find_sgbm_settings(CALIB_FILE_LEFT,
                       CALIB_FILE_RIGHT,
                       RECT_FILE,
                       video_left, video_right, img_left.shape[0:2])
