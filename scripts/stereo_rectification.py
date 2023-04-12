import os
import cv2
import numpy as np
import pickle

# Look for chessboard pattern in 1 out of this many video frames
CALIBRATION_VIDEO_FRAMES_PER_ITERATION = 4

# Interactively capture images for calibration and store them in the given directory
def capture_rectification_images(left_port, right_port, chessboard_size, save_directory):
    for dir in ["left", "right"]:
        assert (os.path.exists(save_directory+dir))
    done = False
    picture_number = 0
    corner_sets_left = []
    corner_sets_right = []
    left_cam = cv2.VideoCapture(left_port)
    right_cam = cv2.VideoCapture(right_port)
    while not done:
        selecting = True
        while selecting:
            got_left, left = left_cam.read()
            got_right, right = right_cam.read()
            if got_left and got_right:
                disp_left = left.copy()
                disp_right = right.copy()
                for corner_set in corner_sets_left:
                    cv2.drawChessboardCorners(disp_left, chessboard_size, corner_set, True)
                for corner_set in corner_sets_right:
                    cv2.drawChessboardCorners(disp_right, chessboard_size, corner_set, True)
                disp = np.concatenate((disp_left, disp_right), axis=1)
                cv2.imshow("enter: capture, escape: done", disp)
                # press enter to capture, escape to finish
                key = cv2.waitKey(40)
                if key == 13:
                    selecting = False
                elif key == 27:
                    print("exiting")
                    selecting = False
                    done = True
            else:
                raise Exception("couldn't read from cameras")
        if not done:
            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            found_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
            found_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
            if found_left and found_right:
                print("Saving calibration image " + str(picture_number))
                cv2.imwrite(save_directory+ "left/" + str(picture_number) + ".png", left)
                cv2.imwrite(save_directory+ "right/" + str(picture_number) + ".png", right)
                picture_number += 1
                corner_sets_left.append(corners_left)
                corner_sets_right.append(corners_right)
            else:
                print("Couldn't find chessboard in both images")

# Generate rectification parameters from stereo image pairs of a chessboard
# requires calibration values (Intrinsic Matrix, Distortion Parameters) for each camera
def generate_rectification_parameters(stereo_img_path, calib_file_left, calib_file_right,
                                      out_file, chessboard_size, square_size):

    for file in [stereo_img_path, calib_file_left, calib_file_right]:
        assert(os.path.exists(file))
    calib_left = pickle.load(open(calib_file_left, "rb"))
    calib_right = pickle.load(open(calib_file_right, "rb"))
    cols, rows = chessboard_size
    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = square_size * np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    img_points_left, img_points_right = [], []
    img_size = None
    showing = True
    img_num = 0
    skipped = 0

    # criteria for refining corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Finding chessboard corners in images...")
    while True:
        left_file = stereo_img_path + "left/" + str(img_num) + ".png"
        right_file = stereo_img_path + "right/" + str(img_num) + ".png"
        if not (os.path.exists(left_file) and os.path.exists(right_file)):
            print("Processed", img_num, "stereo image pairs")
            break
        img_num += 1
        img_left = cv2.cvtColor(cv2.imread(left_file), cv2.COLOR_BGR2GRAY)
        img_right = cv2.cvtColor(cv2.imread(right_file), cv2.COLOR_BGR2GRAY)
        found_left, corners_left = cv2.findChessboardCorners(img_left, chessboard_size, None)
        found_right, corners_right = cv2.findChessboardCorners(img_right, chessboard_size, None)
        if not (found_left and found_right):
            print("Skipping image", img_num - 1)
            skipped += 1
            continue
        corners_left = cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1),criteria)
        corners_right = cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)

        if (showing):
            cv2.drawChessboardCorners(img_left, chessboard_size, corners_left, True)
            cv2.drawChessboardCorners(img_right, chessboard_size, corners_right, True)
            cv2.imshow("Left (any key: next - escape: skip)", img_left)
            cv2.imshow("Right (any key: next - escape: skip)", img_right)
            if cv2.waitKey(0) == 27:
                showing = False

            img_points_left.append(corners_left)
            img_points_right.append(corners_right)

            if not img_size:
                img_size = tuple(reversed(img_left.shape))
    print("Starting Rectification...")
    success, _, _, _, _, R, T, E, F = cv2.stereoCalibrate([obj_points] * img_num, img_points_left, img_points_right,
                                                          calib_left['K'], calib_left['dist'], calib_right['K'], calib_right['dist'],
                                                          img_size, flags=cv2.CALIB_FIX_INTRINSIC)
    print("R: ", R)
    print("T: ", T)
    print("E: ", E)
    print("F: ", F)
    pickle.dump({"R": R, "T": T, "E": E, "F": F}, open(out_file, "wb"))

if __name__ == "__main__":
    calibPath = "calibration/"
    #capture_rectification_images(4, 2, (8,6), calibPath + "StereoCaptures/")
    generate_rectification_parameters(
        calibPath + "StereoCaptures/",
        calibPath + "intrinsics_left.p",
        calibPath + "intrinsics_right.p",
        calibPath + "rectParams.p", (8, 6), 0.0235)
