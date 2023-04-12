import os
import cv2
import glob
import numpy as np
import pickle

# Function to write K matrix and dist coeffs to npz files
# K matrix is a 3x3 and dist coeffs is of length 4
def writeKandDistNPZ(lk: np.ndarray, rk: np.ndarray, ld: np.ndarray, rd: np.ndarray, calibrationPath):
    # saves the np.arrays inputed to their respective files
    np.save(calibrationPath + "leftK.npy", lk)
    np.save(calibrationPath + "rightK.npy", rk)
    np.save(calibrationPath + "leftDistC.npy", ld)
    np.save(calibrationPath + "rightDistC.npy", rd)

# Interactively capture images for calibration and store them in the given directory
def capture_calibration_images(camera_port, chessboard_size, save_directory):
    assert (os.path.exists(save_directory))
    done = False
    picture_number = 0
    corner_sets = []
    cam = cv2.VideoCapture(camera_port)
    while not done:
        selecting = True
        success, img = cam.read()
        while selecting:
            success, img = cam.read()
            if success:
                disp = img.copy()
                for corner_set in corner_sets:
                    cv2.drawChessboardCorners(disp, chessboard_size, corner_set, True)
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
                raise Exception("couldn't read from camera")
        if not done:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            if found:
                print("Saving calibration image " + str(picture_number))
                cv2.imwrite(save_directory + str(picture_number) + ".png", img)
                picture_number += 1
                corner_sets.append(corners)
            else:
                print("Couldn't find chessboard")


# Find K and distortion parameters from a let of calibration images
# chessboard_size: (cols, rows); square_size: side length of chessboard squares
# Function based on main_calib.py from CO School of Mines CSCI507 course materials
def generate_camera_intrinsics(image_directory, chessboard_size):
    assert(os.path.exists(image_directory))
    filenames = glob.glob(image_directory + "*")
    cols, rows = chessboard_size
    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    img_points = []
    print("finding chessboard corners...")
    for filename in filenames:
        img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(img, chessboard_size, None)
        if success:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            refined_corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(refined_corners)
        else:
          print("Could not find chessboard in " + filename)
    count = len(img_points)
    print("generating calibration data from " + str(count) + " images...")
    success, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj_points] * count, img_points, chessboard_size, None, None)
    print("camera intrinsic matrix:", K)
    print("distortion coefficients:", dist)
    error = 0
    for i in range(count):
        proj_points, _ = cv2.projectPoints(obj_points, rvecs[i], tvecs[i], K, dist)
        error += cv2.norm(proj_points, img_points[i], cv2.NORM_L2)
    print("mean error: {} (should be close to zero)".format(error/(count*cols*rows)))

    for filename in filenames:
        img = cv2.imread(filename)
        undistorted = cv2.undistort(img, K, dist)
        cv2.imshow("original image | esc to quit", img)
        cv2.imshow("adjusted for distortion | esc to quit", undistorted)
        if cv2.waitKey(0) == 27:
            break

    return K, dist


if __name__ == "__main__":
    calibrationPath = "calibration"

    #capture_calibration_images(4, (8, 6), calibrationPath + "/LeftCaptures/")
    #capture_calibration_images(2, (8, 6), calibrationPath + "/RightCaptures/")
    leftK, leftDist = generate_camera_intrinsics(calibrationPath + "/LeftCaptures/", (8, 6))
    pickle.dump({"K": leftK, "dist": leftDist}, open(calibrationPath + "/intrinsics_left.p", "wb"))
    rightK, rightDist = generate_camera_intrinsics(calibrationPath + "/RightCaptures/", (8, 6))
    pickle.dump({"K": rightK, "dist": rightDist}, open(calibrationPath + "/intrinsics_right.p", "wb"))
    writeKandDistNPZ(leftK, rightK, leftDist, rightDist, calibrationPath)
