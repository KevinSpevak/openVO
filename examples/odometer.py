import os
import time
import pickle

import cv2

from openVO import StereoCamera, StereoOdometer, drawPoseOnImage


# options are "rocks", "grass", "pavement"
VIDEO = "grass"

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
VIDEO_DIR = os.path.join(DIR_PATH, "data", "videos", VIDEO)
LEFT_VID = os.path.join(VIDEO_DIR, "stereo_left.avi")
RIGHT_VID = os.path.join(VIDEO_DIR, "stereo_right.avi")
CALIBRATION_DIR = os.path.join(DIR_PATH, "data", "calibration")
# get the actual calibration files
LEFT = os.path.join(CALIBRATION_DIR, "intrinsics_left.p")
RIGHT = os.path.join(CALIBRATION_DIR, "intrinsics_right.p")
RECT_PARAMS = os.path.join(CALIBRATION_DIR, "rectParams.p")
SGBM_PARAMS = os.path.join(CALIBRATION_DIR, "sgbmParams.p")

# create the stereo camera object
stereo_camera = StereoCamera.from_pfiles(
    LEFT, RIGHT, RECT_PARAMS, SGBM_PARAMS, (640, 480)
)

# create the odometry object
odometer = StereoOdometer(
    stereo_camera,
    nfeatures=1000,
    match_threshold=0.9,
    rigidity_threshold=0.06,
    outlier_threshold=0.1,
    preprocessed_frames=False,
    min_matches=10,
)

# load the video
# ensure the videos exist
assert os.path.exists(LEFT_VID), "Left video " + LEFT_VID + " does not exist"
assert os.path.exists(RIGHT_VID), f"Right video " + RIGHT_VID + " does not exist"
left_vid = cv2.VideoCapture(LEFT_VID)
right_vid = cv2.VideoCapture(RIGHT_VID)

# process the frames
while left_vid.isOpened() and right_vid.isOpened():
    # read the frames
    ret, left_frame = left_vid.read()
    ret, right_frame = right_vid.read()

    # if we reached the end of the video, break
    if not ret:
        break

    # compute the odometry
    start_time = time.perf_counter()
    odometer.update(left_frame, right_frame)
    print(f"Elapsed time for odometer update: {time.perf_counter() - start_time}")

    # get the current pose
    pose = odometer.current_pose()

    # # print the pose
    # print(pose)

    # draw the pose on the image
    drawPoseOnImage(pose, left_frame)

    # display the frames
    cv2.imshow("left", left_frame)
    # kill program if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
