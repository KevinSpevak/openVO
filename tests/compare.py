import os
import pickle

import cv2

from openVO import StereoCamera, StereoOdometer, drawPoseOnImage
from openVO.oakd import OAK_Camera, OAK_Odometer

# create oak pieces
oak_cam = OAK_Camera()
oak_odom = OAK_Odometer(oak_cam,
                        rigidity_threshold=0,
                        outlier_threshold=0,)
oak_cam.start(block=True)

# create the stereo camera object
stereo_camera = StereoCamera(
    oak_cam._K_left,
    oak_cam._D_left,
    oak_cam._K_right,
    oak_cam._D_right,
    {
        "R": oak_cam._R_primary,
        "T": oak_cam._T_primary,
    },
    {
        "minDisparity": 0,
        "numDisparities": 80,
        "blockSize": 1,
        "P1": 0,
        "P2": 135,
        "disp12MaxDiff": 2,
        "preFilterCap": 0,
        "uniquenessRatio": 0,
        "speckleWindowSize": 50,
        "speckleRange": 2,
    },
    (640, 480),
)

# create the odometry object
odometer = StereoOdometer(
    stereo_camera,
    nfeatures=500,
    match_threshold=0.9,
    rigidity_threshold=0,
    outlier_threshold=0,
    preprocessed_frames=False,
    min_matches=10,
)

# process the frames
while True:
    # read the frames
    left_frame, right_frame = oak_cam.left, oak_cam.right
    odometer.update(left_frame, right_frame)

    oak_odom.update()

    # get the current pose
    pose = odometer.current_pose()
    oak_pose = oak_odom.current_pose()

    # print the poses
    print("Stereo pose: ", pose)
    print("OAK pose: ", oak_pose)

    # draw the pose on the image
    drawPoseOnImage(pose, left_frame)
    # write Stereo on left_Rect
    cv2.putText(left_frame, "Stereo", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    drawPoseOnImage(oak_pose, right_frame)
    # write OAK on right_Rect
    cv2.putText(right_frame, "OAK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # combine the two images side by side
    combined = cv2.hconcat([left_frame, right_frame])

    # display the frames
    cv2.imshow("Result", combined)
    # kill program if q is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        oak_cam.stop()
        break
