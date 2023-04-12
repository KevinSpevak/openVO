import os

from openVO import StereoCamera, StereoOdometer, drawPoseOnImage
from openVO.oakd import OAK_Camera


# options are "rocks", "grass", "pavement"
VIDEO = "grass"

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = os.path.join(DIR_PATH, "..", "..", "examples")
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

# print out all the data
print("CUSTOM STEREO SETUP")
print("Left Camera Matrix:")
print(stereo_camera.K_left)
print("Left Distortion Coefficients:")
print(stereo_camera.dist_left)
print("Right Camera Matrix:")
print(stereo_camera.K_right)
print("Right Distortion Coefficients:")
print(stereo_camera.dist_right)
print("Rectification Parameters:")
print(stereo_camera.rectParams)
print("SGBM Parameters:")
print(stereo_camera.sgbmParams)
print("Q matrix:")
print(stereo_camera.Q)

# create OAK_CAMERA 
oak_cam = OAK_Camera()

# print out all the data
print("OAKD STEREO SETUP")
print("Left Camera Matrix:")
print(oak_cam._K_left)
print("Left Distortion Coefficients:")
print(oak_cam._D_left)
print("Right Camera Matrix:")
print(oak_cam._K_right)
print("Right Distortion Coefficients:")
print(oak_cam._D_right)
print("L to R rotation rectification matrix:")
print(oak_cam._R1)
print("L to R translation rectification matrix:")
print(oak_cam._T1)
print("R to L rotation rectification matrix:")
print(oak_cam._R2)
print("R to L translation rectification matrix:")
print(oak_cam._T2)

print("Q matrix:")
print(oak_cam._Q_primary)
