import time

import cv2
from openVO import drawPoseOnImage
from openVO.oakd import OAK_Camera, OAK_Odometer


STOPPED = False


cam = OAK_Camera()
cam.start(block=True)
odom = OAK_Odometer(cam)
odom.start()

while True:
    rgb_frame = cam.rgb
    drawPoseOnImage(odom.current_pose(), rgb_frame)
    cv2.imshow("Annotated", rgb_frame)
    # quit if the user presses q
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

odom.stop()
cam.stop()
