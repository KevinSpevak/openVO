from threading import Thread
import time

import cv2
from openVO import drawPoseOnImage
from openVO.oakd import OAK_Odometer, OAK_Camera


STOPPED = False
def target():
    global STOPPED
    print(odom.cam._valid_region_left)
    print(odom.cam._valid_region_right)
    while not STOPPED:
        T = odom.current_pose
        img = odom.cam.rgb
        img = cv2.resize(img, (640, 480))
        drawPoseOnImage(T, img)
        cv2.putText(img, odom.skip_cause, (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("Annotated", img)
        cv2.waitKey(33)
    cv2.destroyWindow("Annotated")

cam = OAK_Camera()
cam.start()

time.sleep(5)

odom = OAK_Odometer(cam,
                    nfeatures=10000)

odom.start()
# odom.cam.start_display()

t = Thread(target=target)
t.start()

input("Press Enter to continue...")

STOPPED = True
t.join()
odom.stop()
