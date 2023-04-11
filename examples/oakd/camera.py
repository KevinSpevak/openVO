from threading import Thread
import time

import numpy as np
import cv2

from openVO.oakd import OAK_Camera


STOPPED = False


def target():
    while not STOPPED:
        # when on_demand flag is set to false, computed everytime new data is ready
        # when on_demand flag is set to true, computed only when the function is called
        cam.compute_im3d()
        cam.compute_point_cloud()

        im3d = cv2.reprojectImageTo3D(cam.disparity, cam._Q_primary)

        time.sleep(1)


cam = OAK_Camera(
    display_depth=True,
    display_mono=True,
    display_rectified=True,
    display_point_cloud=False,
    compute_im3d_on_demand=True,
    compute_point_cloud_on_demand=True,
)

cam.start_display()
cam.start()

thread = Thread(target=target)
thread.start()

input("Press Enter to continue...")

STOPPED = True
thread.join()
cam.stop()
