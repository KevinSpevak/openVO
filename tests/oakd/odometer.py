import time

from openVO.oakd import OAK_Odometer, OAK_Camera


cam = OAK_Camera()
cam.start()

time.sleep(5)

odom = OAK_Odometer(cam)

odom.start()
odom.cam.start_display()

input("Press Enter to continue...")

odom.stop()
