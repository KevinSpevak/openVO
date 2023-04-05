from threading import Thread
import time

from openVO.oakd import OAK_Odometer, OAK_Camera


STOPPED = False
def target():
    global STOPPED
    while not STOPPED:
        print(odom.current_pose)
        time.sleep(1)

cam = OAK_Camera()
cam.start()

time.sleep(5)

odom = OAK_Odometer(cam)

odom.start()
odom.cam.start_display()

t = Thread(target=target)
t.start()

input("Press Enter to continue...")

STOPPED = True
t.join()
odom.stop()
