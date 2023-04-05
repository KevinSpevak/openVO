from threading import Thread
import time

from openVO.oakd import OAK_Camera


STOPPED = False
def target():
    while not STOPPED:
        print(cam.compute_3d())
        time.sleep(1)

cam = OAK_Camera()

cam.start()
cam.start_display()

thread = Thread(target=target)
thread.start()

input("Press Enter to continue...")

STOPPED = True
thread.join()
cam.stop()
