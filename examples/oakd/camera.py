from threading import Thread
import time

from openVO.oakd import OAK_Camera


STOPPED = False
def target():
    while not STOPPED:

        print("pose")
        print(cam.imu_pose)
        print("rotation")
        print(cam.imu_rotation)
        
        time.sleep(1)

cam = OAK_Camera(
    display_depth=True,
    display_point_cloud=True
)

cam.start()
cam.start_display()

thread = Thread(target=target)
thread.start()

input("Press Enter to continue...")

STOPPED = True
thread.join()
cam.stop()
