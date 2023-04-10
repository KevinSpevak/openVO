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
        
        # when on_demand flag is set to false, computed everytime new data is ready
        # when on_demand flag is set to true, computed only when the function is called
        cam.compute_im3d()
        cam.compute_point_cloud()

        time.sleep(1)

cam = OAK_Camera(
    display_depth=True,
    display_point_cloud=True,
    compute_im3d_on_demand=True,
    compute_point_cloud_on_demand=True,
)

cam.start()
cam.start_display()

thread = Thread(target=target)
thread.start()

input("Press Enter to continue...")

STOPPED = True
thread.join()
cam.stop()
