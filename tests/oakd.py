import time

from openVO.oakd import OAK_Camera, OAK_Odometer

odom = OAK_Odometer(OAK_Camera())

odom.start()

odom.cam.start_display()

input("Press Enter to stop...")

odom.stop()
