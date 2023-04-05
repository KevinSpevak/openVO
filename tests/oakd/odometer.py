from openVO.oakd import OAK_Odometer, OAK_Camera


odom = OAK_Odometer(OAK_Camera())

odom.start()
odom.cam.start_display()

input("Press Enter to continue...")

odom.stop()
