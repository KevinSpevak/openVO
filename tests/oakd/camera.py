from openVO.oakd import OAK_Camera


cam = OAK_Camera()

cam.start()
cam.start_display()

input("Press Enter to continue...")

cam.stop()
