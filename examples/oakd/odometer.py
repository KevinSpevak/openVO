import cv2
from openVO import drawPoseOnImage
from openVO.oakd import OAK_Odometer


odom = OAK_Odometer()
cam = odom.stereo

while True:
    odom.update()
    rgb_frame = cam.rgb
    rgb_frame = cv2.resize(rgb_frame, (640, 480))
    pose = odom.current_pose()
    print(pose)
    drawPoseOnImage(pose, rgb_frame)
    cv2.imshow("Annotated", rgb_frame)
    # quit if the user presses q
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cam.stop()
