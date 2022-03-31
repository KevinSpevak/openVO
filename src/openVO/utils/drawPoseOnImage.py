import cv2
import numpy as np
from openVO.utils.rot2RPY import rot2RPY

def drawPoseOnImage(T, img):
    roll, pitch, yaw = rot2RPY(T)
    # pick RPY representation with smaller magnitude rotations
    rep1, rep2 = [np.linalg.norm([roll[i], pitch[i], yaw[i]]) for i in [0, 1]]
    if rep1 > rep2:
        r = roll[1]
        p = pitch[1]
        y = yaw[1]
    else:
        r = roll[0]
        p = pitch[0]
        y = yaw[0]

    t_x = float(T[0,3])
    t_y = float(T[1,3])
    t_z = float(T[2,3])
    image_height = img.shape[0]
    image_width = img.shape[1]
    # Display RPY in frame similar to aircraft definition:
    # x/roll = forward = camera z/yaw
    # y/ptich = up = -1 * camera y/pitch
    # z/yaw = right = camera x/roll
    pose_text1 = 'Roll = '+ str(np.round(y, 3))
    pose_text2 = 'Pitch = ' + str(np.round(-p, 3))
    pose_text3 = 'Yaw = ' + str(np.round(r, 3))
    pose_text4 = 'x,y,z = ' + str(np.round(t_x, 1)) + ', ' + str(np.round(t_y, 1)) + ', ' + str(np.round(t_z, 1))
    cv2.putText(img, text=pose_text1, org=(0, image_height - 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0, color=(0, 0, 255), thickness=3)
    cv2.putText(img, text=pose_text2, org=(0, image_height - 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0, color=(0, 0, 255), thickness=3)
    cv2.putText(img, text=pose_text3, org=(0, image_height - 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0, color=(0, 0, 255), thickness=3)
    cv2.putText(img, text=pose_text4, org=(0, image_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.6, color=(0, 0, 255), thickness=3)
