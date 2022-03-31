import numpy as np

def rot2RPY(T):
    # Roll : RotX
    # Pitch : RotY
    # Yaw : RotZ

    R = T[0:3,0:3]

    roll = np.zeros((2, 1)) # psi
    pitch = np.zeros((2, 1)) # theta
    yaw = np.zeros((2, 1)) # phi

    cos_theta = np.sqrt(R[0][0]**2 + R[1][0]**2)
    eps = 1E-4

    if ((cos_theta < eps) and (cos_theta > -eps)):

        pitch[0] = -R[2][0] * (np.pi / 2)
        pitch[1] = -R[2][0] * (np.pi / 2)

        roll[0] = R[2][0] * np.arctan2(-R[0][1], R[1][1])
        roll[1] = R[2][0] * np.arctan2(-R[0][1], R[1][1])

        yaw[0] = 0
        yaw[1] = 0

    else:
        pitch[0] = np.arctan2(-R[2][0], cos_theta)
        pitch[1] = np.arctan2(-R[2][0], -cos_theta)

        roll[0] = np.arctan2((R[2][1] / np.cos(pitch[0])), (R[2][2] / np.cos(pitch[0])))
        roll[1] = np.arctan2((R[2][1] / np.cos(pitch[1])), (R[2][2] / np.cos(pitch[1])))

        yaw[0] = np.arctan2((R[1][0] / np.cos(pitch[0])), (R[0][0] / np.cos(pitch[0])))
        yaw[1] = np.arctan2((R[1][0] / np.cos(pitch[1])), (R[0][0] / np.cos(pitch[1])))

    return roll, pitch, yaw
