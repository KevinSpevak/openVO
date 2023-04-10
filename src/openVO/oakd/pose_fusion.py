from typing import Optional
import time

import numpy as np


def create_covariance_matrices(imu_uncertainty=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vo_uncertainty=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
    # Define the state vector and measurement vector sizes
    state_size = 7  # (x, y, z, q0, q1, q2, q3)
    measurement_size = 16  # (IMU pose, VO pose)

    # Create the covariance matrices for IMU and VO data
    imu_covariance = np.zeros((measurement_size, measurement_size))
    vo_covariance = np.zeros((measurement_size, measurement_size))

    # Fill in the diagonal elements of the covariance matrices
    for i in range(state_size):
        imu_covariance[i, i] = imu_uncertainty[i]**2
        vo_covariance[i + state_size, i + state_size] = vo_uncertainty[i]**2

    return imu_covariance, vo_covariance

class PoseFusion:
    def __init__(self, imu_covariance: Optional[np.ndarray], vo_covariance: Optional[np.ndarray]):
        if imu_covariance is None:
            imu_covariance, _ = create_covariance_matrices()
        if vo_covariance is None:
            _, vo_covariance = create_covariance_matrices()
        
        # Define the state vector and measurement vector sizes
        self.state_size = 7  # (x, y, z, q0, q1, q2, q3)
        self.measurement_size = 16  # (IMU pose, VO pose)
        
        # Define the state transition matrix (constant velocity model)
        self.F = np.eye(self.state_size)
        self.F[:3, 3:7] = 0.5 * np.array([[0, 0, 0, 0],
                                           [0, 0, 0, -1],
                                           [0, 0, 0, 1],
                                           [0, 1, -1, 0]])
        
        # Define the measurement matrix
        self.H = np.zeros((self.measurement_size, self.state_size))
        self.H[:8, :7] = np.eye(7)  # IMU pose
        self.H[8:, :7] = np.eye(7)  # VO pose
        self.H[8:, :3] = np.eye(3)  # VO position offset
        
        # Define the process noise covariance matrix (IMU noise model)
        self.Q = np.zeros((self.state_size, self.state_size))
        self.Q[3:, 3:] = imu_covariance
        
        # Define the measurement noise covariance matrix
        self.R = np.zeros((self.measurement_size, self.measurement_size))
        self.R[:8, :8] = imu_covariance
        self.R[8:, 8:] = vo_covariance
        
        # Initialize the state estimate and covariance matrix
        self.x = np.zeros((self.state_size, 1))
        self.P = np.eye(self.state_size)

        # start time
        self.prev_time: Optional[float] = 0.0
    
    def update(self, imu_pose, imu_velocity, imu_acceleration, vo_pose):
         # Convert the IMU pose to a quaternion
        q_imu = imu_pose[:3, :3]
        t_imu = imu_pose[:3, 3]
        q0 = np.sqrt(1 + q_imu[0, 0] + q_imu[1, 1] + q_imu[2, 2]) / 2
        q1 = (q_imu[2, 1] - q_imu[1, 2]) / (4 * q0)
        q2 = (q_imu[0, 2] - q_imu[2, 0]) / (4 * q0)
        q3 = (q_imu[1, 0] - q_imu[0, 1]) / (4 * q0)
        imu_state = np.vstack((t_imu, q0, q1, q2, q3))
        
        # Concatenate the IMU and VO poses into a single measurement vector
        measurement = np.vstack((imu_state, vo_pose.reshape((12, 1))))
        
        # dt = time.perf_counter() - self.prev_time
        dt = 0.01
        
        # Update the state transition matrix and process noise covariance matrix with the IMU data
        omega = imu_velocity.reshape((3, 1))
        a = imu_acceleration.reshape((3, 1))
        wx = np.zeros((3, 3))
        wx[1, 0] = omega[2, 0]
        wx[2, 0] = -omega[1, 0]
        wx[0, 1] = -omega[2, 0]
        wx[2, 1] = omega[0, 0]
        wx[0, 2] = omega[1, 0]
        wx[1, 2] = -omega[0, 0]
        self.F[:3, :3] = np.eye(3) + wx * dt
        ax = np.zeros((3, 3))
        ax[0, 1] = -a[2, 0]
        ax[0, 2] = a[1, 0]
        ax[1, 0] = a[2, 0]
        ax[1, 2] = -a[0, 0]
        ax[2, 0] = -a[1, 0]
        ax[2, 1] = a[0, 0]
        self.Q[3:, 3:] = np.dot(ax, ax.T) * dt + np.eye(3) * 0.01
        
        # Predict the next state estimate and covariance matrix using the state transition matrix and process noise
        x_prior = np.dot(self.F, self.x)
        P_prior = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        
        # Compute the Kalman gain matrix
        K = np.dot(np.dot(P_prior, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, P_prior), self.H.T) + self.R))
        
        # Update the state estimate and covariance matrix using the measurement and Kalman gain
        x_post = x_prior + np.dot(K, (measurement - np.dot(self.H, x_prior)))
        P_post = np.dot(np.eye(self.state_size) - np.dot(K, self.H), P_prior)
        
        # Convert the state estimate back to a transformation matrix
        t_post = x_post[:3].reshape((3, 1))
        q_post = x_post[3:]
        R_post = self.quat_to_rot(q_post)
        pose_post = np.hstack((R_post, t_post))
        pose_post = np.vstack((pose_post, np.array([0, 0, 0, 1])))
        
        # Update the state and covariance matrix for the next iteration
        self.x = x_post
        self.P = P_post
        
        return pose_post
