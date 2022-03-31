import cv2
import numpy as np

class StereoOdometer:
    # Range for disparities where we can get an accurate depth calculation
    MIN_VALID_DISPARITY = 4 # pixels
    MAX_VALID_DISPARITY = 100 # pixels
    # Consider feature matches with too large of a change in distance to be false positives
    # skip frames with computed transformation with too large of a change in position
    MAX_DISTANCE_CHANGE = 1 # Meters
    # skip frames with computed transformation with too large change in rotation
    MAX_ROTATION_CHANGE = np.pi/12 # Radians

    def __init__(self, stereo_camera):
        self.stereo = stereo_camera
        # image data for current and previous frames
        self.current_img, self.current_disparity, self.current_3d = None, None, None
        self.prev_img, self.prev_disparity, self.prev_3d = None, None, None
        # orb feature detector and matcher
        self.orb, self.matcher = cv2.ORB_create(), cv2.BFMatcher.create(cv2.NORM_HAMMING)
        # orb key points and descriptors for current and previous frames
        self.prev_kps, self.current_kps = None, None
        self.current_kps, self.current_desc = None, None
        # Number of successive frames with no coordinate transformation found
        self.skipped_frames = 0
        # transformation of the world frame in the camera's coordinate system
        self.c_T_w = np.eye(4)

    # image mask for pixels with acceptable disparity values
    def feature_mask(self, disparity):
        mask = (disparity >= self.MIN_VALID_DISPARITY) * (disparity <= self.MAX_VALID_DISPARITY)
        return mask.astype(np.uint8)*255

    def valid_distance_change(self, prev_kp_idx, current_kp_idx):
        p_x, p_y = self.prev_kps[prev_kp_idx].pt
        c_x, c_y = self.current_kps[current_kp_idx].pt
        return (np.linalg.norm(self.prev_3d[int(p_y)][int(p_x)])
                - np.linalg.norm(self.current_3d[int(c_y)][int(c_x)]) <= self.MAX_DISTANCE_CHANGE * (self.skipped_frames + 1))

    def update(self, img_left, img_right):
        im3d, disp, left = self.stereo.compute_3d(img_left, img_right)
        self.prev_img = self.current_img
        self.prev_disparity = self.current_disparity
        self.prev_3d = self.current_3d
        self.current_img, self.current_disparity, self.current_3d = left, disp, im3d
        self.prev_kps, self.prev_desc = self.current_kps, self.current_desc
        self.current_kps, self.current_desc = self.orb.detectAndCompute(left, self.feature_mask(disp))
        if self.prev_img is None:
            return

        matches = self.matcher.knnMatch(self.prev_desc, self.current_desc, k=2)
        # TODO ambiguous match threshold
        matches = [m[0] for m in matches if m[0].distance < 0.8 * m[1].distance]
        matches = [m for m in matches if self.valid_distance_change(m.queryIdx, m.trainIdx)]

        # TODO consider higher threshold for accuracy?
        if len(matches) < 10:
            self.skipped_frames += 1
            return

        pts_3d = np.array([self.current_3d[int(y)][int(x)] for x, y in [self.current_kps[m.trainIdx].pt for m in matches]])
        prev_pts_3d = np.array([self.prev_3d[int(y)][int(x)] for x, y in [self.prev_kps[m.queryIdx].pt for m in matches]])

        # Single-pass outlier removal using error from estimated transformation
        T, _ = cv2.estimateAffine3D(prev_pts_3d, pts_3d, force_rotation=True)
        T = np.vstack([T, [0,0,0,1]])
        h_pts = np.hstack([pts_3d, np.array([[1]*len(pts_3d)]).transpose()])
        h_prev = np.hstack([prev_pts_3d, np.array([[1]*len(prev_pts_3d)]).transpose()])
        errors = np.array([np.linalg.norm(h_pts[i] - T @ h_prev[i])/np.linalg.norm(h_pts[i]) for i in range(len(h_pts))])
        threshold = 0.03 + np.median(errors) # TODO better outlier detection?
        pts_3d = np.array([pts_3d[i] for i in range(len(pts_3d)) if errors[i] < threshold])
        prev_pts_3d = np.array([prev_pts_3d[i] for i in range(len(prev_pts_3d)) if errors[i] < threshold])

        if len(pts_3d) < 10:
            self.skipped_frames += 1
            return

        T, _ = cv2.estimateAffine3D(np.array(prev_pts_3d), np.array(pts_3d), force_rotation=True)
        T = np.vstack([T, [0,0,0,1]])

        if np.isnan(T).any():
            self.skipped_frames += 1
        else:
            disp = T[0:3, 3]
            rot,_ = cv2.Rodrigues(T[0:3, 0:3])

            if (np.linalg.norm(disp) > self.MAX_DISTANCE_CHANGE  * (self.skipped_frames + 1)
                or np.linalg.norm(rot) > self.MAX_ROTATION_CHANGE  * (self.skipped_frames + 1)):
                self.skipped_frames += 1
            else:
                self.c_T_w = T @ self.c_T_w
                self.skipped_frames = 0

    def current_pose(self):
        return np.linalg.inv(self.c_T_w)
