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
    MAX_ROTATION_CHANGE = np.pi/3 # Radians

    def __init__(self, stereo_camera, nfeatures=500, match_threshold=0.8, rigidity_threshold=0, \
                 outlier_threshold=0, preprocessed_frames=False):
        self.stereo = stereo_camera
        # image data for current and previous frames
        self.current_img, self.current_disparity, self.current_3d = None, None, None
        self.prev_img, self.prev_disparity, self.prev_3d = None, None, None
        # orb feature detector and matcher
        # TODO crosscheck
        # TODO config # features 
        self.orb, self.matcher = cv2.ORB_create(nfeatures=nfeatures), cv2.BFMatcher.create(cv2.NORM_HAMMING)
        # orb key points and descriptors for current and previous frames
        self.prev_kps, self.current_kps = None, None
        self.current_kps, self.current_desc = None, None
        self.match_threshold, self.rigidity_threshold = match_threshold, rigidity_threshold
        self.outlier_threshold, self.preprocessed_frames = outlier_threshold, preprocessed_frames
        # Number of successive frames with no coordinate transformation found
        self.skipped_frames = 0
        # transformation of the world frame in the camera's coordinate system
        self.c_T_w = np.eye(4)

        # TODO
        self.skip_cause = ""

    # image mask for pixels with acceptable disparity values
    def feature_mask(self, disparity):
        # TODO config  
        mask = (disparity >= self.MIN_VALID_DISPARITY) * (disparity <= self.MAX_VALID_DISPARITY)
        return mask.astype(np.uint8)*255

    def valid_distance_change(self, prev_kp_idx, current_kp_idx):
        p_x, p_y = self.prev_kps[prev_kp_idx].pt
        c_x, c_y = self.current_kps[current_kp_idx].pt
        return (np.linalg.norm(self.prev_3d[int(p_y)][int(p_x)])
                - np.linalg.norm(self.current_3d[int(c_y)][int(c_x)]) <= self.MAX_DISTANCE_CHANGE * (self.skipped_frames + 1))

    # Paper alg
    def rigid_body_filter(self, prev_pts, pts):
        # d1-d2 where columns of d1 = pts and rows of d2 = pts
        # result is matrix with entry [i, j] = pts[i] - pts[j]
        dists = np.tile(pts, (len(pts), 1, 1)).transpose((1, 0, 2)) - np.tile(pts, (len(pts), 1, 1))
        prev_dists = np.tile(prev_pts, (len(pts), 1, 1)).transpose((1, 0, 2)) - np.tile(prev_pts, (len(pts), 1, 1))
        delta_dist = np.abs(np.linalg.norm(dists, axis=2) - np.linalg.norm(prev_dists, axis=2))
        consistency = (np.abs(delta_dist) < self.rigidity_threshold).astype(int)
        clique = np.zeros(len(pts), int)
        num_consistent = np.sum(consistency, axis=0)
        max_consistent = np.argmax(num_consistent)
        clique[max_consistent] = 1
        clique_size = 1
        compatible = consistency[max_consistent]
        for _ in range(len(pts)):
            candidates = (compatible - clique).astype(int)
            if (np.sum(candidates) == 0):
                break
            selected = np.argmax(num_consistent * candidates)
            clique[selected] = 1
            clique_size += 1
            #leniency = 1 if clique_size > 4 else 0
            leniency = 0
            compatible = (consistency @ clique >= sum(clique) - leniency).astype(int)
        return clique

    def save_frame_update(self, next_img, next_disp, next_3d, next_kps, next_desc):
        self.prev_img = self.current_img
        self.prev_disparity = self.current_disparity
        self.prev_3d = self.current_3d
        self.current_img, self.current_disparity, self.current_3d = next_img, next_disp, next_3d
        self.prev_kps, self.prev_desc = self.current_kps, self.current_desc
        self.current_kps, self.current_desc = next_kps, next_desc

    def update(self, img_left, img_right):
        next_3d, next_disp, next_img = self.stereo.compute_3d(img_left, img_right, preprocessed=self.preprocessed_frames)
        next_kps, next_desc = self.orb.detectAndCompute(next_img, self.feature_mask(next_disp))

        # TODO config for this
        if len(next_kps) < 10:
            self.skipped_frames+= 1
            self.skip_cause = "keypoints"
            return False

        if not self.current_img is None:
            matches = self.matcher.knnMatch(self.current_desc, next_desc, k=2)

            matches = [m[0] for m in matches if m[0].distance < self.match_threshold * m[1].distance]
            if (False): # TODO config
                matches = [m for m in matches if self.valid_distance_change(m.queryIdx, m.trainIdx)]

            # TODO config
            if len(matches) < 10:
                self.skipped_frames += 1
                #self.save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)
                self.skip_cause = "matches"
                return False

            # TODO can we get subpix 3d values??
            next_pts = np.array([next_3d[int(y)][int(x)] for x, y in [next_kps[m.trainIdx].pt for m in matches]])
            current_pts = np.array([self.current_3d[int(y)][int(x)] for x, y in [self.current_kps[m.queryIdx].pt for m in matches]])
            T = self.point_cloud_transform(current_pts, next_pts)
            if T is None:
                self.skipped_frames += 1
                #self.save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)
                return False
            else:
                self.c_T_w = T @ self.c_T_w
                self.skipped_frames = 0
        self.save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)
        return True

    def point_cloud_transform(self, current_pts, next_pts):
        if (self.rigidity_threshold > 0):
            inlier_mask = self.rigid_body_filter(current_pts, next_pts)
            current_pts = current_pts[inlier_mask>0]
            next_pts = next_pts[inlier_mask>0]

        rigidity_cause = False
        if (len(current_pts) < 10):
            rigidity_cause = True
            self.skip_cause = "rigidity"

        # Single-pass outlier removal using error from estimated transformation
        if self.outlier_threshold > 0 and len(current_pts) >= 10:
            T, _ = cv2.estimateAffine3D(current_pts, next_pts, force_rotation=True)
            T = np.vstack([T, [0,0,0,1]])
            h_pts = np.hstack([next_pts, np.array([[1]*len(next_pts)]).transpose()])
            h_prev = np.hstack([current_pts, np.array([[1]*len(current_pts)]).transpose()])
            errors = np.array([np.linalg.norm(h_pts[i] - T @ h_prev[i])/np.linalg.norm(h_pts[i]) for i in range(len(h_pts))])
            threshold = self.outlier_threshold + np.median(errors) # TODO better outlier detection?
            current_pts = current_pts[errors < threshold]
            next_pts = next_pts[errors < threshold]

        # TODO config
        if len(current_pts) < 10:
            if not rigidity_cause:
                self.skip_cause = "outlier"
            return

        T, _ = cv2.estimateAffine3D(current_pts, next_pts, force_rotation=True)
        T = np.vstack([T, [0,0,0,1]])

        if np.isnan(T).any():
            self.skip_cause = "nan"
            return
        else:
            disp = T[0:3, 3]
            rot,_ = cv2.Rodrigues(T[0:3, 0:3])

            # TODO config
            if (np.linalg.norm(disp) > self.MAX_DISTANCE_CHANGE  * (self.skipped_frames + 1)
                or np.linalg.norm(rot) > self.MAX_ROTATION_CHANGE  * (self.skipped_frames + 1)):
                if np.linalg.norm(disp) > self.MAX_DISTANCE_CHANGE  * (self.skipped_frames + 1):
                    self.skip_cause = "bigdist"
                if np.linalg.norm(rot) > self.MAX_ROTATION_CHANGE  * (self.skipped_frames + 1):
                    self.skip_cause = "bigrot"
                return
            else:
                return T

    def current_pose(self):
        return np.linalg.inv(self.c_T_w)
