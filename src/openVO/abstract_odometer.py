from abc import ABC
from typing import Any

import cv2
import numpy as np

from .abstract_camera import AbstractCamera


class AbstractOdometer(ABC):
    def __init__(
        self,
        stereo_camera: AbstractCamera,
        nfeatures=500,
        match_threshold=0.8,
        rigidity_threshold=0,
        outlier_threshold=0,
        min_matches=10,
        min_valid_disparity=4,
        max_valid_disparity=100,
        max_distance_change=1,
        max_rotation_change=np.pi / 3,
    ):
        """
        Parameters:
            stereo_camera: AbstractCamera object
            nfeatures: number of features to detect
            match_threshold: threshold for matching features
            rigidity_threshold: threshold for feature rigidity
            outlier_threshold: threshold for outlier rejection
            preprocessed_frames: whether to preprocess the frames
            min_matches: minimum number of matches to compute a transformation
            min_valid_disparity: minimum disparity for valid depth
            max_valid_disparity: maximum disparity for valid depth
            max_distance_change: maximum distance change for valid transformation
            max_rotation_change: maximum rotation change for valid transformation
        """
        self._stereo = stereo_camera
        # image data for current and previous frames
        self._current_img, self._current_disparity, self._current_3d = None, None, None
        self._prev_img, self._prev_disparity, self._prev_3d = None, None, None
        # orb feature detector and matcher
        # TODO crosscheck
        self._orb, self._matcher = cv2.ORB_create(
            nfeatures=nfeatures
        ), cv2.BFMatcher.create(cv2.NORM_HAMMING)
        # orb key points and descriptors for current and previous frames
        self._prev_kps, self._current_kps = None, None
        self._current_kps, self._current_desc = None, None
        self._match_threshold, self._rigidity_threshold = (
            match_threshold,
            rigidity_threshold,
        )
        self._outlier_threshold = outlier_threshold
        self._min_matches = min_matches
        self._min_valid_disparity = min_valid_disparity
        self._max_valid_disparity = max_valid_disparity
        self._max_distance_change = max_distance_change
        self._max_rotation_change = max_rotation_change

        # Number of successive frames with no coordinate transformation found
        self._skipped_frames = 0
        # transformation of the world frame in the camera's coordinate system
        self._c_T_w = np.eye(4)
        self._c_T_w_prev = np.eye(4)

        self._skip_cause = ""

    # image mask for pixels with acceptable disparity values
    def feature_mask(self, disparity):
        # TODO config
        mask = (disparity >= self._min_valid_disparity) * (
            disparity <= self._max_valid_disparity
        )
        return mask.astype(np.uint8) * 255

    def valid_distance_change(self, prev_kp_idx, current_kp_idx):
        p_x, p_y = self._prev_kps[prev_kp_idx].pt
        c_x, c_y = self._current_kps[current_kp_idx].pt
        # TODO if we use this function
        return np.linalg.norm(self._prev_3d[int(p_y)][int(p_x)]) - np.linalg.norm(
            self._current_3d[int(c_y)][int(c_x)]
        ) <= self._max_distance_change * (self._skipped_frames + 1)

    def bilinear_interpolate_pixels(self, img, x, y):
        floor_x, floor_y = int(x), int(y)
        p10, p01, p11 = None, None, None
        p00 = img[floor_y, floor_x]
        h, w = img.shape[0:2]
        if floor_x + 1 < w:
            p10 = img[floor_y, floor_x + 1]
            if floor_y + 1 < h:
                p11 = img[floor_y + 1, floor_x + 1]
        if floor_y + 1 < h:
            p01 = img[floor_y + 1, floor_x]
        r_x, r_y, num, den = x - floor_x, y - floor_y, 0, 0

        if not np.isinf(p00).any():
            num += (1 - r_x) * (1 - r_y) * p00
            den += (1 - r_x) * (1 - r_y)
            # return p00
        if not (p01 is None or np.isinf(p01).any()):
            num += (1 - r_x) * (r_y) * p01
            den += (1 - r_x) * (r_y)
            # return p01
        if not (p10 is None or np.isinf(p10).any()):
            num += (r_x) * (1 - r_y) * p10
            den += (r_x) * (1 - r_y)
            # return p10
        if not (p11 is None or np.isinf(p11).any()):
            num += r_x * r_y * p11
            den += r_x * r_y
            # return p11
        return num / den

    # Paper alg
    def rigid_body_filter(self, prev_pts, pts):
        # d1-d2 where columns of d1 = pts and rows of d2 = pts
        # result is matrix with entry [i, j] = pts[i] - pts[j]
        dists = np.tile(pts, (len(pts), 1, 1)).transpose((1, 0, 2)) - np.tile(
            pts, (len(pts), 1, 1)
        )
        prev_dists = np.tile(prev_pts, (len(pts), 1, 1)).transpose((1, 0, 2)) - np.tile(
            prev_pts, (len(pts), 1, 1)
        )
        delta_dist = np.abs(
            np.linalg.norm(dists, axis=2) - np.linalg.norm(prev_dists, axis=2)
        )
        consistency = (np.abs(delta_dist) < self._rigidity_threshold).astype(int)
        clique = np.zeros(len(pts), int)
        num_consistent = np.sum(consistency, axis=0)
        max_consistent = np.argmax(num_consistent)
        clique[max_consistent] = 1
        clique_size = 1
        compatible = consistency[max_consistent]
        for _ in range(len(pts)):
            candidates = (compatible - clique).astype(int)
            if np.sum(candidates) == 0:
                break
            selected = np.argmax(num_consistent * candidates)
            clique[selected] = 1
            clique_size += 1
            # leniency = 1 if clique_size > 4 else 0
            leniency = 0
            compatible = (consistency @ clique >= sum(clique) - leniency).astype(int)
        return clique

    def save_frame_update(self, next_img, next_disp, next_3d, next_kps, next_desc):
        self._prev_img = self._current_img
        self._prev_disparity = self._current_disparity
        self._prev_3d = self._current_3d
        self._current_img, self._current_disparity, self._current_3d = (
            next_img,
            next_disp,
            next_3d,
        )
        self._prev_kps, self._prev_desc = self._current_kps, self._current_desc
        self._current_kps, self._current_desc = next_kps, next_desc

    def update(self, img_left, img_right):
        next_3d, next_disp, next_img = self._stereo.compute_3d(
            img_left, img_right, preprocessed=self._preprocessed_frames
        )
        next_kps, next_desc = self._orb.detectAndCompute(
            next_img, self._feature_mask(next_disp)
        )

        if len(next_kps) < self._min_matches:
            self._skipped_frames += 1
            self._skip_cause = "keypoints"
            return False

        if self._current_img is None:
            self._save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)
            return True

        T = None
        current_pts, next_pts = self._point_clouds(
            self._current_kps,
            next_kps,
            self._current_desc,
            next_desc,
            self._current_3d,
            next_3d,
        )

        if current_pts is None:
            self._skip_cause = "matches"
        else:
            T = self._point_cloud_transform(current_pts, next_pts)
            if not (T is None):
                self._c_T_w_prev = self._c_T_w
                self._c_T_w = T @ self._c_T_w
        if T is None and not (self._prev_img is None):
            prev_pts, next_pts = self._point_clouds(
                self._prev_kps,
                next_kps,
                self._prev_desc,
                next_desc,
                self._prev_3d,
                next_3d,
            )
            if prev_pts is None:
                self._skip_cause = "matches"
            else:
                T = self._point_cloud_transform(prev_pts, next_pts)
                if not (T is None):
                    T_prev = self._c_T_w_prev
                    self._c_T_w_prev = self._c_T_w
                    self._c_T_w = T @ T_prev
                    self._skipped_frames = 0

        if T is None:
            self._skipped_frames += 1
            # self._save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)
            return False
        else:
            self._skipped_frames = 0
            self._save_frame_update(next_img, next_disp, next_3d, next_kps, next_desc)

        return True

    def point_clouds(self, kps1, kps2, desc1, desc2, im3d1, im3d2):
        matches = self._matcher.knnMatch(desc1, desc2, k=2)
        matches = [
            m[0]
            for m in matches
            if m[0].distance < self._match_threshold * m[1].distance
        ]
        if False:  # TODO config
            matches = [
                m for m in matches if self._valid_distance_change(m.queryIdx, m.trainIdx)
            ]
        if len(matches) < self._min_matches:
            return None, None

        pts1 = [kps1[m.queryIdx].pt for m in matches]
        pts2 = [kps2[m.trainIdx].pt for m in matches]
        for i in range(len(matches)):
            pts1[i] = self._bilinear_interpolate_pixels(im3d1, pts1[i][0], pts1[i][1])
            pts2[i] = self._bilinear_interpolate_pixels(im3d2, pts2[i][0], pts2[i][1])
        return np.array(pts1), np.array(pts2)

    def point_cloud_transform(self, current_pts, next_pts):
        if self._rigidity_threshold > 0:
            inlier_mask = self._rigid_body_filter(current_pts, next_pts)
            current_pts = current_pts[inlier_mask > 0]
            next_pts = next_pts[inlier_mask > 0]

        rigidity_cause = False
        if len(current_pts) < 10:
            rigidity_cause = True
            self._skip_cause = "rigidity"

        # Single-pass outlier removal using error from estimated transformation
        if self._outlier_threshold > 0 and len(current_pts) >= 10:
            T, _ = cv2.estimateAffine3D(current_pts, next_pts, force_rotation=True)
            T = np.vstack([T, [0, 0, 0, 1]])
            h_pts = np.hstack([next_pts, np.array([[1] * len(next_pts)]).transpose()])
            h_prev = np.hstack(
                [current_pts, np.array([[1] * len(current_pts)]).transpose()]
            )
            errors = np.array(
                [
                    np.linalg.norm(h_pts[i] - T @ h_prev[i]) / np.linalg.norm(h_pts[i])
                    for i in range(len(h_pts))
                ]
            )
            threshold = self._outlier_threshold + np.median(
                errors
            )  # TODO better outlier detection?
            current_pts = current_pts[errors < threshold]
            next_pts = next_pts[errors < threshold]

        if len(current_pts) < self._min_matches:
            if not rigidity_cause:
                self._skip_cause = "outlier"
            return

        T, _ = cv2.estimateAffine3D(current_pts, next_pts, force_rotation=True)
        T = np.vstack([T, [0, 0, 0, 1]])

        if np.isnan(T).any():
            self._skip_cause = "nan"
            return
        else:
            disp = T[0:3, 3]
            rot, _ = cv2.Rodrigues(T[0:3, 0:3])

            # TODO config
            if np.linalg.norm(disp) > self._max_distance_change * (
                self._skipped_frames + 1
            ) or np.linalg.norm(rot) > self._max_rotation_change * (
                self._skipped_frames + 1
            ):
                if np.linalg.norm(disp) > self._max_distance_change * (
                    self._skipped_frames + 1
                ):
                    self._skip_cause = "bigdist"
                if np.linalg.norm(rot) > self._max_rotation_change * (
                    self._skipped_frames + 1
                ):
                    self._skip_cause = "bigrot"
                return
            else:
                return T

    def current_pose(self):
        return np.linalg.inv(self._c_T_w)
