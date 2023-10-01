import time
import itertools
import sys

import cv2
import numpy as np

from openVO import drawPoseOnImage, rot2RPY
from openVO.oakd import OAK_Camera, OAK_Odometer


DEBUG = False
TIME_PER_TEST = 25
TRIALS_PER_TEST = 3

CAM_ARGS = {
    "median_filter": [None, 3, 5, 7],
    "stereo_threshold_filter_min_range": [200, 400],
    "stereo_threshold_filter_max_range": [15000, 20000],
}

ODOM_ARGS = {
    "nfeatures": [250, 500]
}

def get_args(arg_option_dict):
    keys, values = zip(*arg_option_dict.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return experiments

def run_test(cam_args, odom_args, expected=(0, 0, 0, 0, 0, 0)):
    if DEBUG:
        print(f"Testing with args:\n {cam_args}\n {odom_args}\n {expected}\n")

    cam = OAK_Camera(**cam_args)
    odom = OAK_Odometer(cam, **odom_args)
    cam.start(block=True)

    pose = None

    stopped = False

    start_time = time.perf_counter()
    while True:
        odom.update()
        rgb_frame = cam.rgb
        rgb_frame = cv2.resize(rgb_frame, (640, 480))
        pose = odom.current_pose()
        drawPoseOnImage(pose, rgb_frame)
        cv2.imshow("Annotated", rgb_frame)
        # quit if the user presses q
        if cv2.waitKey(100) & 0xFF == ord("q"):
            stopped = True
            break

        if time.perf_counter() - start_time >= TIME_PER_TEST:
            break
    
    cam.stop()

    if stopped:
        sys.exit(-1)

    roll, pitch, yaw = rot2RPY(pose)
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]

    # compute the magnitude error on rotation and translation
    rot_error = sum((roll - expected[0], pitch - expected[1], yaw - expected[2]))

    # for translational error compute the L2 norm between the expected and actual
    # translation vectors
    expected_translation = np.array(expected[3:])
    actual_translation = np.array([x, y, z])

    # compute the L2 norm
    trans_error = np.linalg.norm(expected_translation - actual_translation)

    return trans_error

def run_tests(cam_args, odom_args):
    results = {}
    for odom_arg in odom_args:
        for cam_arg in cam_args:
            test_results = []
            # print info
            print("Testing with:")
            print(" Camera Args:")
            print(f"    {cam_arg}")
            print(" Odometer Args:")
            print(f"    {odom_arg}")

            for i in range(TRIALS_PER_TEST):
                print(f" Trial: {i}")
                test_results.append(run_test(cam_arg, odom_arg))

            hashable_cam_arg = tuple((k,cam_arg[k]) for k in sorted(cam_arg))
            hashable_odom_arg = tuple((k,odom_arg[k]) for k in sorted(odom_arg))
            results[(hashable_cam_arg, hashable_odom_arg)] = np.mean(test_results)

    return results

def parse_results(results):
    # print out the key with the smallest value
    best_key = min(results, key=results.get)
    print(f"Best parameters: {best_key}, with error: {results[best_key]}")
    
def main():
    cam_args = get_args(CAM_ARGS)
    odom_args = get_args(ODOM_ARGS)
    results = run_tests(cam_args, odom_args)
    parse_results(results)

if __name__ == "__main__":
    main()
