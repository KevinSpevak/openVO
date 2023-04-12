import numpy as np

def feature_mask(disparity: np.ndarray, min_disp: int, max_disp: int):
    mask = (disparity >= min_disp) * (
        disparity <= max_disp
    )
    return mask.astype(np.uint8) * 255

def bilinear_interpolate_pixels(img: np.ndarray, x: float, y: float):
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
