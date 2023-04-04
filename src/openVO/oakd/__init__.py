from typing import Tuple

import numpy as np

from .camera import OAK_Camera
from .odometer import OAK_Odometer

__all__ = ["OAK_Camera", "OAK_Odometer"]


# function which takes all parameters from camera and odometer and returns a single object
def create(**kwargs) -> Tuple[OAK_Camera, OAK_Odometer]:
    cam = OAK_Camera(
        **kwargs,
    )
    odom = OAK_Odometer(
        **kwargs,
    )
    return cam, odom
