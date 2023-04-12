from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class AbstractCamera(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_im3d(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass
