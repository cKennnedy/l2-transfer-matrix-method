from dataclasses import dataclass
from typing import Callable
from transfer_matrix_method import RefractiveIndex
import numpy as np

@dataclass
class Layer:
    thickness: float
    refractive_index: RefractiveIndex

    @property
    def D(self) -> Callable[[float], np.ndarray]:
        return lambda wavelength: (1/self.thickness)* np.array([[1, self.refractive_index[wavelength]["n"]],[self.refractive_index[wavelength]["n"],1]])


def calculate_reflectance(layers: list[Layer]) -> float:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        float: reflectance of multi-layer film
    """
    pass