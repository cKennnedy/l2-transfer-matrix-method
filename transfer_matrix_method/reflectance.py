from dataclasses import dataclass
from typing import Callable
from transfer_matrix_method import RefractiveIndex
from math import e
import numpy as np

@dataclass
class Layer:
    thickness: float
    refractive_index: RefractiveIndex

    def relative_index(self, other: "Layer") -> Callable[[float], float]:
        return lambda wavelength: (
            abs(
                (self.refractive_index[wavelength]["n"] - other.refractive_index[wavelength]["n"])/
                (self.refractive_index[wavelength]["n"] + other.refractive_index[wavelength]["n"])
            ) ** 2
        )

    def D(self, other: "Layer") -> Callable[[float], np.ndarray]:
        return lambda wavelength: (1/self.thickness)* np.array([
            [1, self.relative_index(other)(wavelength)],
            [self.relative_index(other)(wavelength), 1]
        ])
    
    @property
    def P(self) -> Callable[[float], np.ndarray]:
        return lambda wavelength: np.array([
            [e**(self.refractive_index[wavelength]["k"]*self.thickness*1j), 0],
            [0, e**(self.refractive_index[wavelength]["k"]*self.thickness*1j)]
        ])



def reflectance(layers: list[Layer]) -> float:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        float: reflectance of multi-layer film
    """
    matrices = [layers[0].P]
    for i in range(1, len(layers)):
        matrices.extend([
            layers[i].D(layers[i-1]),
            layers[i].P
        ])

    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.matmul(result, matrix)
