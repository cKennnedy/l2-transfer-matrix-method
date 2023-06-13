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



def reflectance(layers: list[Layer]) -> tuple[float,float]:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        tuple[float,float]: (reflectance, transmission) of multi-layer film
    """

    air = Layer(1000, RefractiveIndex({0: {"n":1, "k":0}, 1000: {"n": 1, "k":1}}))
    augmented_layers = [air, *layers, air]
    
    matrices = [augmented_layers[0].D[augmented_layers[1]]]
    for i in range(1, len(augmented_layers) - 1):
        matrices.extend([
            augmented_layers[i].P,
            augmented_layers[i].D(augmented_layers[i+1])
        ])

    M = matrices[0]
    for matrix in matrices[1:]:
        M = np.matmul(M, matrix)

    return M[1,0]/M[0,0], M[0,0]
