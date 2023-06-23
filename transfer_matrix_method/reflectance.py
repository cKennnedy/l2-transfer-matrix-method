from dataclasses import dataclass
from typing import Callable
from .refractive_index import RefractiveIndex
from math import e
import numpy as np

@dataclass
class Layer:
    thickness: float
    refractive_index: RefractiveIndex

    def r(self, other: "Layer") -> Callable[[float], float]:
        return lambda wavelength: (
                (self.refractive_index[wavelength] - other.refractive_index[wavelength])/
                (self.refractive_index[wavelength] + other.refractive_index[wavelength])
        )
    
    def t(self, other: "Layer") -> Callable[[float], float]:
        return lambda wavelength: (
            2*self.refractive_index[wavelength] / (self.refractive_index[wavelength] + other.refractive_index[wavelength])
        )

    def D(self, other: "Layer") -> Callable[[float], np.ndarray]:
        return lambda wavelength: (1/self.t(other)(wavelength)) * np.array([
            [1, self.r(other)(wavelength)],
            [self.r(other)(wavelength), 1]
        ])
    
    @property
    def P(self) -> Callable[[float], np.ndarray]:
        wavenumber = lambda wavelength: (2*np.pi) / wavelength

        coeff = lambda wavelength: self.refractive_index[wavelength] * self.thickness*1j* wavenumber(wavelength)
        return lambda wavelength: np.array([
            [np.exp(-coeff(wavelength)), 0],
            [0, np.e**(coeff(wavelength))]
        ])



def reflectance(layers: list[Layer], wavelength: float, substrate: Layer | None = None) -> tuple[float,float]:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        tuple[float,float]: (reflectance, transmission) of multi-layer film
    """

    air = Layer(1000, RefractiveIndex({0: {"n":1, "k":0}, 1000: {"n": 1, "k":0}}))
    augmented_layers = [air, *layers, substrate or air]
    
    matrices = [augmented_layers[0].D(augmented_layers[1])(wavelength)]
    for i in range(1, len(augmented_layers) - 1):
        matrices.extend([
            augmented_layers[i].P(wavelength),
            augmented_layers[i].D(augmented_layers[i+1])(wavelength)
        ])

    M = matrices[0]
    for matrix in matrices[1:]:
        M = np.matmul(M, matrix)

    return {"reflectance": abs((M[1,0]/M[0,0]))**2, "transmittance": abs(1/M[0,0])**2}
