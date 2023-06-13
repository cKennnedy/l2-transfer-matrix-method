from dataclasses import dataclass
from typing import Callable
from .refractive_index import RefractiveIndex
from math import e
import numpy as np
import tmm.tmm_core as tmm

@dataclass
class Layer:
    thickness: float
    refractive_index: RefractiveIndex

    def r(self, other: "Layer") -> Callable[[float], float]:
        return lambda wavelength: (
            abs(
                (self.refractive_index[wavelength]["n"] - other.refractive_index[wavelength]["n"])/
                (self.refractive_index[wavelength]["n"] + other.refractive_index[wavelength]["n"])
            )
        )
    
    def t(self, other: "Layer") -> Callable[[float], float]:
        return lambda wavelength: (
            2*self.refractive_index[wavelength]["n"] / (self.refractive_index[wavelength]["n"] + other.refractive_index[wavelength]["n"])
        )

    def D(self, other: "Layer") -> Callable[[float], np.ndarray]:
        return lambda wavelength: (1/self.t(other)(wavelength)) * np.array([
            [1, self.r(other)(wavelength)],
            [self.r(other)(wavelength), 1]
        ])
    
    @property
    def P(self) -> Callable[[float], np.ndarray]:
        return lambda wavelength: np.array([
            [e**(self.refractive_index[wavelength]["k"]*self.thickness*1j), 0],
            [0, e**-(self.refractive_index[wavelength]["k"]*self.thickness*1j)]
        ])



def reflectance(layers: list[Layer], wavelength: float) -> tuple[float,float]:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        tuple[float,float]: (reflectance, transmission) of multi-layer film
    """

    air = Layer(np.inf, RefractiveIndex({0: {"n":1, "k":0}, 1000: {"n": 1, "k":0}}))
    augmented_layers = [air, *layers, air]
    
    n_list = [l.refractive_index[wavelength]["n"]+l.refractive_index[wavelength]["k"]*1j for l in augmented_layers]
    d_list = [l.thickness for l in augmented_layers]

    return tmm.coh_tmm("s", n_list=n_list, d_list=d_list, th_0=0, lam_vac=wavelength)
    
