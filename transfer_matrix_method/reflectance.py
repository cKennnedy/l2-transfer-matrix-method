from dataclasses import dataclass
from typing import Callable

@dataclass
class Layer:
    thickness: float
    refractive_index: Callable

def calculate_reflectance(layers: list[Layer]) -> float:
    """Calculate the reflectance of a list of thin layers

    Args:
        layers (list[Layer]): list of layer objects

    Returns:
        float: reflectance of multi-layer film
    """
    pass