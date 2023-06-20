from abc import ABC, abstractmethod
from functools import wraps
from typing import Optional, Iterable

from keras.models import Model, load_model
from keras.callbacks import History

import os
import pandas as pd


class ModelStateException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

def require_trained(extra_message=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self: "SerialisableModel", *args, **kwargs):
            if not self.is_trained:
                raise ModelStateException("Model is not yet trained" + (f": {extra_message}" if extra_message else ""))
            return func(self, *args, **kwargs)
        return wrapper  
    return decorator

class SerialisableModel:
    is_trained: bool = False

    def __init__(
            self,
            retrain: bool = False,
            serialised_model_path: Optional[str] = "",
        ):


        model_is_saved = os.path.isfile(serialised_model_path) or os.path.isdir(serialised_model_path)
        self.serialised_model_path = serialised_model_path
        if retrain or not model_is_saved:
            self.model = self._model()
            self.is_trained = False
        else:
            self.model: Model = load_model(serialised_model_path)
            self.is_trained = True

        self.compile()
    
    @abstractmethod
    def _model(self) -> Model:
        pass
    
    @abstractmethod
    def compile(self) -> None:
        pass
    
    def train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float = 0.1, epochs: int = 100) -> History:
        history = self._train(features, labels, validation_split, epochs)
        self.is_trained = True
        return history
    
    @abstractmethod
    def _train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float, epochs: float) -> History:
        pass
    
    @require_trained()
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        return self._predict(features)
    
    @abstractmethod
    def _predict(self, features: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @require_trained()
    def save(self, fp: Optional[str] = None):
        if fp:
            self.serialised_model_path = fp
        if not self.serialised_model_path:
            raise ModelStateException("Model has no filepath to save to")
        self.model.save(self.serialised_model_path)
