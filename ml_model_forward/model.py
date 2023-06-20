import pandas as pd
import numpy as np
import tensorflow as tf
import os


from keras.layers import Input, Embedding, Dense, Concatenate, Flatten, TextVectorization
from keras.models import Model, load_model

from typing import Optional, Iterable
from functools import wraps

class ModelStateException(Exception):
    def __init__(self, message: str):
        super().__init__(message)

def require_trained(extra_message=None):
    def decorator(func):
        @wraps(func)
        def wrapper(self: "ForwardTMMModel", *args, **kwargs):
            if not self.is_trained:
                raise ModelStateException("Model is not yet trained" + (f": {extra_message}" if extra_message else ""))
            return func(self, *args, **kwargs)
        return wrapper  
    return decorator

class ForwardTMMModel:
    is_trained: bool = False
    material_feature_cols = ["First Layer", "Second Layer"]

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
            self.model = load_model(serialised_model_path)
            self.is_trained = True

        self.model.compile("adam", loss="mean_squared_error")
    
    def _model(self) -> Model:
        thicknesses = Input(shape=(6,))
        materials = Input(shape=(2,),dtype=tf.string)
        vocab_size = 300

        vectorize_layer = TextVectorization(
            max_tokens=vocab_size,
            output_sequence_length=1,
            name="string_vec"
        )

        vectorized_mat1 = vectorize_layer(materials[:,:1])
        vectorized_mat2 = vectorize_layer(materials[:,1:])

        v_materials = Concatenate()([vectorized_mat1, vectorized_mat2])

        material_embedding = Embedding(input_dim=vocab_size, output_dim=8)(v_materials)
        material_embedding = Flatten()(material_embedding)

        dense_layer1 = Dense(units=256, activation='PReLU')(thicknesses)

        dense_layer2 = Dense(units=128, activation='PReLU')(dense_layer1)

        dense_layer3 = Dense(units=64, activation='PReLU')(dense_layer2)

        concatenated_features = Concatenate()([dense_layer3, material_embedding])

        dense_layer4 = Dense(units=128, activation='PReLU')(concatenated_features)

        dense_layer5 = Dense(units=64, activation='PReLU')(dense_layer4)

        output_layer = Dense(units=351, activation='sigmoid')(dense_layer5)

        return Model(inputs=[thicknesses,materials], outputs=output_layer)
    
    def _split_features_to_inputs(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            features[[c for c in features.columns if c not in self.material_feature_cols]],
            features[self.material_feature_cols]
        )
        
    def train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float = 0.1, epochs: int = 100):
        thicknesses, materials = self._split_features_to_inputs(features)

        vectoriser = self.model.get_layer("string_vec")
        vectoriser.adapt(np.unique(materials.values).T)
        history = self.model.fit(
            [
                thicknesses,
                materials,
            ],
            labels,
            epochs=epochs,
            validation_split=validation_split
        )

        self.is_trained = True
        return history
    
    @require_trained()  
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        inputs = self._split_features_to_inputs(features)
        res_array = self.model.predict(inputs)
        cols = pd.Series(np.arange(400,751))
        return pd.DataFrame(res_array, columns=cols)
    
    def predict_one(self, thicknesses: Iterable, materials: tuple[str, str]) -> pd.DataFrame:
        return self.predict(
            pd.DataFrame(
                [[*thicknesses, *materials]],
                columns=["d1","d2","d3","d4","d5","d6","First Layer","Second Layer"]
            )
        )
    
    @require_trained()
    def save(self, fp: Optional[str] = None):
        if fp:
            self.serialised_model_path = fp
        if not self.serialised_model_path:
            raise ModelStateException("Model has no filepath to save to")
        self.model.save(self.serialised_model_path)
