from typing import Any, Callable, Optional
from keras.callbacks import History
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate, Dropout
from keras.models import Model
from tensorflow import keras
import tensorflow as tf
import numpy as np

import json

from .serialisable_model import SerialisableModel, ModelStateException

class ReverseTMMModel(SerialisableModel):
    material_label_columns = ["First Layer", "Second Layer"]
    material_library = dict()

    def __init__(self, retrain: bool = False, serialised_model_path: str | None = ""):
        super().__init__(retrain, serialised_model_path)
        if self.model:
            try:
                self.material_library = self.load_library(f"{self.serialised_model_path}/materials.json")
            except FileNotFoundError as e:
                raise ModelStateException("Model loaded from disk, but no materials library found")

    def _create_model_factory(self) -> tuple[Callable[[dict[str, Any]], Model], dict[str, Callable[..., Any]]]:
        def model_factory(parameters):
            num_wavelengths = parameters["num_wavelengths"]
            num_materials = parameters["num_materials"]
            i = Input(num_wavelengths)

            dl1 = Dense(num_wavelengths, activation="PReLU", input_shape=(num_wavelengths,))(i)

            dl2 = Dense(512, activation="PReLU")(dl1)
            dl2 = Dropout(0.2)(dl2)  # Adding dropout layer with a dropout rate of 0.5

            dl3 = Dense(256, activation="PReLU")(dl2)
            dl3 = Dropout(0.2)(dl3)

            dl4 = Dense(256, activation="PReLU")(dl3)
            dl4 = Dropout(0.2)(dl4)

            out1 = Dense(num_materials, "softmax", name="first_layer")(dl4)
            out2 = Dense(num_materials, "softmax", name="second_layer")(dl4)

            join = Concatenate()([dl4, out1, out2])

            dl5 = Dense(256, activation="PReLU")(join)
            dl5 = Dropout(0.2)(dl5)

            dl6 = Dense(256, activation="PReLU")(dl5)
            dl6 = Dropout(0.2)(dl6)

            t1 = Dense(6, "relu", name="thicknesses")(dl6)


            return Model(inputs=i, outputs=[out1,out2,t1])
        return (
            model_factory,
            {
                "num_materials": lambda features, labels: 32,
                "num_wavelengths": lambda features, labels: len(features.columns)
            }
        )
    
    def compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss={
                "first_layer": keras.losses.SparseCategoricalCrossentropy(),
                "second_layer": keras.losses.SparseCategoricalCrossentropy(),
                "thicknesses": "mean_squared_error",
            
            },
            loss_weights=[1,1,0.01],
            metrics={
                "first_layer": tf.metrics.SparseCategoricalAccuracy("First_Layer_Accuracy"),
                "second_layer": tf.metrics.SparseCategoricalAccuracy("Second_Layer_Accuracy"),
                "thicknesses": tf.keras.metrics.MeanSquaredError("Thickness_MSE"),
            
            }
        )

    def load_library(self, fp: str):
        with open(fp) as f:
            return json.load(f) 

    def save_library(self, fp: str):
        with open(fp, "w") as f: 
            json.dump(self.material_library, f)

    def material_enc(self, material: str):
        return self.material_library[material]

    def _reverse_material(self, enc: str):
        return {v:k for k,v in self.material_library.items()}[enc]

    def _train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float, epochs: float) -> History:
        unique_training_materials = np.unique(labels[["First Layer", "Second Layer"]].values)
        for material in unique_training_materials:
            if not self.material_library:
                self.material_library[material] = 0
            elif material not in self.material_library:
                self.material_library[material] = max(self.material_library.values()) + 1

        return self.model.fit(
            features,
            {
                "first_layer": labels["First Layer"].apply(self.material_enc),
                "second_layer": labels["Second Layer"].apply(self.material_enc),
                "thicknesses": labels[["d1","d2","d3","d4","d5","d6"]],
            },
            epochs=epochs,
            validation_split=validation_split
        )
        

    def _predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.material_library:
            try:
                self.material_library = self.load_library(f"{self.serialised_model_path}/materials.json")
            except FileNotFoundError as e:
                raise ModelStateException("Model trained but no material encoding exists")
    
        res_array = self.model.predict(features)
        cols = pd.Series(["d1", "d2", "d3", "d4", "d5", "d6"])
        derive_material = lambda arr: self._reverse_material(np.argmax(arr))
        transform_materials = lambda arr: np.array([[derive_material(wgts)] for wgts in arr])
        res_df = pd.DataFrame(
            res_array[2],
            columns=cols,
            index=features.index
        )
        res_df["First Layer"] = transform_materials(res_array[0])
        res_df["Second Layer"] = transform_materials(res_array[1])
        return res_df

    def _evaluate(self, features: pd.DataFrame, labels: pd.DataFrame):
        return self.model.evaluate(
            features,
            {
                "first_layer": labels["First Layer"].apply(self.material_enc),
                "second_layer": labels["Second Layer"].apply(self.material_enc),
                "thicknesses": labels[["d1","d2","d3","d4","d5","d6"]],
            }
        )
    
    def save(self, fp: str | None = None):
        ret = super().save(fp)
            
        self.save_library(f"{self.serialised_model_path}/materials.json")
        return ret
