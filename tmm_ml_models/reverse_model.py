from typing import Any, Callable, Optional
from keras.callbacks import History
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import keras
import tensorflow as tf
import numpy as np

import json

from .serialisable_model import SerialisableModel, ModelStateException

class ReverseTMMModel(SerialisableModel):
    material_label_columns = ["First Layer", "Second Layer"]
    material_library = dict()


    def _create_model_factory(self) -> tuple[Callable[[dict[str, Any]], Model], dict[str, Callable[..., Any]]]:
        def model_factory(parameters):
            num_wavelengths = parameters["num_wavelengths"]
            num_materials = parameters["num_materials"]
            i = Input(num_wavelengths,)

            dl1 = Dense(num_wavelengths, activation="PReLU", input_shape=(num_wavelengths,))(i)

            m1dl2 = Dense(256, activation="PReLU")(dl1)
            m2dl2 = Dense(256, activation="PReLU")(dl1)
            tdl2 = Dense(256, activation="PReLU")(dl1)

            concatenated_input = Concatenate()([m1dl2, m2dl2, tdl2])

            common_layer = Dense(500)(concatenated_input)

            m1dl3 = Dense(128, activation="PReLU")(common_layer)
            m2dl3 = Dense(128, activation="PReLU")(common_layer)
            tdl3 = Dense(128, activation="PReLU")(common_layer)

            out1 = Dense(num_materials, "softmax", name="first_layer")(m1dl3)
            out2 = Dense(num_materials, "softmax", name="second_layer")(m2dl3)

            material1 = Concatenate()([tdl3,out1])
            material2 = Concatenate()([tdl3,out2])
            t1 = Dense(1, "relu", name="t1")(material1)
            t2 = Dense(1, "relu", name="t2")(material2)
            t3 = Dense(1, "relu", name="t3")(material1)
            t4 = Dense(1, "relu", name="t4")(material2)
            t5 = Dense(1, "relu", name="t5")(material1)
            t6 = Dense(1, "relu", name="t6")(material2)

            return Model(inputs=i, outputs=[out1,out2,t1,t2,t3,t4,t5,t6])
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
                "t1": "mean_squared_error",
                "t2": "mean_squared_error",
                "t3": "mean_squared_error",
                "t4": "mean_squared_error",
                "t5": "mean_squared_error",
                "t6": "mean_squared_error",
            },
            metrics={
                "first_layer": tf.metrics.SparseCategoricalAccuracy("acc1"),
                "second_layer": tf.metrics.SparseCategoricalAccuracy("acc2"),
                "t1": "accuracy",
                "t2": "accuracy",
                "t3": "accuracy",
                "t4": "accuracy",
                "t5": "accuracy",
                "t6": "accuracy",
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


        self.model.fit(
            features,
            {
                "first_layer": labels["First Layer"].apply(self.material_enc),
                "second_layer": labels["Second Layer"].apply(self.material_enc),
                "t1": labels["d1"],
                "t2": labels["d2"],
                "t3": labels["d3"],
                "t4": labels["d4"],
                "t5": labels["d5"],
                "t6": labels["d6"],
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
            np.hstack((*res_array[2:],)),
            columns=cols,
            index=features.index
        )
        res_df["First Layer"] = transform_materials(res_array[0])
        res_df["Second Layer"] = transform_materials(res_array[1])
        return res_df
    
    def save(self, fp: str | None = None):
        ret = super().save(fp)
        self.save_library(f"{self.serialised_model_path}/materials.json")
        return ret
