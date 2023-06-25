from typing import Any, Callable, Optional
from keras.callbacks import History
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.metrics import SparseCategoricalAccuracy, MeanSquaredError as MeanSquaredErrorMetric
from keras.losses import SparseCategoricalCrossentropy, MeanSquaredError as MeanSquaredErrorLoss
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

            i = Input(num_wavelengths,)

            dl1 = Dense(num_wavelengths, activation="PReLU", input_shape=(num_wavelengths,))(i)

            dl2 = Dense(512, activation="PReLU")(dl1)
            dl3 = Dense(256, activation="PReLU")(dl2)
            dl4 = Dense(256, activation="PReLU")(dl3)
            
            out1 = Dense(num_materials, "softmax", name="first_layer")(dl4)
            out2 = Dense(num_materials, "softmax", name="second_layer")(dl4)
            
            dl5 = Dense(256, activation="PReLU")(dl4)
            dl6 = Dense(256, activation="PReLU")(dl5)
            t1 = Dense(6, "relu", name="thicknesses")(dl6)

            return Model(inputs=i, outputs=[out1,out2,t1])
        return (
            model_factory,
            {
                "num_materials": lambda features, labels: len(np.unique(labels[self.material_label_columns].values)),
                "num_wavelengths": lambda features, labels: len(features.columns)
            }
        )
    
    def compile(self) -> None:
        self.model.compile(
            optimizer="adam",
            loss={
                "first_layer": SparseCategoricalCrossentropy(),
                "second_layer": SparseCategoricalCrossentropy(),
                "thicknesses": MeanSquaredErrorLoss(),
            
            },
            loss_weights={
                "first_layer": 1,
                "second_layer": 1,
                "thicknesses": 0.01
            },
            metrics={
                "first_layer": SparseCategoricalAccuracy("accuracy"),
                "second_layer": SparseCategoricalAccuracy("accuracy"),
                "thicknesses": MeanSquaredErrorMetric("MSE"),
            
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
        unique_training_materials = np.unique(labels[self.material_label_columns].values)
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
        

    def _predict(self, features: pd.DataFrame, allowed_materials: Optional[list[str]] = None) -> pd.DataFrame:
        if not self.material_library:
            try:
                self.material_library = self.load_library(f"{self.serialised_model_path}/materials.json")
            except FileNotFoundError as e:
                raise ModelStateException("Model trained but no material encoding exists")
            
        if allowed_materials:
            for material in allowed_materials:
                if material not in self.material_library:
                    raise ValueError(f"Material: {material} not in model material library")
            allowed_indices = [self.material_enc(material) for material in allowed_materials]
            
    
        res_array = self.model.predict(features)
        cols = pd.Series(["d1", "d2", "d3", "d4", "d5", "d6"])
        pick_from_all = np.argmax
        pick_from_allowed = lambda arr: max({i:wgt for i,wgt in enumerate(arr) if i in allowed_indices}.items(), key=lambda tup: tup[1])[0]
        pick_function = pick_from_allowed if allowed_materials else pick_from_all
        derive_material = lambda arr: self._reverse_material(pick_function(arr))
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
