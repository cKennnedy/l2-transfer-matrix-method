import pandas as pd
import numpy as np
import tensorflow as tf

from keras.layers import Input, Embedding, Dense, Concatenate, Flatten, TextVectorization
from keras.models import Model

from typing import Iterable, Callable, Any
from .serialisable_model import SerialisableModel


class ForwardTMMModel(SerialisableModel):
    is_trained: bool = False
    _material_feature_cols = ["First Layer", "Second Layer"]
    
    def _create_model_factory(self) -> tuple[Callable[[dict[str, Any]], Model], dict[str, Callable]]:
        def model_factory(parameters):
            thicknesses = Input(shape=(parameters["num_thicknesses"],))
            materials = Input(shape=(parameters["num_materials"],),dtype=tf.string)
            vocab_size = 300

            vectorize_layer = TextVectorization(
                max_tokens=vocab_size,
                output_sequence_length=1,
                name="string_vec"
            )

            vec_materials = [
                vectorize_layer(string) for string in tf.transpose(materials)
            ]

            v_materials = Concatenate()(vec_materials)

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
        return (
            model_factory,
            {
                "num_materials": lambda features, labels: 2,
                "num_thicknesses": lambda features, labels: 6
            }
        )

    def compile(self) -> None:
        self.model.compile("adam", loss="mean_squared_error")
    
    def _split_features_to_inputs(self, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        return (
            features[[c for c in features.columns if c not in self._material_feature_cols]],
            features[self._material_feature_cols]
        )
        
    def _train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float = 0.1, epochs: int = 100):
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

        return history
    
    def _predict(self, features: pd.DataFrame) -> pd.DataFrame:
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