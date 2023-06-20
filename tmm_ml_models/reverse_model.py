from keras.callbacks import History
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, Concatenate
from keras.models import Model
import keras
import tensorflow as tf

from .serialisable_model import SerialisableModel

class ReverseTMMModel(SerialisableModel):
    def _model(self) -> Model:
        num_wavelengths = 351
        num_materials = 32
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
    
    def _train(self, features: pd.DataFrame, labels: pd.DataFrame, validation_split: float, epochs: float) -> History:
        