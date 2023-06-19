import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Embedding, Dense, Concatenate, Flatten, Dropout,BatchNormalization
from keras.models import Model
import numpy as np

from typing import Optional
import os

class ForwardTMMModel:
    @property
    def _model(self):
        input_layer = Input(shape=(8,))

        input_materials = input_layer[:, 6:8]
        material_embedding = Embedding(input_dim=num_materials, output_dim=8)(input_materials)
        material_embedding = Flatten()(material_embedding)

        dense_layer1 = Dense(units=256, activation='PReLU')(input_layer[:,:6])
        dense_layer1 = BatchNormalization()(dense_layer1)  # Apply batch normalization

        dense_layer2 = Dense(units=128, activation='PReLU')(dense_layer1)
        dense_layer2 = BatchNormalization()(dense_layer2)  # Apply batch normalization

        dense_layer3 = Dense(units=64, activation='PReLU')(dense_layer2)
        dense_layer3 = BatchNormalization()(dense_layer3)  # Apply batch normalization

        concatenated_features = Concatenate()([dense_layer3, material_embedding])

        dense_layer4 = Dense(units=128, activation='PReLU')(concatenated_features)
        dense_layer4 = BatchNormalization()(dense_layer4)  # Apply batch normalization

        dense_layer5 = Dense(units=64, activation='PReLU')(dense_layer4)
        dense_layer5 = BatchNormalization()(dense_layer5)  # Apply batch normalization

        output_layer = Dense(units=num_wavelengths, activation='sigmoid')(dense_layer5)

        return Model(inputs=input_layer, outputs=output_layer)
    
    def train(self, *args, **kwargs):
        return self._model.fit(*args, **kwargs)

    def __init__(
            self,
            retrain: bool = False,
            serialised_model_path: Optional[str] = None,
            training_data: Optional[pd.DataFrame] = None,
            validation_prop: float = 0.1,
            test_prop: float = 0.1
        ):

        model_is_saved = os.path.isfile(serialised_model_path)
        if retrain or not model_is_saved:
            self._model.compile(optimizer='adam', loss='mean_squared_error', metrics="accuracy")
            self
                    



# Read the CSV file
data = pd.read_csv("R.csv")

labels = data.copy()

feature_headings = [
    "d1","d2","d3","d4","d5","d6","First Layer","Second Layer"
]

input_features = labels[feature_headings]
output_values = labels[[c for c in labels.columns if c not in feature_headings]]
label_encoder = LabelEncoder()
input_features["First Layer"] = label_encoder.fit_transform(input_features["First Layer"])
input_features["Second Layer"] = label_encoder.transform(input_features["Second Layer"])

unique_materials = pd.unique(data[['First Layer', 'Second Layer']].values.ravel())


num_materials = len(unique_materials)
num_wavelengths = 351

input_train, input_test, output_train, output_test = train_test_split(input_features, output_values, test_size=0.2, random_state=42)
input_train, input_val, output_train, output_val = train_test_split(input_train, output_train, test_size=0.2, random_state=42)



model.compile(optimizer='adam', loss='mean_squared_error', metrics="accuracy")


history=model.fit(input_train, output_train,validation_data=(input_val, output_val), epochs=100, batch_size=32)
