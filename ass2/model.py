import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers, regularizers
from keras.models import Model, Sequential
from matplotlib import pyplot as plt

class AnomalyDetector():
    def __init__(self, train_data, SL:int=5, opt:str="adam", loss:str="mae"):
        self._optimizer = opt
        self._loss = loss
        self._sequence_length = SL
        self._model = self._init_model(train_data=train_data)

    def _init_model(self, train_data):
        model = Sequential()
        model.add(layers.LSTM(128, activation="relu", return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.00), input_shape=(5,3)))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(layers.LSTM(32, activation="relu", return_sequences=False))
        model.add(layers.RepeatVector(5))
        model.add(layers.LSTM(32, activation="relu", return_sequences=True))
        model.add(layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(layers.LSTM(128, activation="relu", return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(3)))
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=["accuracy"])
        model.build((None,self._sequence_length, 3))
        model.summary()
        return model

    def evaluate_model(hist):
        pass

    def fit_model(self, train, validation, batch_size:int=4, epochs:int=5):
        return self.model.fit(train, validation_data=validation, batch_size=batch_size, epochs=epochs)
    
    @property
    def model(self):
        return self._model


        