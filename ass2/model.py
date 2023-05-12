import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from keras import layers, regularizers
from keras.models import Model, Sequential
from keras.metrics import Accuracy
from matplotlib import pyplot as plt

tf.config.run_functions_eagerly(True)

def supercoolmetric(y_true, y_pred):
        true_stat = 0
        pred_stat = 0
        acc = 0
        for i in range(len(y_true[0])):
            ts = sum(y_true[0][i].numpy()[3:-1])
            ps = sum(y_pred[0][i].numpy()[3:-1])
        return 1

class AnomalyDetector():
    def __init__(self, train_data, SL:int=5, opt:str="adam", loss:str="mse"):
        self._optimizer = opt
        self._loss = loss
        self._sequence_length = SL
        self._model = self._init_model()

    def _init_model(self):
        model = Sequential()
        model.add(layers.LSTM(128, activation="tanh", return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.00), input_shape=(5,4)))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(layers.LSTM(32, activation="tanh", return_sequences=False))
        model.add(layers.RepeatVector(5))
        model.add(layers.LSTM(32, activation="relu", return_sequences=True))
        model.add(layers.LSTM(64, activation="tanh", return_sequences=True))
        model.add(layers.LSTM(128, activation="relu", return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(4)))
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=[ supercoolmetric, "accuracy" ])
        model.build((None,self._sequence_length, 3))
        model.summary()
        return model

    def evaluate_model(hist):
        pass

    def fit_model(self, train, validation, batch_size:int=4, epochs:int=5):
        return self.model.fit(train, validation_data=validation, batch_size=batch_size, epochs=epochs)
    
    def predict_model(self, data, batch_size:int=4):
        return self._model.predict(x=data,batch_size=batch_size)
        

    @property
    def model(self):
        return self._model


        