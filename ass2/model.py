########################################################################################
#
# DV2586: Deep Machine Learning
# Assignment 2
# Model module
#
# Code by:
#   Samuel Jonsson
#   DVAMI19h
#
#########################################################################################

# Import entire packages
import numpy as np
import tensorflow as tf

# Import seperate functions and classes
from keras import layers, regularizers
from keras.models import Sequential

# I needed this for something, but not sure if I can remove it without accidentally destroy the code
tf.config.run_functions_eagerly(True)

class AnomalyDetector():
    def __init__(self, train_data, SL:int=5, opt:str="adam", loss:str="mse"):
        self._optimizer = opt
        self._loss = loss
        self._sequence_length = SL
        self._model = self._init_model()

    def _init_model(self):
        # Create the model
        model = Sequential()
        model.add(layers.LSTM(32, activation="tanh", return_sequences=True,
                               kernel_regularizer=regularizers.l2(0.00), input_shape=(5,4)))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(128, activation="tanh", return_sequences=False))
        model.add(layers.BatchNormalization())
        
        model.add(layers.RepeatVector(5))
        model.add(layers.LSTM(64, activation="relu", return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(32, activation="tanh", return_sequences=True))
        model.add(layers.BatchNormalization())
        model.add(layers.LSTM(128, activation="relu", return_sequences=True))
        model.add(layers.TimeDistributed(layers.Dense(4, activation='sigmoid')))
        
        # Compile the model
        print("Compiling model...")
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=[])
        
        # Present model summary
        model.build((None,self._sequence_length, 5))
        model.summary()
        return model

    # Anomaly htreshold for each Bearing is the MAE value that Bearing.
    def create_thresholds(self, val, data_actual, batch_size:int=4):
        preds = self._predict_model(data=val, batch_size=batch_size)
        e = np.abs(preds - data_actual)
        errors = e[0,:,:]
        error = np.concatenate([errors,e[1:,-1,:]])
        thresholds=[0,0,0,0]
        for item in error:
            for i in range(len(item.tolist())):
                thresholds[i] += item[i]
        for i in range(len(thresholds)):
            thresholds[i] /= len(error)
            thresholds[i] += thresholds[i] * 0.5
        return thresholds

    def fit_model(self, train, validation, batch_size:int=4, epochs:int=5):
        return self.model.fit(train, train, validation_data=(validation, validation), batch_size=batch_size, epochs=epochs)
    
    def _predict_model(self, data, batch_size:int=4):
        return self._model.predict(x=data,batch_size=batch_size)
        
    # Easy access to model
    @property
    def model(self):
        return self._model


        