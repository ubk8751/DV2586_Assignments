# Import libs
from keras.optimizers import Adam
from keras.layers import Conv2D,MaxPool2D,Dropout,BatchNormalization,Dense
from keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
import tensorflow as tf
from eval import evaluate
from data import get_class_weights

class ImgGPT():
    def __init__(self, model, input_shape=(32,32,3)):
        self._model = model
        self._model.add(BatchNormalization())
        self._model.add(Dense(32,activation="relu"))
        self._model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=input_shape))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(Conv2D(32, 3, padding="same", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(Conv2D(64, 3, padding="same", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))
        self._model.add(Dense(32,activation="relu"))
        self._model.add(Dropout(0.4))
    
    def fit(self, trainds:tf.data.Dataset, valds:tf.data.Dataset, epochs:int=20, batch_size:int=128, path:str="./ass1/DIDA.ds"):
        return self._model.fit(trainds, validation_data=valds, epochs=epochs, batch_size=batch_size) 
    
    def compile(self, opt=Adam(learning_rate=0.000001), loss="categorical_crossentropy"):
        self._model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "TrueNegatives", "TruePositives", "FalseNegatives", "FalsePositives"])

    def build(self, input_shape=(1, 32,32,3)):
        self._model.build(input_shape)
        return self._model
    
    def summary(self):
        return self._model.summary()
    
    def evaluate(self, xv, yv, batch_size:int=128):
        return self._model.evaluate(xv,yv, batch_size=batch_size)        

    @property
    def model(self):
        return self._model
        
