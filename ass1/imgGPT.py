# Import libs
import numpy as np
from keras.optimizers import Adam,SGD
from keras.layers import Conv2D,MaxPool2D,Dropout
from keras.models import Model
from keras.losses import sparse_categorical_crossentropy

class ImgGPT(Model):
    def __init__(self, model, input_shape=(32,32,3)):
        super().__init__()
        self._model = model
        self._model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=input_shape))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(Conv2D(32, 3, padding="same", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(Conv2D(64, 3, padding="same", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))
        self._model.add(Dropout(0.4))
    
    def fit(self, xt:np.array, yt:np.array, xv:np.array, yv:np.array, epochs:int=20, batch_size:int=128):
        return self._model.fit((xt,yt), validation_data=(xv, yv), epochs=epochs, batch_size=batch_size)
    
    def mcompile(self, opt=Adam(lr=0.000001)):
        self._model().compile(optimizer=opt, loss=sparse_categorical_crossentropy(from_logits=True), metrics=["accuracy"])

    @property
    def model(self):
        return self._model
        
