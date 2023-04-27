# Import libs
from keras.optimizers import Adam
from keras.layers import Conv2D,MaxPool2D,Dropout,BatchNormalization,Dense
import tensorflow as tf

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
    
    def fit(self, trainds, valds:tf.data.Dataset, epochs:int=20, batch_size:int=128):
        return self._model.fit(trainds, validation_data=valds, epochs=epochs, batch_size=batch_size) 
    
    def compile(self, opt=Adam(learning_rate=0.000001), loss="categorical_crossentropy"):
        self._model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "TrueNegatives", "TruePositives", "FalseNegatives", "FalsePositives"])
    
    def summary(self):
        return self._model.summary()
    
    def evaluate(self, valds, hist, model, batch_size:int=128):
        ret = {
            "eval":     model.evaluate(valds, batch_size=128),
            "accuracy": hist.history["accuracy"][-1], 
            "val_acc":  hist.history["val_accuracy"][-1],
            "loss":     hist.history["loss"][-1],
            "val_loss": hist.history["val_loss"][-1],
            "f1_score": _f1_score(hist=hist)
        }
        return ret        
    
    @property
    def model(self):
        return self._model
def _f1_score(hist):
    tp = hist.history["true_positives"][-1]
    fp = hist.history["false_positives"][-1]
    fn = hist.history["false_negatives"][-1]
    if tp == 0.0 or fp == 0.0 or fn == 0:    
        if tp == 0.0:
            print("Model has 0 true positives")
        if fp == 0.0:
            print("Model has 0 false positives")
        if fn == 0.0:
            print("Model has 0 false negatives")
        if (tp == 0.0 and fp == 0.0) or (tp == 0 and fn == 0):
            return 0
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 2*((precision*recall)/(precision+recall))