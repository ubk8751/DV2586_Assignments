# Import libs
from keras.optimizers import Adam
from keras.layers import Conv2D,MaxPool2D,Dropout,BatchNormalization,Dense,Flatten
import tensorflow as tf
from keras.models import Sequential

class ImgGPT():
    def __init__(self, input_shape=(32,32,3)):
        self._model = Sequential()
        self._model.add(BatchNormalization())
        self._model.add(Dense(32,activation="relu"))
        self._model.add(Conv2D(32,1,padding="valid", activation="relu", input_shape=input_shape))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(BatchNormalization())
        self._model.add(Conv2D(32, 1, padding="valid", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(BatchNormalization())
        self._model.add(Conv2D(64, 1, padding="valid", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(BatchNormalization())
        self._model.add(Conv2D(128, 1, padding="valid", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))

        self._model.add(BatchNormalization())
        self._model.add(Conv2D(64, 1, padding="valid", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))
        
        self._model.add(BatchNormalization())
        self._model.add(Conv2D(32, 1, padding="valid", activation="relu"))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='valid', data_format=None))
        self._model.add(Flatten())
        self._model.add(Dropout(0.4))
        self._model.add(Dense(10, activation='softmax'))
    
    def fit(self, trainds, valds:tf.data.Dataset, epochs:int=10, batch_size:int=1024):
        return self._model.fit(trainds, validation_data=valds, epochs=epochs, batch_size=batch_size) 
    
    def compile(self, opt=Adam(learning_rate=0.0000001), loss="categorical_crossentropy"):
        self._model.compile(optimizer=opt, loss=loss, metrics=["accuracy", "TrueNegatives", "TruePositives", "FalseNegatives", "FalsePositives"])
    
    def summary(self):
        return self._model.summary()
    
    def vgg_evaluate(self, valds, hist, model):
        ret = {
            "eval"    : model.evaluate(valds),
            "accuracy": hist.history["accuracy"][-1], 
            "val_acc" : hist.history["val_accuracy"][-1],
            "loss"    : hist.history["loss"][-1],
            "val_loss": hist.history["val_loss"][-1],
            "TP"      : hist.history["true_positives"][-1],
            "FP"      : hist.history["false_positives"][-1],
            "TN"      : hist.history["true_negatives"][-1],
            "FN"      : hist.history["false_negatives"][-1],
            "F1"      : _f1_score(hist=hist)
        }
        return ret

    def evaluate(self, valds):
        return self._model.evaluate(valds)     
    
    @property
    def model(self):
        return self._model

def get_imgGPT():
    print("\nCreating imgGPT")
    model = ImgGPT(input_shape=(64,64,1))
    model.compile(opt='sgd', loss="categorical_crossentropy")
    return model

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