# Maths
import numpy as np
import matplotlib as plt

# Models
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg19 import VGG19

# Other
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet import decode_predictions
from keras.utils import to_categorical, img_to_array
from keras.models import Sequential

from imgGPT import ImgGPT

# Evaluate model
def evaluate(model, hist, xv, yv):
    ret = {
        "score":    model.evaluate(xv, yv),
        "accuracy": hist.history["accuracy"], 
        "val_acc":  hist.history["val_accuracy"],
        "loss":     hist.history["loss"],
        "val_loss": hist.history["val_loss"]
    }
    return ret

# Train model
def _build_model(mod):
    model = Sequential()
    model.add(mod)
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

# Fit model
def fit_model(mod, xt, yt, xv, yv):
    fitmod = mod.fit(xt, yt, epochs=20, batch_size=128, validation_data=(xv, yv))
    return fitmod

# External for generating singular models
def get_vgg():
    mod = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = _build_model(mod)
    return model

def get_densenet():
    mod = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = _build_model(mod)
    return model

def get_resnet():
    mod = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = _build_model(mod)
    return model

def get_imgGPT():
    mod = ImgGPT(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = _build_model(mod)
    return model

# To generate all three models at a time
def get_models():
    vgg        = get_vgg()
    densenet   = get_densenet()
    resnet     = get_resnet()
    #imgGPT      = get_imgGPT()
    return vgg, densenet, resnet#, imgGPT