# Importing the libs to work
# Maths
import numpy as np
import matplotlib as plt

# Models
# ResNet-50
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input

# Densenet-121
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input

# VGG-19
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

# Other
import pandas as pd 
import os
from tqdm import tqdm
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.applications.resnet import decode_predictions
from keras.utils import to_categorical, img_to_array
from keras.models import Sequential

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

# Train the premade models
def _train_vgg(xt, yt, xv, yv):
    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = Sequential()
    # Add model
    model.add(vgg19)

    # Last model steps
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    fitmod = model.fit(xt, yt, epochs=20, batch_size=128, validation_data=(xv, yv))
    return model, fitmod

def _train_densenet(xt, yt, xv, yv):
    densenet = DenseNet121(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = Sequential()
    # Add model
    model.add(densenet)

    # Last model steps
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    fitmod = model.fit(xt, yt, epochs=20, batch_size=128, validation_data=(xv, yv))
    return model, fitmod

def _train_resnet(xt, yt, xv, yv):
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))
    model = Sequential()
    # Add model
    model.add(resnet)

    # Last model steps
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    fitmod = model.fit(xt, yt, epochs=20, batch_size=128, validation_data=(xv, yv))
    return model, fitmod

# Train self-made architecture
def _train_imgGPT():
    return None

#External functions for generating models
def get_vgg(xt, yt, xv, yv):
    model, fitmod = _train_vgg(xt, yt, xv, yv)
    return model, fitmod

def get_densenet(xt, yt, xv, yv):
    model, fitmod = _train_densenet(xt, yt, xv, yv)
    return model, fitmod

def get_resnet(xt, yt, xv, yv):
    model, fitmod = _train_resnet(xt, yt, xv, yv)
    return model, fitmod

def get_imgGPT(xt, yt, xv, yv):
    model, fitmod = _train_imgGPT(xt, yt, xv, yv)
    return model, fitmod

# To generate all three models at a time
def get_models(xt, yt, xv, yv):
    vgg, fitvgg        = get_vgg(xt, yt, xv, yv)
    densenet, fitDN    = get_densenet(xt, yt, xv, yv)
    resnet, fitRN      = get_resnet(xt, yt, xv, yv)
    #imgGPT, fitGPT      = get_imgGPT()
    return vgg, fitvgg, densenet, fitDN, resnet, fitRN#, imgGPT