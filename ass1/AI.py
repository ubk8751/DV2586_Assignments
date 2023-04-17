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
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.applications.resnet import decode_predictions
from keras.preprocessing.image import image
from keras.preprocessing.image import img_to_array

def _train_vgg():
    pass

def _train_densenet():
    pass

def _train_resnet():
    pass

def _train_imgGPT():
    pass

def get_vgg():
    model = _train_vgg()
    return model

def get_densenet():
    model = _train_densenet()
    return model

def get_resnet():
    model = _train_resnet()
    return model

def get_imgGPT():
    model = _train_imgGPT()
    return model

def get_models():
    vgg         = get_vgg()
    densenet    = get_densenet()
    resnet      = get_resnet()
    imgGPT      = get_imgGPT()
    return vgg, densenet, resnet, imgGPT