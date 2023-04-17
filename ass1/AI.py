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

def gen_vgg():
    return 0

def gen_densenet():
    return 0

def gen_resnet():
    return 0

def gen_imgGPT():
    return 0

def gen_models():
    vgg = gen_vgg()
    densenet = gen_densenet()
    resnet = gen_resnet()
    imgGPT = gen_imgGPT()
    return vgg, densenet, resnet, imgGPT