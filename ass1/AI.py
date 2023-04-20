# Models
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.vgg19 import VGG19

# Other
from keras.layers import *
from keras.callbacks import *
from keras.models import Sequential
from keras.metrics import TruePositives, TrueNegatives, FalseNegatives, FalsePositives
import eval

from imgGPT import ImgGPT

# Evaluate model
def evaluate(model, hist, xv, yv):
    ret = eval.evaluate(model, hist, xv, yv)
    return ret

# Train model
def _build_model(mod):
    model = Sequential()
    model.add(mod)
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy', "TrueNegatives", "TruePositives", "FalseNegatives", "FalsePositives"])
    return model

# Fit model
def fit_model(mod, xt:np.array, yt:np.array, xv:np.array, yv:np.array, epochs:int=20, batch_size:int=128):
    fitmod = mod.fit(xt, yt, epochs=epochs, batch_size=batch_size, validation_data=(xv, yv))
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
    mod = Sequential()
    model = ImgGPT(mod, input_shape=(32,32,3))
    model.model.add(Flatten())
    model.model.add(Dense(10, activation='softmax'))
    model.compile(opt='sgd', loss="categorical_crossentropy")
    #model.build((1,32,32,3))
    return model

# To generate all three base models at a time
def get_models():
    vgg        = get_vgg()
    densenet   = get_densenet()
    resnet     = get_resnet()
    return vgg, densenet, resnet