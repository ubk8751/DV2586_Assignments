# Self-made modules
from AI import get_imgGPT, get_densenet, get_vgg, get_resnet, fit_model, get_models, evaluate
from data import load_images, image_files, split_data

# External libs
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
from keras.applications.resnet import decode_predictions
from keras.utils import load_img,  img_to_array, to_categorical
from keras.models import Sequential


# Run the program
if __name__ == "__main__":
    data_path = "./ass1/MiniDIDA" # Might need t ochange when running on other computers
    filepaths, labels = image_files(data_path)
    images = load_images(filepaths)

    # SPlit data
    X_train, X_test, y_train, y_test = split_data(images, labels)

    #Train models
    #vgg, densenet, resnet = get_models() # imgGPT

    # Fit models
    #fit_models = {}
    #for mod in [vgg, densenet, resnet]:
    #    fit_models[mod] = fit_model(mod, X_train, y_train, X_test, y_test)

    # Evaluate models
    #vgg_stat        = evaluate(vgg, fit_models[vgg], X_test, y_test)
    #densenet_stat   = evaluate(densenet, fit_models[densenet], X_test, y_test)
    #resnet_stat     = evaluate(resnet, fit_models[resnet], X_test, y_test)

    imggpt = get_imgGPT()
    fitimggpt = imggpt.fit(xt=X_train, yt=y_train, xv=X_test, yv=y_test, epochs=20, batch_size=128)

    imggpt.summary()
