from AI import get_models, get_imgGPT, get_densenet, get_vgg, get_resnet, evaluate

import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import *
from sklearn.model_selection import train_test_split
from keras.callbacks import *
from keras.applications.resnet import decode_predictions
from keras.utils import load_img,  img_to_array, to_categorical
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# Retrieve the iamges and labels (folder names 0-9) 
def image_files(input_directory):
    paths=[]
    labels=[]
    
    folders=os.listdir(input_directory)
    #print(digit_folders)
    
    for folder in folders:
        path=os.path.join(input_directory, folder)
        flist=os.listdir(path)
        for f in flist:
            fpath=os.path.join(path,f)        
            paths.append(fpath)
            labels.append(folder) 
    return paths,labels

# Load the images from the stored data paths
def load_images(filepaths):
    images = []
    for i in tqdm(range(len(filepaths))):
        img = load_img(filepaths[i], target_size=(32,32,3), grayscale=False)
        img = img_to_array(img)
        img.astype('float32')
        img = img/255
        images.append(img)

    images = np.array(images)
    return images

# Support function for splitting data, not really nescessary, but it is waht it is
def split_data(data):
    y = to_categorical(labels,num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(images, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test


# Run the program
if __name__ == "__main__":
    data_path = "./ass1/MiniDIDA" # Might need t ochange when running on other computers
    filepaths, labels = image_files(data_path)
    images = load_images(filepaths)

    # SPlit data
    X_train, X_test, y_train, y_test = split_data(images)

    #Train models
    vgg, fitvgg, densenet, fitDN, resnet, fitRN = get_models(X_train, y_train, X_test, y_test) # imgGPT

    # Evaluate models
    vgg_stat        = evaluate(vgg, fitvgg, X_test, y_test)
    densenet_stat   = evaluate(densenet, fitDN, X_test, y_test)
    resnet_stat     = evaluate(resnet, fitRN, X_test, y_test)

    print(vgg_stat)
    print(densenet_stat)
    print(resnet_stat)