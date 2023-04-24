# Import everything
import shutil
import os
import numpy as np
from tqdm import tqdm
from keras.utils import load_img, img_to_array, to_categorical, image_dataset_from_directory
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import pandas as pd

def create_data_set_from_directory(dir:str="./ass1/MiniDIDA.ds", vs:int=0.2):
    trainds, valds = image_dataset_from_directory(directory=dir, validation_split=vs, subset="both", seed=123, image_size=(32,32))
    return trainds, valds


# Create for sets of data
def create_data_set(input_directory:str="./ass1/DIDA", rs:int=42, ts:float=0.2, num_classes:int=10):
    paths, labels, images =  ([] for i in range(3))
    folders=os.listdir(input_directory)

    # Fetch images  
    for folder in folders:
        path=os.path.join(input_directory, folder)
        flist=os.listdir(path)
        for f in flist:
            fpath=os.path.join(path,f)        
            paths.append(fpath)
            labels.append(folder) 

    # Fancy schmancy prog bar while handling images
    for i in tqdm(range(len(paths))):
        img = load_img(paths[i], target_size=(32,32,3), grayscale=False)
        img = img_to_array(img)
        img.astype('float32')
        img = img/255
        images.append(img)
    
    # Make x, y arrays
    images = np.array(images)
    y = to_categorical(labels,num_classes=num_classes)

    #Split data into various data sets
    X_train, X_test, y_train, y_test = train_test_split(images, y, random_state=rs, test_size=ts)

    return X_train, X_test, y_train, y_test

def _get_beegest_class(input_directory:str="./ass1/MiniDIDA"):
    folders=os.listdir(input_directory)
    temp = {}
    counts = []
    # Fetch images  
    for folder in folders:
        path=os.path.join(input_directory, folder)
        flist=os.listdir(path)
        count = 0
        for f in flist:
            count += 1
        temp[folder] = count
        counts.append([folder, count])
    m = 0
    k = "0"
    for i in counts:
        if i[1] > m:
            m = i[1]
            k = str(i[0])
    del temp[k]
    return temp, m


def get_class_weights(input_directory:str="./ass1/MiniDIDA.ds"):
    class_weights = {}
    temp_dict, max_val = _get_beegest_class(input_directory)
    for key in temp_dict:
        class_weights[key] = max_val/temp_dict[key]
    
    return class_weights

# Create training tensorflow dataset
def create_tds(X_train, y_train, tds_name:str="trainds.tfds", buffer_size:int=10, batches:int=2):
    if os.path.exists(tds_name):
        trainds = tf.data.Dataset.load(tds_name)
        return trainds
    trainds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    trainds = trainds.shuffle(buffer_size=buffer_size).batch(batches)
    trainds.save(tds_name)
    return trainds

# Create validation tensorflow dataset
def create_vds(X_test, y_test, vds_name:str="validationds.tfds", buffer_size:int=10, batches:int=2):
    if os.path.exists(vds_name):
        valds = tf.data.Dataset.load(vds_name)
        return valds
    valds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    valds = valds.shuffle(buffer_size=buffer_size).batch(batches)
    valds.save(vds_name)
    return valds

def remove(path:str="/TrainingDataSet.tfds"):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f'Removed {path}')

if __name__ == "__main__":
    create_data_set_from_directory()