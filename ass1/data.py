# Import everything
import os
import numpy as np
from tqdm import tqdm
from keras.utils import load_img,  img_to_array, to_categorical
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

# Support function for splitting data, not really nescessary, but it is what it is
def split_data(images, labels):
    y = to_categorical(labels,num_classes=10)
    X_train, X_test, y_train, y_test = train_test_split(images, y, random_state=42, test_size=0.2)
    return X_train, X_test, y_train, y_test