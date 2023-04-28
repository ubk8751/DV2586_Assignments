import os
from tqdm import tqdm
import tensorflow as tf
import shutil
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

image_dimensions = (64, 64)
batch_size = 2
shuffle_buffer = 1024

@tf.function
def scale_values(image, label):
    image = image / 255.0
    return image, label

@tf.function
def resize_image(image, label):
    image = tf.image.resize(image, image_dimensions)
    return image, label

@tf.function
def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

def create_data(path : str):
    if not os.path.exists(path):
        raise Exception(f'Cannot find path to images: "{path}"')
    print(f'Found folder with images: "{path}"')

    dataset, validation_data = tf.keras.utils.image_dataset_from_directory(
        path,
        label_mode='categorical', 
        shuffle=False,
        batch_size=None, 
        image_size=image_dimensions,
        validation_split=0.3,
        subset='both'
        )

    
    validation_data = (
        # dataset.take(20)
        validation_data
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(shuffle_buffer)
        .batch(batch_size)
    )

    dataset = (
        dataset
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(shuffle_buffer)
        .batch(batch_size)
    )
    return dataset, validation_data

def remove(path:str="/TrainingDataSet.tfds"):
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        print(f'Removed {path}')

def train_val_dataset(dataset, val_split=0.3):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def print_stat_row(stat, name):
    print(f'{name:<15} {stat["accuracy"]:<15.5f} {stat["val_acc"]:<15.5f} {stat["loss"]:<15.5f} {stat["val_loss"]:<15.5f} {stat["TP"]:<15.5f} {stat["FP"]:<15.5f} {stat["TN"]:<15.5f} {stat["FN"]:<15.5f} {stat["F1"]:<15.5f}'.format(name,stat["accuracy"],stat["val_acc"],stat["loss"],stat["val_loss"],stat["TP"],stat["FP"],stat["TN"],stat["FN"],stat["F1"]))