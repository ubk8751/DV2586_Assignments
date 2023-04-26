from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import image_dataset_from_directory, to_categorical
from keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from data2 import create_data

RESIZE_IMAGE_DIMESIONS = (64, 64)
BATCH_SIZE = 2
SHUFFLE_BUFFER_SIZE = 1024

@tf.function
def scale_values(image, label):
    image = image / 255.0
    return image, label

@tf.function
def resize_image(image, label):
    image = tf.image.resize(image, RESIZE_IMAGE_DIMESIONS)
    return image, label

@tf.function
def to_grayscale(image, label):
    image = tf.image.rgb_to_grayscale(image)
    return image, label

#train, val = create_data(path="./ass1/MiniDIDA.ds")
#train, validation = image_dataset_from_directory(directory="./ass1/MiniDIDA.ds", image_size=(224,224), validation_split=0.2, subset="validation",seed=123)
val = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  label_mode="categorical",
  validation_split=0.2,
  subset="validation",
  image_size=RESIZE_IMAGE_DIMESIONS,
  shuffle=False,
  batch_size=None)

train = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  label_mode="categorical",
  validation_split=0.2,
  subset="training",
  image_size=RESIZE_IMAGE_DIMESIONS,
  shuffle=False,
  batch_size=None)

train = (
        train
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )

val = (
        val
        .map(scale_values)
        .map(to_grayscale)
        .shuffle(SHUFFLE_BUFFER_SIZE)
        .batch(BATCH_SIZE)
    )
# train_ds = train.unbatch()
# xt = list(train_ds.map(lambda x, y: x))
# yt = list(train_ds.map(lambda x, y: y))
# # val_ds = val.unbatch()
# # xv = list(val_ds.map(lambda x, y: x))
# # yv = list(val_ds.map(lambda x, y: y))
# #yt = to_categorical(yt,num_classes=10)
# xt = np.array(xt)
# # xv = np.array(xv)
# #yt = np.array(yt)
# yt = np.argmax(yt, axis=1)
# print(yt)
# #yv = np.argmax(yv, axis=1)
Vgg19 = VGG19(weights=None, classes=10, input_shape=(64,64,1))
Vgg19.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy', 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
#Vgg19.predict

Vgg19.fit(train, validation_data=val, epochs=5, batch_size=128)