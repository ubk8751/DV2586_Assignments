from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import image_dataset_from_directory, to_categorical
import tensorflow as tf
import numpy as np


#train, validation = image_dataset_from_directory(directory="./ass1/MiniDIDA.ds", image_size=(224,224), validation_split=0.2, subset="validation",seed=123)
validation = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  label_mode="categorical",
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(32,32),
  shuffle=True,
  batch_size=32)

train = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  label_mode="categorical",
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(32,32),
  shuffle=True,
  batch_size=32)

train_ds = train.unbatch()
x = list(train_ds.map(lambda x, y: x))
y = list(train_ds.map(lambda x, y: y))
#y = to_categorical(y,num_classes=10)
#print(len(y))
x = np.array(x)
y = tf.stack(y)
Vgg19 = Sequential()
Vmod = VGG19(weights=None, include_top=False, classes=10, input_shape=(32,32,3))
Vgg19.add(Vmod)
Vgg19.add(Flatten())
Vgg19.add(Dense(10, activation='softmax'))
Vgg19.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy', 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
Vgg19.fit(x,y, validation_data=validation, epochs=5, batch_size=128)