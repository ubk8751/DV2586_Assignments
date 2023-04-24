from keras.applications.vgg16 import VGG16
from keras.utils import image_dataset_from_directory

#train, validation = image_dataset_from_directory(directory="./ass1/MiniDIDA.ds", image_size=(224,224), validation_split=0.2, subset="validation",seed=123)
validation = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  validation_split=0.2,
  subset="validation",
  seed=42,
  image_size=(64,64),
  shuffle=True,
  batch_size=32)

train = image_dataset_from_directory(
  directory="./ass1/MiniDIDA.ds",
  validation_split=0.2,
  subset="training",
  seed=42,
  image_size=(64,64),
  shuffle=True,
  batch_size=32)

Vgg16 = VGG16(weights=None, classes=10, input_shape=(64,64,3))
Vgg16.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=['accuracy', 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
Vgg16.fit(train, validation_data=validation, epochs=5, batch_size=128)