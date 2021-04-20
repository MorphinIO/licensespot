import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# place your dataset path here
# dataset download links:
# http://www.zemris.fer.hr/projects/LicensePlates/english/results.shtml
# https://www.kaggle.com/alessiocorrado99/animals10  DO NOT USE
# https://www.kaggle.com/prasunroy/natural-images
datasetPath = "dataset"

class CNNData:
    def __init__(self, datasetPath):
        self.trainingLabel = []
        self.trainingData = []
        self.testLabel = []
        self.testData = []

        # crawl dataset path
        for dirpath, dirnames, filenames in os.walk(datasetPath):
            #whatever you want to do with these folders
            # go through each filename
            for filename in filenames:
                # make sure its an image
                count = 0;
                if (".jpg" in filename) or (".png" in filename) or (".jpeg" in filename):
                    # if from licenseplate dataset, place it in the licensePlate Set
                    # else put it in the Random set
                    newImage = CNNImage(dirpath, filename)
                    if "newlicenseplates" in dirpath:
                        if count % 5 == 0:
                            self.testData.append(newImage.image)
                            self.testLabel.append(1)
                        else:
                            self.trainingData.append(newImage.image)
                            self.trainingLabel.append(1)
                    else:
                        if count % 5 == 0:
                            self.testData.append(newImage.image)
                            self.testLabel.append(0)
                        else:
                            self.trainingData.append(newImage.image)
                            self.trainingLabel.append(0)
                    count += 1


class CNNImage:
    def __init__(self, dirpath, filename):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.filepath = dirpath + '/' + filename
        self.filename = filename



        self.image = Image.open(self.filepath).convert('RGB')
        self.image = transform(self.image)

        


if __name__ == "__main__":
    print('Hello World')

data_dir = pathlib.Path('dataset/carPhotos')

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 256
img_width = 256


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 2

data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal", 
                                                 input_shape=(img_height, 
                                                              img_width,
                                                              3)),
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.1),
  ]
)

model = Sequential([
  data_augmentation,
  layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()


epochs=7
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('saved_model/licensespot_model')