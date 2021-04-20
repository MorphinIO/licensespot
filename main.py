# conda install tensorflow
import tensorflow as tf
import os
# conda install pillow
from PIL import Image
# conda install torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pathlib
#import matplotlib.pyplot as plt

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
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.filepath = dirpath + '/' + filename
        self.filename = filename


        #print(self.filepath)
        # convert to tensor and output

        self.image = Image.open(self.filepath).convert('RGB')
        self.image = transform(self.image)

        #print(self.image)
        #print(self.isPlate)
        


if __name__ == "__main__":
    print('Hello World')

data_dir = tf.keras.utils.get_file('carPhotos')
data_dir = pathlib.Path(data_dir)

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


model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(Data.trainingData, Data.trainingLabel, epochs=10, 
                    validation_data=(Data.testData, Data.testLabel))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(Data.testData,  Data.testLabel, verbose=2)

print(test_acc)
