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
                    if "licenseplates" in dirpath:
                        if count % 5 == 0:
                            self.testData.append(newImage.image)
                            self.testLabel.append(1)
                        else:
                            self.trainingData.append(newImage.image)
                            self.testLabel.append(1)
                    else:
                        if count % 5 == 0:
                            self.testData.append(newImage.image)
                            self.testLabel.append(0)
                        else:
                            self.trainingData.append(newImage.image)
                            self.testLabel.append(0)


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
        self.isPlate = 0 # not a license plate
        if "newlicenseplates" in dirpath:
            self.isPlate = 1 # is a license plate

        #print(self.filepath)
        # convert to tensor and output

        self.image = Image.open(self.filepath).convert('RGB')
        self.image = transform(self.image)

        #print(self.image)
        print(self.isPlate)
        


if __name__ == "__main__":
    print('Hello World')

Data = CNNData(datasetPath)

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

history = model.fit(Data.trainingData, Data.trainingLabels, epochs=10, 
                    validation_data=(Data.testData, Data.testLabels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
