# conda install tensorflow
import tensorflow as tf
import os
# conda install pillow
from PIL import Image
# conda install torchvision
from torchvision.transforms import ToTensor
from torchvision import transforms

from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt

# place your dataset path here
# dataset download links:
# http://www.zemris.fer.hr/projects/LicensePlates/english/results.shtml
# https://www.kaggle.com/alessiocorrado99/animals10  DO NOT USE
# https://www.kaggle.com/prasunroy/natural-images
datasetPath = "/Users/michael/documents/dev/licensespot/dataset"

class CNNData:
    def __init__(self, datasetPath):
        self.testLicensePlates = []
        self.testRandomData = []
        self.trainingLicensePlates = []
        self.trainingRandomData = []

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
                    if "licenseplates" in dirpath:
                        if count % 5 == 0:
                            self.testLicensePlates.append(CNNImage(dirpath, filename))
                        else:
                            self.trainingLicensePlates.append(CNNImage(dirpath, filename))
                    else:
                        if count % 5 == 0:
                            self.testRandomData.append(CNNImage(dirpath, filename))
                        else:
                            self.trainingRandomData.append(CNNImage(dirpath, filename))


class CNNImage:
    def __init__(self, dirpath, filename):
        transform = transforms.Compose([
            transforms.Resize(256)
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.filepath = dirpath + '/' + filename
        self.filename = filename
        self.isPlate = 0 # not a license plate
        if "licenseplates" in dirpath:
            self.isPlate = 1 # is a license plate

        print(self.filepath)
        # convert to tensor and output
        self.image = Image.open(self.filepath)
        self.image = transform(self.image)
        #self.image = ToTensor()(self.image).unsqueeze(0) # unsqueeze to add artificial first dimension
        print(self.image)




if __name__ == "__main__":
    print('Hello World')

Data = CNNData(datasetPath)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

