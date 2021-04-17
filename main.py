# conda install tensorflow
import tensorflow as tf
import os
# conda install pillow
from PIL import Image
# conda install torchvision
from torchvision.transforms import ToTensor

from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt

# place your dataset path here
# dataset download links:
# http://www.zemris.fer.hr/projects/LicensePlates/english/results.shtml
# https://www.kaggle.com/alessiocorrado99/animals10
# https://www.kaggle.com/prasunroy/natural-images
datasetPath = "dataset"

class CNNData:
    def __init__(self, datasetPath):
        self.licensePlates = []
        self.randomData = []

        # crawl dataset path
        for dirpath, dirnames, filenames in os.walk(datasetPath):
            #whatever you want to do with these folders
            # go through each filename
            for filename in filenames:
                # make sure its an image
                if (".jpg" in filename) or (".png" in filename) or (".jpeg" in filename):
                    # if from licenseplate dataset, place it in the licensePlate Set
                    # else put it in the Random set
                    if "licenseplates" in dirpath:
                        self.licensePlates.append(CNNImage(dirpath, filename))
                    else:
                        self.randomData.append(CNNImage(dirpath, filename))


class CNNImage:
    def __init__(self, dirpath, filename):
        self.filepath = dirpath + '/' + filename
        self.filename = filename
        self.isPlate = 0 # not a license plate
        if "licenseplates" in dirpath:
            self.isPlate = 1 # is a license plate

        print(self.filepath)
        # convert to tensor and output
        self.image = Image.open(self.filepath).convert('LA') #convert image to grayscale
        # converts the image size to given ratio
        self.image = self.image.resize((400,400))
        print(self.image.size)
        self.image = ToTensor()(self.image).unsqueeze(0) # unsqueeze to add artificial first dimension
        print(self.image)
        


if __name__ == "__main__":
    print('Hello World')

test = CNNData(datasetPath)

