import tensorflow as tf
import os

from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
datasetPath = "/Users/michael/documents/dev/licensespot/dataset"

class CNNData:
    def __init__(datasetPath):
        self.licensePlates = [];
        self.randomData = [];

        for dirpath, dirnames, filenames in os.walk(dataset):
            #whatever you want to do with these folders
            for filename in filenames:
                if "licenseplates" in dirpath:
                    licensePlates.append(filename);
                else:
                    randomData.append(filename);


if __name__ == "__main__":
    print('Hello World')

