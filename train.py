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
# https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper
datasetPath = "dataset"


# this was used as apart of our previous implementation
# its purpose was to scrape our dataset directory and 
# place them into code as well as label them
# this went unused in the final implentation, hense why it is commented out

# class CNNData:
#     def __init__(self, datasetPath):
#         self.trainingLabel = []
#         self.trainingData = []
#         self.testLabel = []
#         self.testData = []

#         # crawl dataset path
#         for dirpath, dirnames, filenames in os.walk(datasetPath):
#             #whatever you want to do with these folders
#             # go through each filename
#             for filename in filenames:
#                 # make sure its an image
#                 count = 0;
#                 if (".jpg" in filename) or (".png" in filename) or (".jpeg" in filename):
#                     # if from licenseplate dataset, place it in the licensePlate Set
#                     # else put it in the Random set
#                     newImage = CNNImage(dirpath, filename)
#                     if "newlicenseplates" in dirpath:
#                         if count % 5 == 0:
#                             self.testData.append(newImage.image)
#                             self.testLabel.append(1)
#                         else:
#                             self.trainingData.append(newImage.image)
#                             self.trainingLabel.append(1)
#                     else:
#                         if count % 5 == 0:
#                             self.testData.append(newImage.image)
#                             self.testLabel.append(0)
#                         else:
#                             self.trainingData.append(newImage.image)
#                             self.trainingLabel.append(0)
#                     count += 1


# class CNNImage:
#     def __init__(self, dirpath, filename):
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])
#         self.filepath = dirpath + '/' + filename
#         self.filename = filename



#         self.image = Image.open(self.filepath).convert('RGB')
#         self.image = transform(self.image)

        


# defining the directory to our datast
data_dir = pathlib.Path('dataset/carPhotos')

# get the total count of images in our dataset directory 
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# define the batch size of the model, as well as the size of the images
batch_size = 32
img_height = 256
img_width = 256

# make a call to the keras api to preprocess the data in our dataset directory
# It looks in our directory and then resizes each image to a 256x256, it also
# sets a validation split where 0.8 of the images are placed into the training set 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#another call to the keras api to preproces the data in our dataset directory
# It looks in our directory and does the same thing as above, except it uses a validation
# split where 0.2 of the images are placed into the validation set.
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# check the class names from our dataset
# should be yes and no
#class_names = train_ds.class_names
#print(class_names)

# check the shape of the images and labels
#for image_batch, labels_batch in train_ds:
#  print(image_batch.shape)
# print(labels_batch.shape)
#  break

# define a normalization layer the processed images can be passed to
# for changing RGB values from 0 - 255 to 0 - 1
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# pass the dataset through the normalization layer
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# Notice the pixels values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))

# define the numbers of classes
num_classes = 2

# define a data augmentation layer with preprocessing on the images for randomly flipping images
# horizontally, aswell as a random rotation and zoom. This will help with accuracy and overfitting
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

# Here we define our model. It is a sequential model utilizing convolution2D,
# maxpooling2D and dense layers. We pass it the data_augmentation layer first for preprocessing
# we also define the input layer and the shape of data it will accept. Here it is a 256x256 image array containing 3 values RGB
# We then define the convolutional layers, with maxpooling layers following each one. 
# The conv2d layer is fed a filter amount, kernel_size, padding and activation function
# we increase the number of filters with more conv2D layer for accuracy and keeping the same kernel_size for
# each layer. We also have same padding keyword and the activation function relu
# we Then flatten our layers inputs to 1 dimension and add two dense (deep layers)
# The final dense layer is our output layer and will output yes or no
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

# Here we compile our model witht he optimizer adam and the loss function provided by the 
# keras API. We Also want accuracy metrics from our model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# print the summary of layers on our model
model.summary()


# define the number of epochs, this number is decided experiementally
epochs=7

# train our model with our training dataset and test it with the validation dataset
# over a given number of epochs
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# record accuracy from the model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
# record loss from model
loss = history.history['loss']
val_loss = history.history['val_loss']

# All for graphing our accuracy and loss over the given number of epochs
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

# save the model for use in run.py, and testing single images
model.save('saved_model/licensespot_model')