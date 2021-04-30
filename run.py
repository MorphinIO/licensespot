import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# define class names, test image directory
# define image size
class_names = ['no', 'yes']
img_directory = 'dataset/demo/'
img_height = 256
img_width = 256

# load model saved from train.py
model = tf.keras.models.load_model('saved_model/licensespot_model')

# load test images
img_path = None
for file in os.listdir(img_directory):
    file_path = img_directory + file
    img_path = pathlib.Path(file_path)
    print('Testing image: ' + (file_path))
    break

#perform preprocessing onthe image
img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
# convert the image to an array that is readable by the model
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

# test the image against the network
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# print out the class name and percent confidence
print(
    "{} most likely belongs to {} with a {:.2f} percent confidence."
    .format(file_path, class_names[np.argmax(score)], 100 * np.max(score))
)