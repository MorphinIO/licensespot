import pathlib
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

class_names = ['no', 'yes']
img_directory = 'dataset/demo/'
img_height = 256
img_width = 256

model = tf.keras.models.load_model('saved_model/licensespot_model')

img_path = None
for file in os.listdir(img_directory):
    file_path = img_directory + file
    img_path = pathlib.Path(file_path)
    print('Testing image: ' + (file_path))
    break

img = keras.preprocessing.image.load_img(
    img_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "{} most likely belongs to {} with a {:.2f} percent confidence."
    .format(file_path, class_names[np.argmax(score)], 100 * np.max(score))
)