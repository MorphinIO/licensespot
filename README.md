# licensespot
A NN to detect a license plate in a given piece of data


 

1. python project anaconda environment with tensorflow  

2. import dataset  

3. figure out how to express an image as a tensor -> then convert the images to greyscale and express all images as an array of single dimensional greyscale tensors 

4. Perform preprocessing on image data to compress all images to a give size  

5. Perform normalization on greyscale tensors to give us a value between 0–1 rather than 0 - 255  

6. build model using our image tensor as input 

5. split dataset into 2/3 training data and 1/3 test data – RANDOMLY 

6. train model on training data 

8. evaluate model on test data 

9. write final report  

 

if we are detecting yes/no the license plate is there, we only need 1 NN. If we are detecting the text on the license plate, we need 2 NN, one for verification, one for detecting the characters. The later will be trained on license-plate-font images of random text (7 characters). The first will be trained on a dataset containing images with or without license plates. Utilizing these images as tensors for input on the models. 

 
